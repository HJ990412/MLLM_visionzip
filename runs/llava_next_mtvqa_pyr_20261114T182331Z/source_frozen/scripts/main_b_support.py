"""Common experiment receipts; production attention and selection remain unchanged."""
from pathlib import Path
import os,json,time,hashlib,uuid,importlib.util,sys
from mmimpress.prefix25 import sha,inventory,process_io

def source_hashes(root):
    paths=list((root/'mmimpress').rglob('*.py'))+[p for p in (root/'scripts').glob('*.py') if p.name!='113_report_mt_gqa_B.py']+list((root/'tests').glob('test_*.py'))
    return {str(p.relative_to(root)):sha(p) for p in sorted(paths)}

def publish(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    for p in (Path(path).parent,Path(path).parent.parent):
        fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

def complete(path,records,name):
    publish(path/name,records)
    records=dict(records);p=path/name;records[name]={'size':p.stat().st_size,'allocated_bytes':p.stat().st_blocks*512,'sha256':sha(p)}
    return records

def persist_llava(call,path,kind,source):
    start=time.perf_counter();before=process_io();receipt=call();writer_ms=(time.perf_counter()-start)*1000
    t=time.perf_counter();records=inventory(path);hash_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter();files=complete(path,records,'completion.json' if kind=='mpic' else 'integrity.json');seal_ms=(time.perf_counter()-t)*1000
    elapsed=(time.perf_counter()-start)*1000;after=process_io()
    receipt.update(event_id=str(uuid.uuid4()),event_kind='physical_build',source=source,files=files,service_persistence_ms=elapsed,
        native_writer_wall_ms=writer_ms,completion_full_hash_ms=hash_ms,completion_publication_ms=seal_ms,
        os_write_bytes=after['wchar']-before['wchar'],os_write_calls=after['syscw']-before['syscw'],
        file_logical_bytes=sum(x['size'] for x in files.values()),allocated_bytes=sum(x['allocated_bytes'] for x in files.values()),
        provenance='MEASURED',completion_policy='one full checksum scan after native durable publication; completion file and parent fsync')
    return receipt

def persist_qwen(runner,cap,path,layout,source):
    start=time.perf_counter();before=process_io();receipt=runner.persist(cap,path,layout=layout,storage_policy='prefix25' if layout=='repacked' else 'full');writer_ms=(time.perf_counter()-start)*1000
    # Native writer hashes exact serialized bytes. Reuse its hashes rather than rereading payload.
    t=time.perf_counter();records={k:{**v,'allocated_bytes':(path/k).stat().st_blocks*512} for k,v in receipt['metadata']['files'].items()}
    p=path/'meta.json';records['meta.json']={'size':p.stat().st_size,'allocated_bytes':p.stat().st_blocks*512,'sha256':sha(p)}
    files=complete(path,records,'completion.json');seal_ms=(time.perf_counter()-t)*1000
    elapsed=(time.perf_counter()-start)*1000;after=process_io()
    receipt.update(event_id=str(uuid.uuid4()),event_kind='physical_build',source=source,files=files,service_persistence_ms=elapsed,
        native_writer_wall_ms=writer_ms,completion_full_hash_ms=0.,completion_publication_ms=seal_ms,
        os_write_bytes=after['wchar']-before['wchar'],os_write_calls=after['syscw']-before['syscw'],file_logical_bytes=sum(x['size'] for x in files.values()),allocated_bytes=sum(x['allocated_bytes'] for x in files.values()),
        provenance='MEASURED',completion_policy='native per-payload complete serialized-byte SHA reused; metadata hash and completion+parent fsync')
    return receipt

def verify_completion(path,name,only_probe=False):
    start=time.perf_counter();records=json.loads((path/name).read_text());total=0;count=0
    for rel,row in records.items():
        if only_probe and not rel.endswith('probe_k.bin'):continue
        p=path/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or p.is_symlink() or not p.resolve().is_relative_to(path.resolve()) or p.stat().st_nlink!=1 or p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:raise ValueError('completion integrity mismatch '+rel)
        total+=row['size'];count+=1
    return {'integrity_ms':(time.perf_counter()-start)*1000,'integrity_read_bytes':total,'files':count,'scope':'unused bundle sidecar' if only_probe else 'full completion scan'}

def protection(run):
    root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('_main_b_protection',root/'scripts/69_protect_rekv_artifacts.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
    policy=p.FingerprintPolicy.from_dict(json.loads((run/'protection_policy.json').read_text()));changed=[];count=total=0
    with (run/'protected_artifacts_before.jsonl').open() as f:
        for line in f:
            rec=json.loads(line);path=root/rec['path'];count+=1
            if not path.exists() and not path.is_symlink():changed.append({'path':rec['path'],'reason':'MISSING'});continue
            if 'symlink' in rec:
                if not path.is_symlink() or os.readlink(path)!=rec['symlink']:changed.append({'path':rec['path'],'reason':'LINK_CHANGED'})
            else:
                digest,st=p._fingerprint_regular_file(path,policy);total+=st.st_size
                if digest!=rec['integrity'] or p._file_metadata(st)!=rec['metadata']:changed.append({'path':rec['path'],'reason':'CONTENT_METADATA_CHANGED'})
    before=json.loads((run/'source_before.json').read_text());sources=[name for name,digest in before.items() if not (root/name).is_file() or sha(root/name)!=digest]
    return {'status':'PASS' if not changed and not sources else 'FAIL','protected_files':count,'protected_logical_bytes':total,'changed_artifacts':changed,'changed_existing_source':sources,'policy':policy.as_dict(),'limitation':'historical files above 8 MiB use nine 64 KiB windows plus inode/mtime/ctime, not full SHA; existing source/small artifacts and all new artifacts use full SHA'}
