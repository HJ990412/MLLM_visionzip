from pathlib import Path
import datetime,json,hashlib,os,shutil,importlib.util,sys
root=Path.cwd();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run=root/'runs'/('mt_gqa_B_llava5_qwen3_'+stamp);run.mkdir();(root/'results'/run.name).mkdir();Path('/tmp/main_b_run').write_text(str(run))
spec=importlib.util.spec_from_file_location('protect',root/'scripts/69_protect_rekv_artifacts.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
policy=p.FingerprintPolicy(full_hash_max_bytes=8<<20,sample_window_bytes=65536,sample_count=9)
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as x:
  for b in iter(lambda:x.read(8<<20),b''):h.update(b)
 return h.hexdigest()
source={}
for folder in ('mmimpress','scripts','tests','docs'):
 for f in (root/folder).rglob('*'):
  if f.is_file() and f.suffix in ('.py','.md','.json','.yaml','.yml','.sh'):
   name=str(f.relative_to(root));source[name]=sha(f);dest=run/'source_before'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
(run/'source_before.json').write_text(json.dumps(source,indent=2))
count=size=0
with (run/'protected_artifacts_before.jsonl').open('x') as out:
 for directory in [root/'data',root/'runs',root/'results',*root.glob('kvstore*')]:
  if not directory.is_dir():continue
  for base,dirs,files in os.walk(directory,followlinks=False):
   bp=Path(base)
   dirs[:]=[d for d in dirs if not (bp/d).is_relative_to(run) and not (bp/d).is_relative_to(root/'results'/run.name)]
   if bp.is_relative_to(run) or bp.is_relative_to(root/'results'/run.name):continue
   for name in files:
    f=bp/name;rec={'path':str(f.relative_to(root))}
    if f.is_symlink():rec['symlink']=os.readlink(f)
    else:
     integrity,st=p._fingerprint_regular_file(f,policy);rec.update(integrity=integrity,metadata=p._file_metadata(st));size+=st.st_size
    out.write(json.dumps(rec,separators=(',',':'))+'\n');count+=1
 out.flush();os.fsync(out.fileno())
(run/'protection_policy.json').write_text(json.dumps(policy.as_dict(),indent=2));print(json.dumps({'run':str(run),'protected_files':count,'logical_bytes':size,'source_files':len(source)}),flush=True)
