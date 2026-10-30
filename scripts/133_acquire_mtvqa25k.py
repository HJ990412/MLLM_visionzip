#!/usr/bin/env python3
"""Acquire original official validation archives; never alter existing datasets."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, time, zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'v2_Questions_Val_mscoco.zip': 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
    'v2_Annotations_Val_mscoco.zip': 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
    'val2014.zip': 'http://images.cocodataset.org/zips/val2014.zip',
}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=ROOT/'data/mt_vqa_v2_reconstructed_25k/source');a=p.parse_args()
    dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
    for name,url in SOURCES.items():
        final=dest/name;part=dest/(name+'.part');receipt=dest/(name+'.receipt.json')
        if final.exists():
            if not receipt.exists() or json.loads(receipt.read_text())['sha256']!=sha(final):
                raise RuntimeError('Existing archive lacks matching provenance: '+str(final))
            print('verified existing '+name,flush=True);continue
        if shutil.disk_usage(dest).free < (30<<30)+15_000_000_000: raise RuntimeError('30 GiB reserve + download/extraction headroom unavailable')
        print('download '+url,flush=True);started=time.time()
        subprocess.run(['curl','-fL','--retry','3','--connect-timeout','30','--speed-time','120','--speed-limit','1024','-C','-',url,'-o',str(part),'-D',str(dest/(name+'.headers.txt'))],check=True)
        with zipfile.ZipFile(part) as z:
            infos=z.infolist()
            if any(i.filename.startswith('/') or '..' in Path(i.filename).parts for i in infos): raise ValueError('unsafe archive')
            if len({i.filename for i in infos})!=len(infos): raise ValueError('duplicate archive member')
            # Full CRC is checked while extracting below, not repeatedly scanning 6 GB.
        os.link(part,final);part.unlink()
        record={'url':url,'official_download_page':'https://visualqa.org/download.html','filename':name,'bytes':final.stat().st_size,'sha256':sha(final),'version':'VQAv2 validation 2017 v2.0 / COCO val2014','started_unix':started,'finished_unix':time.time(),'members':len(infos)}
        with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(record),flush=True)
    for name in SOURCES:
        with zipfile.ZipFile(dest/name) as z:
            for i in z.infolist():
                if i.is_dir():continue
                target=dest/i.filename;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    import zlib
                    crc=0
                    with target.open('rb') as f:
                        for b in iter(lambda:f.read(1<<20),b''):crc=zlib.crc32(b,crc)
                    if target.stat().st_size!=i.file_size or crc!=i.CRC:raise ValueError('existing extracted file mismatch: '+str(target))
                    continue
                if shutil.disk_usage(dest).free < (30<<30)+i.file_size:raise RuntimeError('free space floor')
                partial=target.with_name(target.name+'.extracting')
                with z.open(i) as src,partial.open('xb') as out:shutil.copyfileobj(src,out,1<<20)
                os.link(partial,target);partial.unlink()
        print('extracted and CRC verified '+name,flush=True)

if __name__=='__main__':main()
