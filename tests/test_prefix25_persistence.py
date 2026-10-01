"""Actual prefix-only payloads, independent reference and malformed-store tests."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from mmimpress.prefix25 import seal_integrity, verify_integrity, ReadTrace, validate_llava_prefix_meta
from mmimpress.qwen25.store import write_qwen_store, QwenStore, plan_prefix_budget
from mmimpress.store import write_image_store, ChunkReader
from mmimpress.serve import ImageContext, CVPR25ChunkSelector, BIAS
from test_qwen25_kv25 import _fixture as qfixture
from test_llava_kv25 import _fixture as lfixture
CASES = (1,63,64,65,127,128,129,255,256,257,349,2200)

class PrefixPersistence(unittest.TestCase):
    def test_qwen_boundaries_bits_reads_and_negative(self):
        for n in CASES:
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); layers,start,p,scores,ids,extra=qfixture(n)
                a=write_qwen_store(root/'a',layers,start,n,ids,scores,extra)
                b=write_qwen_store(root/'b',layers,start,n,ids,scores,extra,storage_policy='prefix25')
                k=(n+3)//4; order=sorted(range(n),key=lambda i:(-scores[i],i))
                self.assertEqual(b['stored_to_original'],order[:k]);self.assertEqual(b['visual_count'],n)
                self.assertEqual(b['stored_rows'],k);self.assertEqual(b['padding_rows'],0)
                logical=sorted([0,1,2,p-1]+[start+i for i in order[:k]])
                with QwenStore(root/'a') as aa, QwenStore(root/'b') as bb:
                    x=aa.load_prefix(.25,budget_unit='visual_kv')
                    with ReadTrace(root/'b') as trace:y=bb.load_prefix(.25,budget_unit='visual_kv')
                    self.assertEqual(trace.summary()['bytes'],y.io.bytes)
                    self.assertEqual(trace.summary()['preads'],y.io.preads)
                    self.assertEqual(y.kept_visual_tokens,k); self.assertEqual(list(y.logical_indices),logical)
                    for orig,xx,yy in zip(layers,x.layers,y.layers):
                        for src,av,bv in zip(orig,xx,yy):
                            self.assertTrue(torch.equal(av.view(torch.int16),bv.view(torch.int16)))
                            self.assertTrue(torch.equal(src[:,:,logical].view(torch.int16),bv.view(torch.int16)))
                    for ratio,unit in [(1,'visual_kv'),(.26,'visual_kv'),(.25,'chunk')]:
                        with self.assertRaises(ValueError):bb.load_prefix(ratio,budget_unit=unit)
                    self.assertEqual(y.extra_valid_visual_rows,0)
                with self.assertRaises(ValueError):QwenStore(root/'b',expected_identity={'image_sha256':'b'*64})
                bad=copy.deepcopy(b);bad['stored_to_original'][0]=(bad['stored_to_original'][0]+1)%max(n,2)
                for field,val in [('original_content_count',k),('stored_content_count',(k+3)//4)]:
                    if n!=k and k!=(k+3)//4:
                        m=copy.deepcopy(b);m[field]=val
                        obj=object.__new__(QwenStore);obj.meta=m;obj.path=root/'b';obj.metadata_file_bytes=b['bytes_metadata_file']
                        with self.assertRaises(ValueError):obj._validate_meta(None)
                f=root/'b/layer_000/k.bin';blob=f.read_bytes();f.write_bytes(bytes([blob[0]^1])+blob[1:])
                with self.assertRaisesRegex(ValueError,'hash'):QwenStore(root/'b')
                f.write_bytes(blob[:-2])
                with self.assertRaisesRegex(ValueError,'size'):QwenStore(root/'b')

    def test_llava_boundaries_full_buffer_mask_and_bits(self):
        for n in CASES:
            with self.subTest(n=n),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);meta,rank,layers=lfixture(root/'a',n);k=(n+3)//4
                reserved={'image_id':'fixture'}
                extra={key:meta[key] for key in ['physical_layout','layout_method','layout_source','visual_kv_source','layout_uses_dataset_question','llm_used_for_layout_scoring','calibration_questions','global_order_all_layers','separator_tail','separator_policy','permutation_sha256','inverse_permutation_sha256']}
                b=write_image_store(root/'b',layers,meta['v_token_start'],meta['v_token_num'],meta['prefix_input_ids'],meta['newline_idx'],extra=extra,stored_to_original=meta['order'],separator_sidecar=True,probe_heads=0,storage_policy='prefix25')
                (root/'b/visionzip_layout.pt').write_bytes((root/'a/visionzip_layout.pt').read_bytes())
                seal_integrity(root/'b');validate_llava_prefix_meta(b)
                ctx=ImageContext(root/'b','cpu',drop_cache=False,require_v_hidden=False)
                try:
                    ctx.validate_visual_kv_layout(); cache=ctx.cache.new_request()
                    runner=SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')))
                    sel=CVPR25ChunkSelector(runner,ctx,None,.25,'prefix',sep_policy='sidecar',budget_unit='visual_kv')
                    with patch('torch.cuda.synchronize'),ReadTrace(root/'b') as trace:sel.prepare(None)
                    self.assertEqual(sel.io.bytes,trace.summary()['bytes']);self.assertEqual(sel.io.preads,trace.summary()['preads'])
                    self.assertEqual(sel.selected_original_ids,rank[:k]);self.assertEqual(sel.stats()['unused_loaded_real_rows'],0)
                    expected=[True]*3+[True]*k+[False]*(n-k)+[True]*2
                    for li,(src_k,src_v) in enumerate(layers):
                        self.assertEqual((BIAS[li].flatten()==0).tolist(),expected)
                        for kind,src in [('keys',src_k),('values',src_v)]:
                            actual=getattr(cache.layers[li],kind)
                            self.assertEqual(actual.shape[2],n+5)
                            self.assertTrue(torch.equal(actual[:,:,3:3+k],src[:,:,torch.tensor(rank[:k])+3].to(torch.bfloat16)))
                    for ratio,unit in [(1,'visual_kv'),(.26,'visual_kv'),(.25,'chunk')]:
                        with self.assertRaises(ValueError):CVPR25ChunkSelector(runner,ctx,None,ratio,'prefix',sep_policy='sidecar',budget_unit=unit)
                    with self.assertRaises(ValueError):ctx.reader.read_full(0,'k')
                finally:ctx.close();BIAS.clear()
                for rel in ['sep_kv.bin','sys_kv.pt']:
                    self.assertEqual((root/'a'/rel).read_bytes(),(root/'b'/rel).read_bytes())
                for li in range(2):
                    for kind in ['k','v']:
                        blob=(root/'b'/f'layer_{li:02d}/{kind}.bin').read_bytes()
                        self.assertEqual(len(blob),k*8)
                        self.assertEqual(blob,(root/'a'/f'layer_{li:02d}/{kind}.bin').read_bytes()[:k*8])
                corrupt=copy.deepcopy(b);corrupt['stored_row_to_original'][0]=-1
                with self.assertRaises(ValueError):validate_llava_prefix_meta(corrupt)
                f=root/'b/layer_00/k.bin';blob=f.read_bytes();f.write_bytes(blob[:-2])
                with self.assertRaises(ValueError):verify_integrity(root/'b')
                with ChunkReader(root/'b',b,False) as reader:
                    with self.assertRaises(AssertionError):reader.read_chunks(0,'k',[0] if k<=64 else [(k-1)//64])

    def test_zero_and_invalid_storage_policy(self):
        layers,start,p,scores,ids,extra=qfixture(0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):write_qwen_store(Path(tmp)/'b',layers,start,0,ids,scores,extra,storage_policy='prefix25')
            with self.assertRaises(ValueError):write_image_store(Path(tmp)/'l',layers,0,0,[],[],storage_policy='prefix25')
            with self.assertRaises(ValueError):write_qwen_store(Path(tmp)/'x',layers,start,0,ids,scores,extra,storage_policy='bogus')

if __name__=='__main__':unittest.main()
