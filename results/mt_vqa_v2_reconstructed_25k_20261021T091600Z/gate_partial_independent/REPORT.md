# llava gate official VQA analysis

Status: PARTIAL; adopted 15/180 requests; 1 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 0.0000 | 0.0000 | 100.0000 | 33.3333 | 50.0000 | 561.3060 | 1782.4552 | 1782.4552 |
| fullload | 0.0000 | 0.0000 | 100.0000 | 33.3333 | 50.0000 | 917.0017 | 5901.4896 | 5901.4896 |
| mpic32_ssd | 0.0000 | 0.0000 | 100.0000 | 33.3333 | 50.0000 | 861.8849 | 5193.2246 | 5193.2246 |
| sparsevlm_ssd_kv25_allhead | 0.0000 | 0.0000 | 100.0000 | 33.3333 | 50.0000 | 889.6966 | 3235.2448 | 5995.0146 |
| ours_prefixstore_kv25 | 0.0000 | 0.0000 | 100.0000 | 33.3333 | 50.0000 | 269.0513 | 2334.1382 | 2334.1382 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
