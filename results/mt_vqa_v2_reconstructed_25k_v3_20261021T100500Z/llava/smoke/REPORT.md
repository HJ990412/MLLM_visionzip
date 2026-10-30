# llava smoke official VQA analysis

Status: COMPLETE; adopted 60/60 requests; 4 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 40.0000 | 75.0000 | 100.0000 | 71.6667 | 87.5000 | 431.7345 | 1356.7155 | 1356.7155 |
| fullload | 40.0000 | 75.0000 | 100.0000 | 71.6667 | 87.5000 | 632.9386 | 3833.5798 | 3833.5798 |
| mpic32_ssd | 40.0000 | 75.0000 | 100.0000 | 71.6667 | 87.5000 | 658.6230 | 3702.6833 | 3702.6833 |
| sparsevlm_ssd_kv25_allhead | 40.0000 | 75.0000 | 100.0000 | 71.6667 | 87.5000 | 693.2630 | 2462.3179 | 4008.9340 |
| ours_prefixstore_kv25 | 40.0000 | 75.0000 | 57.5000 | 57.5000 | 66.2500 | 234.1329 | 1671.5395 | 1671.5395 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
