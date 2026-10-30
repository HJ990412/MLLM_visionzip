# qwen main official VQA analysis

Status: COMPLETE; adopted 225000/225000 requests; 25000 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 75.0316 | 75.3364 | 75.0076 | 75.1252 | 75.1720 | 112.8123 | 404.5072 | 404.5072 |
| fullload | 75.0316 | 75.2940 | 74.9760 | 75.1005 | 75.1350 | 70.7596 | 398.8346 | 398.8346 |
| qwen_ours_prefixstore_kv25 | 75.0316 | 70.4020 | 70.5852 | 72.0063 | 70.4936 | 53.9946 | 338.6559 | 338.6559 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
