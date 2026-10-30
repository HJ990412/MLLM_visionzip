# qwen smoke official VQA analysis

Status: COMPLETE; adopted 36/36 requests; 4 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 65.0000 | 100.0000 | 75.0000 | 80.0000 | 87.5000 | 109.9318 | 393.8817 | 393.8817 |
| fullload | 65.0000 | 100.0000 | 75.0000 | 80.0000 | 87.5000 | 70.4806 | 415.9617 | 415.9617 |
| qwen_ours_prefixstore_kv25 | 65.0000 | 90.0000 | 100.0000 | 85.0000 | 95.0000 | 54.2450 | 344.8771 | 344.8771 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
