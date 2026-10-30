# qwen pilot40 official VQA analysis

Status: COMPLETE; adopted 360/360 requests; 40 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 64.0000 | 65.0000 | 68.5000 | 65.8333 | 66.7500 | 109.6992 | 395.1479 | 395.1479 |
| fullload | 64.0000 | 65.0000 | 68.5000 | 65.8333 | 66.7500 | 69.6572 | 391.1016 | 391.1016 |
| qwen_ours_prefixstore_kv25 | 64.0000 | 59.2500 | 63.5000 | 62.2500 | 61.3750 | 53.3990 | 333.2936 | 333.2936 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
