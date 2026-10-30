# glm pilot40 official VQA analysis

Status: COMPLETE; adopted 360/360 requests; 40 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 47.7500 | 39.5000 | 33.2500 | 40.1667 | 36.3750 | 120.2512 | 671.8726 | 671.8726 |
| fullload | 47.7500 | 39.5000 | 33.2500 | 40.1667 | 36.3750 | 76.9387 | 673.8365 | 673.8365 |
| ours_prefixstore_kv25 | 47.7500 | 42.0000 | 30.0000 | 39.9167 | 36.0000 | 62.0991 | 622.9821 | 622.9821 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
