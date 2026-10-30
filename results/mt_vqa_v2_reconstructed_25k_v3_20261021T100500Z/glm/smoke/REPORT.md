# glm smoke official VQA analysis

Status: COMPLETE; adopted 36/36 requests; 4 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 50.0000 | 75.0000 | 50.0000 | 58.3333 | 62.5000 | 112.5900 | 686.2609 | 686.2609 |
| fullload | 50.0000 | 75.0000 | 50.0000 | 58.3333 | 62.5000 | 73.7386 | 607.1464 | 607.1464 |
| ours_prefixstore_kv25 | 50.0000 | 75.0000 | 25.0000 | 50.0000 | 50.0000 | 61.7260 | 610.4095 | 610.4095 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
