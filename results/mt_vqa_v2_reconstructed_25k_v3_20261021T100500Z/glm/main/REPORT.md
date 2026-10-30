# glm main official VQA analysis

Status: COMPLETE; adopted 225000/225000 requests; 25000 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 51.3200 | 49.5204 | 48.0932 | 49.6445 | 48.8068 | 126.3830 | 655.5769 | 655.5769 |
| fullload | 51.3200 | 49.5216 | 48.0504 | 49.6307 | 48.7860 | 75.6652 | 636.4029 | 636.4029 |
| ours_prefixstore_kv25 | 51.3200 | 46.8772 | 46.2144 | 48.1372 | 46.5458 | 60.7956 | 588.1694 | 588.1694 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
