# glm pilot250 official VQA analysis

Status: COMPLETE; adopted 2250/2250 requests; 250 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 54.2000 | 55.4400 | 50.5200 | 53.3867 | 52.9800 | 130.6336 | 665.2235 | 665.2235 |
| fullload | 54.2000 | 54.9200 | 50.5200 | 53.2133 | 52.7200 | 76.5832 | 642.1482 | 642.1482 |
| ours_prefixstore_kv25 | 54.2000 | 51.5200 | 49.9200 | 51.8800 | 50.7200 | 61.6070 | 595.5934 | 595.5934 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
