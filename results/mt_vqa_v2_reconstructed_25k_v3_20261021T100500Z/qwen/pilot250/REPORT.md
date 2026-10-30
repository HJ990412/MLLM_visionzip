# qwen pilot250 official VQA analysis

Status: COMPLETE; adopted 2250/2250 requests; 250 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 77.6800 | 79.2800 | 78.1600 | 78.3733 | 78.7200 | 116.4980 | 413.9500 | 413.9500 |
| fullload | 77.6800 | 79.0800 | 78.5600 | 78.4400 | 78.8200 | 76.2189 | 417.1939 | 417.1939 |
| qwen_ours_prefixstore_kv25 | 77.6800 | 75.9600 | 75.4400 | 76.3600 | 75.7000 | 54.2391 | 344.2671 | 344.2671 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
