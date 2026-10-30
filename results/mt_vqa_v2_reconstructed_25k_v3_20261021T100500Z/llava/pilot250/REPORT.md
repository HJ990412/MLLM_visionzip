# llava pilot250 official VQA analysis

Status: COMPLETE; adopted 3750/3750 requests; 250 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 53.8400 | 55.3600 | 55.3200 | 54.8400 | 55.3400 | 535.7773 | 1693.5388 | 1693.5388 |
| fullload | 53.8400 | 55.5200 | 54.9200 | 54.7600 | 55.2200 | 776.8891 | 4676.4751 | 4676.4751 |
| mpic32_ssd | 53.8400 | 55.3600 | 55.2000 | 54.8000 | 55.2800 | 791.2056 | 4532.3540 | 4532.3540 |
| sparsevlm_ssd_kv25_allhead | 53.8400 | 55.2400 | 54.1200 | 54.4000 | 54.6800 | 873.6995 | 3096.6517 | 4925.4712 |
| ours_prefixstore_kv25 | 53.8400 | 52.1200 | 52.4400 | 52.8000 | 52.2800 | 265.4245 | 1975.9404 | 1975.9404 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
