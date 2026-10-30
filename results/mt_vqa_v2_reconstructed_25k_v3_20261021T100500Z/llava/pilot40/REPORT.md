# llava pilot40 official VQA analysis

Status: COMPLETE; adopted 600/600 requests; 40 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 42.2500 | 44.5000 | 53.5000 | 46.7500 | 49.0000 | 527.7851 | 1668.1731 | 1668.1731 |
| fullload | 42.2500 | 44.5000 | 53.5000 | 46.7500 | 49.0000 | 756.7646 | 4473.7911 | 4473.7911 |
| mpic32_ssd | 42.2500 | 44.5000 | 53.5000 | 46.7500 | 49.0000 | 774.5947 | 4322.8236 | 4322.8236 |
| sparsevlm_ssd_kv25_allhead | 42.2500 | 45.2500 | 56.0000 | 47.8333 | 50.6250 | 830.8289 | 2980.9950 | 4683.7674 |
| ours_prefixstore_kv25 | 42.2500 | 46.2500 | 42.0000 | 43.5000 | 44.1250 | 254.6231 | 1912.2986 | 1912.2986 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
