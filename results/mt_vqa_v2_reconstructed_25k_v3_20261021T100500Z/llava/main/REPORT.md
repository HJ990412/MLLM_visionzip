# llava main official VQA analysis

Status: COMPLETE; adopted 375000/375000 requests; 25000 images.

| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | 51.7940 | 51.6864 | 51.8592 | 51.7799 | 51.7728 | 530.1472 | 1679.7932 | 1679.7932 |
| fullload | 51.7940 | 51.6852 | 51.8760 | 51.7851 | 51.7806 | 755.7905 | 4541.3798 | 4541.3798 |
| mpic32_ssd | 51.7940 | 51.7756 | 51.8520 | 51.8072 | 51.8138 | 774.5743 | 4418.5849 | 4418.5849 |
| sparsevlm_ssd_kv25_allhead | 51.7940 | 51.4476 | 51.6948 | 51.6455 | 51.5712 | 844.5910 | 3008.4270 | 4770.9978 |
| ours_prefixstore_kv25 | 51.7940 | 49.8396 | 50.3808 | 50.6715 | 50.1102 | 262.4605 | 1953.9982 | 1953.9982 |

AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.
Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.
