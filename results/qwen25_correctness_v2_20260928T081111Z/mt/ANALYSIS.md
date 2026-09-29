# Qwen2.5-VL v2 validated mt_gqa_reconstructed pilot

GPU system correctness: **PASS** (G1–G15 all PASS). Raw coverage audit: **PASS**.
Frozen images: 40; requests: 360; stores: 80.
Request errors: 0; duplicate IDs: 0; nonfinite JSON values: 0; nonfinite first logits: 0; verified logit vectors: 360; truncated outputs: 0.

The workload is the original frozen GQA 40×6 or reconstructed MT 40×3 schedule. GQA questions are independent; MT history uses each method's own generated answers.

## Measured methods

| Method | Scope | Requests | Accuracy | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Visual read B | Structural read B | Metadata read B | preads | kept visual tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | all | 120 | 0.683 | 116.375 | 114.985 | 135.420 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| recompute | hit | 80 | 0.700 | 116.643 | 114.985 | 136.559 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| fullload | all | 120 | 0.683 | 91.790 | 83.034 | 127.972 | 15903402.667 | 802816.000 | 0.000 | 38.000 | — |
| fullload | hit | 80 | 0.700 | 81.114 | 76.590 | 120.670 | 23855104.000 | 1204224.000 | 0.000 | 57.000 | 370.475 |
| ours25 | all | 120 | 0.658 | 83.050 | 68.134 | 126.919 | 4709853.867 | 802816.000 | 0.000 | 38.000 | — |
| ours25 | hit | 80 | 0.662 | 65.837 | 60.662 | 94.087 | 7064780.800 | 1204224.000 | 0.000 | 57.000 | 123.200 |

## Paired output agreement

| Scope | Left | Right | Pairs | First token | Prediction | Generated sequence |
|---|---|---|---:|---:|---:|---:|
| all | recompute | fullload | 120 | 1.000 | 1.000 | 1.000 |
| all | recompute | ours25 | 120 | 0.917 | 0.900 | 0.900 |
| all | fullload | ours25 | 120 | 0.917 | 0.900 | 0.900 |
| hit | recompute | fullload | 80 | 1.000 | 1.000 | 1.000 |
| hit | recompute | ours25 | 80 | 0.875 | 0.850 | 0.850 |
| hit | fullload | ours25 | 80 | 0.875 | 0.850 | 0.850 |

## SSD and visual retention (cache hits)

MB below is MiB (1,048,576 bytes). Valid visual bytes count native BF16 K/V for retained tokens; read bytes include 64-token chunk padding.

| Method | Mean image retention % | Aggregate token retention % | Valid visual MiB | Visual read MiB | Padding MiB | Structural MiB | Metadata MiB | Total SSD ratio vs FullLoad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | — | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fullload | 100.000 | 100.000 | 20.260 | 22.750 | 2.490 | 1.148 | 0.000 | 1.000 |
| ours25 | 33.399 | 33.255 | 6.737 | 6.737 | 0.000 | 1.148 | 0.000 | 0.330 |

## One-time and session costs

GQA total is six independent requests per image plus one-time persistence and activation. MT E2E is a three-turn generated-history session.

| Method | Score ms | Repack ms | Write ms | fsync ms | Persistence ms | Activation ms | Per-image total ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute | — | — | — | — | — | 0.000 | 423.185 |
| fullload | 0.000 | 0.000 | 10.155 | 47.856 | 84.831 | 14.487 | 448.857 |
| ours25 | 1.760 | 0.850 | 10.135 | 38.258 | 77.858 | 14.995 | 417.183 |

## Paired image-cluster intervals

4,000 bootstrap draws with seed 1234 and image as the cluster unit.

| Ours25 minus | Metric | Mean | 95% low | 95% high |
|---|---|---:|---:|---:|
| recompute | ttft_ms | -50.805 | -55.102 | -45.842 |
| recompute | accuracy | -0.037 | -0.113 | 0.025 |
| recompute | visual_read_bytes | 7064780.800 | 6697779.200 | 7340032.000 |
| recompute | structural_read_bytes | 1204224.000 | 1204224.000 | 1204224.000 |
| recompute | metadata_read_bytes | 0.000 | 0.000 | 0.000 |
| fullload | ttft_ms | -15.277 | -19.736 | -10.974 |
| fullload | accuracy | -0.037 | -0.113 | 0.025 |
| fullload | visual_read_bytes | -16790323.200 | -17707827.200 | -16056320.000 |
| fullload | structural_read_bytes | 0.000 | 0.000 | 0.000 |
| fullload | metadata_read_bytes | 0.000 | 0.000 | 0.000 |
A confidence interval crossing zero does not establish equivalence.

Validation SHA256: `a3b851f6fc291711ef53452b6eab164ae84e6d3801ce717b19b8386a120ad860`. Frozen workload content SHA256: `14d97f83bca9edebafd829c20d6f76fc76a3d1f6024ded1dd9a6385dcde719f1`.
