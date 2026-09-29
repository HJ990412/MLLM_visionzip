# Qwen2.5-VL v2 validated gqa pilot

GPU system correctness: **PASS** (G1–G15 all PASS). Raw coverage audit: **PASS**.
Frozen images: 40; requests: 720; stores: 80.
Request errors: 0; duplicate IDs: 0; nonfinite JSON values: 0; nonfinite first logits: 0; verified logit vectors: 720; truncated outputs: 0.

The workload is the original frozen GQA 40×6 or reconstructed MT 40×3 schedule. GQA questions are independent; MT history uses each method's own generated answers.

## Measured methods

| Method | Scope | Requests | Accuracy | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Visual read B | Structural read B | Metadata read B | preads | kept visual tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | all | 240 | 0.567 | 111.541 | 110.900 | 134.511 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| recompute | hit | 200 | 0.590 | 110.889 | 110.519 | 132.287 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| fullload | all | 240 | 0.567 | 78.823 | 73.618 | 113.837 | 18732373.333 | 1003520.000 | 0.000 | 47.500 | — |
| fullload | hit | 200 | 0.590 | 72.712 | 70.086 | 87.238 | 22478848.000 | 1204224.000 | 0.000 | 57.000 | 349.000 |
| ours25 | all | 240 | 0.575 | 68.868 | 59.764 | 117.350 | 5352106.667 | 1003520.000 | 0.000 | 47.500 | — |
| ours25 | hit | 200 | 0.600 | 59.943 | 56.893 | 81.335 | 6422528.000 | 1204224.000 | 0.000 | 57.000 | 112.000 |

## Paired output agreement

| Scope | Left | Right | Pairs | First token | Prediction | Generated sequence |
|---|---|---|---:|---:|---:|---:|
| all | recompute | fullload | 240 | 1.000 | 1.000 | 1.000 |
| all | recompute | ours25 | 240 | 0.871 | 0.850 | 0.850 |
| all | fullload | ours25 | 240 | 0.871 | 0.850 | 0.850 |
| hit | recompute | fullload | 200 | 1.000 | 1.000 | 1.000 |
| hit | recompute | ours25 | 200 | 0.845 | 0.820 | 0.820 |
| hit | fullload | ours25 | 200 | 0.845 | 0.820 | 0.820 |

## SSD and visual retention (cache hits)

MB below is MiB (1,048,576 bytes). Valid visual bytes count native BF16 K/V for retained tokens; read bytes include 64-token chunk padding.

| Method | Mean image retention % | Aggregate token retention % | Valid visual MiB | Visual read MiB | Padding MiB | Structural MiB | Metadata MiB | Total SSD ratio vs FullLoad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute | — | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fullload | 100.000 | 100.000 | 19.086 | 21.438 | 2.352 | 1.148 | 0.000 | 1.000 |
| ours25 | 31.730 | 32.092 | 6.125 | 6.125 | 0.000 | 1.148 | 0.000 | 0.322 |

## One-time and session costs

GQA total is six independent requests per image plus one-time persistence and activation. MT E2E is a three-turn generated-history session.

| Method | Score ms | Repack ms | Write ms | fsync ms | Persistence ms | Activation ms | Per-image total ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute | — | — | — | — | — | 0.000 | 818.224 |
| fullload | 0.000 | 0.000 | 9.650 | 35.631 | 71.400 | 13.752 | 700.183 |
| ours25 | 1.723 | 0.873 | 9.692 | 32.780 | 71.838 | 14.269 | 651.642 |

## Paired image-cluster intervals

4,000 bootstrap draws with seed 1234 and image as the cluster unit.

| Ours25 minus | Metric | Mean | 95% low | 95% high |
|---|---|---:|---:|---:|
| recompute | ttft_ms | -50.946 | -54.408 | -47.249 |
| recompute | accuracy | 0.010 | -0.035 | 0.055 |
| recompute | visual_read_bytes | 6422528.000 | 5872025.600 | 6881280.000 |
| recompute | structural_read_bytes | 1204224.000 | 1204224.000 | 1204224.000 |
| recompute | metadata_read_bytes | 0.000 | 0.000 | 0.000 |
| fullload | ttft_ms | -12.770 | -15.056 | -10.252 |
| fullload | accuracy | 0.010 | -0.035 | 0.055 |
| fullload | visual_read_bytes | -16056320.000 | -16790323.200 | -15414067.200 |
| fullload | structural_read_bytes | 0.000 | 0.000 | 0.000 |
| fullload | metadata_read_bytes | 0.000 | 0.000 | 0.000 |
A confidence interval crossing zero does not establish equivalence.

Validation SHA256: `a3b851f6fc291711ef53452b6eab164ae84e6d3801ce717b19b8386a120ad860`. Frozen workload content SHA256: `f8e39361f546104d4478955f1815e5632982c24ce33e3e342247940e568e7f7f`.
