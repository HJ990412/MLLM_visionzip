# MT-GQA Generated-History: five-method same-run main experiment

## Main results

| Method | Acc T1 | Acc T2 | Acc T3 | Avg Acc | TTFT mean | TTFT p50 | TTFT p95 | SSD MB/hit | Preads/hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 63.19% | 68.11% | 68.85% | 66.72% | 531.93 | 533.73 | 597.90 | 0.00 | 0.00 |
| FullLoad | 63.19% | 68.06% | 68.70% | 66.65% | 744.66 | 741.85 | 902.15 | 1173.92 | 64.00 |
| MPIC-32 (adapted) | 63.19% | 68.01% | 69.10% | 66.77% | 718.02 | 718.07 | 861.48 | 1174.18 | 65.00 |
| ReKV-Chunk25 (adapted) | 63.19% | 66.68% | 67.10% | 65.66% | 604.72 | 621.20 | 726.33 | 307.45 | 166.62 |
| Ours25 | 63.19% | 65.48% | 66.73% | 65.13% | 301.54 | 302.22 | 350.71 | 307.82 | 65.00 |

Accuracy is strict normalized exact match on the reconstructed MT-GQA workload. TTFT values are milliseconds for the pooled **8,122 individual T2/T3 requests per method**; they are not cumulative three-turn session latencies. SSD MB uses decimal 10^6 bytes.

## Dataset, seed, and measurement boundary

- Frozen index SHA-256: 2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224.
- Frozen dialogue workload SHA-256: 0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62.
- 398 images, 4,061 three-turn dialogues, 60,915 logical and physical requests across five methods.
- Inference seed 1234; 16-token greedy generation; method-local generated answers feed T2 and T3.
- TTFT starts before prompt construction and ends after first-token materialization with CUDA synchronization. ReComp includes image processing, vision, and multimodal prefill.
- Cache hits use OS-page-cache-cold conditioning with posix_fadvise(DONTNEED) and buffered pread. Page-cache conditioning is excluded from TTFT. SSD controller cache flush is not established.
- ReKV metadata activation is outside cache-hit TTFT. ReKV and MPIC are SSD workload adaptations, not their original papers' complete serving systems.

## Ours cache-hit TTFT difference against each baseline

Positive values mean Ours25 has a shorter mean TTFT.

| Baseline | Baseline − Ours ms | Reduction vs baseline |
|---|---:|---:|
| ReComp | +230.40 | +43.31% |
| FullLoad | +443.13 | +59.51% |
| MPIC-32 (adapted) | +416.48 | +58.00% |
| ReKV-Chunk25 (adapted) | +303.19 | +50.14% |

## Paired quality

Difference direction is Ours25 minus the named baseline. The bootstrap uses 10,000 image-cluster resamples with analysis seed 1234, keeping all dialogues and all three turns together within a sampled image. Paired contingency cells are in paired_quality.json.

| Baseline | T1 Δpp | T2 Δpp | T3 Δpp | Avg Δpp | Avg 95% CI Δpp |
|---|---:|---:|---:|---:|---:|
| ReKV-Chunk25 (adapted) | +0.00 | -1.21 | -0.37 | -0.53 | [-1.14, +0.10] |
| MPIC-32 (adapted) | +0.00 | -2.54 | -2.36 | -1.63 | [-2.14, -1.12] |
| FullLoad | +0.00 | -2.59 | -1.97 | -1.52 | [-2.03, -1.01] |
| ReComp | +0.00 | -2.63 | -2.12 | -1.58 | [-2.09, -1.08] |

A confidence interval that includes zero supports only 'no statistically clear difference' under this analysis. It does not establish equivalence or the same quality.

## ReKV and Ours selection and I/O

- ReKV T2↔T3 selected-chunk Jaccard: 0.9479 mean; 4,011/4,061 dialogue pairs changed. Same-image question-pair mean: 0.9116.
- Ours T2↔T3 selected-chunk Jaccard: 1.0000; every hit uses the same physical first-k chunk prefix for its image.
- ReKV selected-layout runs/layer mean: 2.59; actual attention key length mean: 635.37.
- ReKV instrumented online decision component sum mean: 10.19 ms; retrieval-forward wall mean 553.15 ms; answer-prefill wall mean 50.72 ms. The component sum includes host intervals and is not isolated GPU time.
- ReKV Stage-A SSD mean: 307.45 MB/hit; Stage-B duplicate payload read total: 0 bytes; duplicate selected-range read total: 0 bytes.
- ReKV metadata mean: 9.250 MB/image. Ours online selector mean: 0.22 ms.
- MPIC recomputes 32 leading image tokens per cache hit. Its traffic is measured in the main table.

## One-time persistence and three-turn session latency

Persistence follows the source method's normal T1 answer forward and is excluded from T2/T3 TTFT. T1 capture instrumentation is already within the measured T1 request. The totals below use T1+T2+T3 request end-to-end times plus charged persistence; when measured, ReKV metadata activation is also charged. Phase means in persistence.csv are descriptive and must not be summed to reconstruct a critical path.

| Method | Stores | Persist mean ms/image | Store MB/image | Physical stream session mean ms | Standalone-equivalent session mean ms |
|---|---:|---:|---:|---:|---:|
| ReComp | 0 | 0.00 | 0.00 | 1668.10 | 1668.10 |
| FullLoad | 398 | 1101.69 | 1264.65 | 2211.81 | 3214.11 |
| MPIC-32 (adapted) | 398 | 1002.98 | 1187.79 | 2143.37 | 3050.74 |
| ReKV-Chunk25 (adapted) | 398 | 1728.77 | 1200.84 | 2021.49 | 3595.98 |
| Ours25 | 398 | 1043.31 | 1191.60 | 1312.43 | 2259.67 |

Physical-stream attribution charges each of the 398 store builds only to its actual source dialogue; later dialogues with the same image reuse it. Standalone-equivalent attribution charges the measured per-image build to every three-turn dialogue as a derived single-session scenario. These columns answer different deployment questions. Per-dialogue values are in session_per_dialogue.csv.

## Six direct questions

1. **Ours25 vs ReKV quality at 25% Visual-KV budget:** Ours minus ReKV observed Avg accuracy is -0.53 percentage points (image-cluster 95% CI [-1.14, +0.10]). No statistically clear difference is established.
2. **Ours cache-hit TTFT reduction vs ReKV:** +50.14% relative to ReKV, with pooled means 301.54 vs 604.72 ms.
3. **FullLoad vs ReComp cache-hit TTFT:** FullLoad is slower by 39.99% (744.66 vs 531.93 ms).
4. **MPIC-32 SSD traffic reduction vs FullLoad:** -0.02% (1174.18 vs 1173.92 MB/hit).
5. **Observed quality gain from query-dependent retrieval:** ReKV minus Ours Avg accuracy is +0.53 percentage points. The image-cluster interval includes zero, so this run shows no statistically clear gain; the method comparison does not isolate retrieval as a causal factor.
6. **Ours vs ReComp with one-time persistence in a three-turn session:** Standalone-equivalent Ours is slower by 591.56 ms (2259.67 vs 1668.10 ms). In the actual shared-image physical stream the corresponding means are 1312.43 vs 1668.10 ms.

## Validation and interpretation limits

- All 60,915 logical requests have unique physical execution IDs and exact method-local generated-history lineage. No failed or duplicate final logical request is present.
- Recorded completed-row retry count: 0; repeated shard invocations: 0; incomplete-image rebuilds: 0; failed/interrupted shard invocations: 0/0. Attempt events are reported separately from final request coverage.
- Prior run/results/source-store artifacts and source code pass before/after protection with zero missing or changed paths.
- The dataset is MT-GQA-reconstructed, not the unavailable official MetaCompress dialogue artifact. The scorer is the repository's strict normalized exact-match metric.
- ReKV-Chunk25 and MPIC-32 are adapted SSD baselines. ReKV's 25% visual-chunk budget is this experiment's comparison setting, not a claim about the original paper's 25% setting.
- OS page-cache conditioning does not prove a cold SSD controller cache. The reported timings are on this GPU/software stack.
- Same observed accuracy or a confidence interval spanning zero is not statistical equivalence.

MT-GQA GENERATED-HISTORY FIVE-ARM MAIN VALIDATED: YES
