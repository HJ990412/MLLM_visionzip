# VisDial Cache-Hit Serving Analysis

Validation: **PASS**

## Analysis scope

The paper-facing main population is strictly VisDial Turns 2--10: 100 dialogs x 9 cache-hit turns = 900 requests per method. Turn 1 is a cache-miss/construction-opportunity sanity check and is not mixed into the main average. One-time persistence is reported separately; it is neither added to cache-hit request TTFT nor described as free.

- Frozen index SHA256: `8c3dd7e983cb39e61d26362a0353b86ac84845bd7537a6331078ab7707777383`
- Ordered request-key SHA256: `395ba928a15eb45bca905aa91d1ec89617981f18b9b22338341b2a189b3f141a`
- History policy: gold teacher-forced
- Main metric: `end_to_end_ttft_ms`

## Timing boundary

OS page-cache conditioning via `posix_fadvise(DONTNEED)` completes outside the timer. The request timer starts before prompt construction and tokenization, then includes input preparation, initial H2D, vision recomputation or SSD read/scatter, prefill, first-token selection, and the first-token CUDA synchronization. `core_ttft_ms` is diagnostic only.

Reads use buffered `pread`; neither `O_DIRECT` nor an SSD-controller cache flush is used. This is an OS-page-cache-cold condition, not a physically true-cold-SSD claim. JPEG read/decode is outside the timer; model resize/crop/tensorization is inside.

## Main cache-hit result: Turns 2--10

| Method | Aux quality† | TTFT mean | p50 | p95 | Δ TTFT vs ReComp | TTFT reduction | E2E mean | SSD MB/req | SSD ratio vs FullLoad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.439 | 539.30 | 548.22 | 595.30 | 0.00 | 0.00% | 571.73 | 0.00 | — |
| FullLoad | 0.440 | 686.29 | 682.80 | 833.99 | 146.99 | -27.26% | 723.88 | 1169.43 | 100.00% |
| Prefix25 | 0.418 | 278.34 | 277.85 | 334.32 | -260.96 | 48.39% | 325.26 | 306.83 | 26.24% |
| Prefix45 | 0.440 | 417.59 | 419.84 | 494.78 | -121.71 | 22.57% | 458.43 | 554.80 | 47.44% |

† Normalized generative-match auxiliary score; not official VisDial MRR/R@K/Mean Rank/NDCG.

### Main comparisons

- Prefix25 vs ReComp: TTFT 48.39% lower, E2E 43.11% lower, quality delta -0.021.
- Prefix45 vs ReComp: TTFT 22.57% lower, E2E 19.82% lower, quality delta 0.001.
- Prefix25/45 reduce SSD bytes versus FullLoad by 73.76% / 52.56%.

## Cache-hit I/O breakdown

| Method | SSD MB/req | SSD read ms | OS preads/req | Scatter ms | Raw prefill ms | First-k planning ms |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.00 | 0.00 | 0.00 | 0.00 | 523.91 | 0.00 |
| FullLoad | 1169.43 | 418.19 | 64.00 | — | 680.96 | 0.00 |
| Prefix25 | 306.83 | 137.34 | 65.00 | 51.51 | 56.05 | 0.21 |
| Prefix45 | 554.80 | 218.17 | 65.00 | 82.38 | 56.68 | 0.33 |

`preads/req` is the measured `ssd_read_preads` count; raw chunk-unit counts are not treated as system calls. The inherited `selector_ms` field is deterministic first-k index planning only: static/query/diversity scorer calls are all zero.
Main-table SSD ratios use actual aggregate read bytes divided by FullLoad aggregate read bytes. They are not nominal budgets or the unweighted mean of per-image selected-KV ratios.

Phase caveat: ReComp raw `prefill_ms` includes vision, FullLoad raw `prefill_ms` includes its hook-based SSD read/cache write, and Prefix prefill starts after separately measured read/scatter. FullLoad scatter is therefore N/A rather than zero. These raw phase columns are diagnostic and must not be compared as mutually exclusive compute phases.

## Turn-1 sanity

| Method | TTFT mean/p50/p95 (ms) | Prediction agreement | First-token agreement | Vision forwards | SSD read |
|---|---:|---:|---:|---:|---:|
| ReComp | 519.42/525.40/567.87 | 100.00% | 100.00% | 1.00 | 0 B |
| FullLoad | 519.56/527.03/563.79 | 100.00% | 100.00% | 1.00 | 0 B |
| Prefix25 | 519.54/527.15/565.72 | 100.00% | 100.00% | 1.00 | 0 B |
| Prefix45 | 520.15/527.62/563.50 | 100.00% | 100.00% | 1.00 | 0 B |

Every arm performs one normal pixel-based multimodal inference at Turn 1. Prefix saliency and Visual KV are captured from that same forward; there is no separate vision or prefix recomputation.

## One-time persistence overhead

| Component | Mean | p50 | p95 |
|---|---:|---:|---:|
| saliency_extra | 0.01 | 0.01 | 0.01 |
| saliency_postprocess | 0.08 | 0.08 | 0.09 |
| permutation | 0.64 | 0.65 | 0.70 |
| kv_repack | 148.02 | 141.93 | 207.72 |
| buffered_ssd_write | 445.06 | 434.92 | 547.89 |
| fsync | 294.27 | 259.93 | 443.09 |
| total_persistence | 979.75 | 961.25 | 1239.38 |

The measured one-time write is 1191.74 MB/image on average. This cost is outside subsequent cache-hit request timing, but is retained here and in the secondary amortization analysis.
In the completed implementation it ran synchronously immediately after the source Answer 1; separating it here is an analysis boundary, not an overlap or background-execution claim.
The total also includes postprocessing, atomic publication, context open, and measured residual time not all shown as headline components; component percentiles must not be summed to reconstruct the total percentile.

## Turn-wise cache-hit TTFT

| Turn | History tokens | ReComp | FullLoad | Prefix25 | Prefix45 | ReComp-P25 | ReComp-P45 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 34.62 | 525.23 | 672.62 | 264.54 | 406.14 | 260.68 | 119.09 |
| 3 | 52.10 | 524.79 | 675.33 | 265.24 | 408.79 | 259.55 | 115.99 |
| 4 | 69.27 | 531.27 | 693.59 | 275.10 | 412.23 | 256.17 | 119.04 |
| 5 | 86.87 | 533.89 | 690.77 | 275.34 | 411.82 | 258.55 | 122.07 |
| 6 | 103.66 | 539.38 | 690.94 | 280.20 | 418.42 | 259.18 | 120.96 |
| 7 | 121.53 | 544.10 | 691.93 | 282.56 | 421.87 | 261.55 | 122.23 |
| 8 | 138.35 | 548.14 | 692.79 | 286.87 | 426.38 | 261.27 | 121.76 |
| 9 | 155.41 | 550.49 | 685.71 | 288.57 | 425.63 | 261.93 | 124.86 |
| 10 | 172.60 | 556.39 | 682.93 | 286.65 | 427.00 | 269.74 | 129.39 |

The absolute advantage remains present as history grows: ReComp-Prefix25 changes from 260.68 ms at Turn 2 to 269.74 ms at Turn 10; the corresponding Prefix45 values are 119.09 and 129.39 ms.

## Storage interpretation

FullLoad reads the complete Visual KV from SSD and is slower than ReComp on cache-hit turns. SSD caching alone therefore does not guarantee lower TTFT; reducing bytes read is necessary in this setup. ReComp performs zero SSD read/write, so no claim is made that Prefix uses less SSD I/O than ReComp.

## Secondary persistence/amortization result

This is not the main cache-hit table. In the conservative back-to-back case, one measured persistence event is charged once for N>=2.

- Prefix25 strict break-even: TTFT N=5, E2E N=5.
- Prefix45 strict break-even: TTFT N=10, E2E N=10.

## Paper-ready conclusion

On VisDial cache-hit turns 2--10, ImageOnly-Repack Prefix25 reduced end-to-end TTFT by 48.39% relative to recomputing the visual context, while reducing SSD bytes read by 73.76% relative to FullLoad, with an auxiliary-quality delta of -0.021. Prefix45 retained comparable normalized generative-match auxiliary quality and reduced end-to-end TTFT by 22.57% versus ReComp. FullLoad was slower than ReComp, showing that SSD caching alone is insufficient when the entire Visual KV is read per request. The one-time cache-persistence cost is reported separately and is not included in these cache-hit request latencies.

## Validation and artifact semantics

All 52 fail-closed checks pass. The input raw run and prior result tree are hashed before and after analysis. Main tables are recomputed from raw rows rather than copied or hardcoded.

- `main_cache_hit_table.csv`: paper main table, Turns 2--10 only.
- `turn1_sanity.csv`: separate Turn-1 fairness/capture evidence.
- `persistence_overhead.csv`: separate one-time cost.
- `ttft_by_turn_cache_hit.csv`: detailed long-form trend.
- `quality_by_turn_cache_hit.csv`: wide auxiliary-quality trend.
- `io_breakdown_cache_hit.csv`: actual pread/scatter/prefill accounting.
- `fig_cache_hit_ttft_by_turn.csv`: paper-figure-ready wide TTFT data.
