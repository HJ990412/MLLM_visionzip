# VisDial Turn-1 Piggyback — End-to-End TTFT

Validation: **PASS**  
Claim verdict: **SUPPORTED**

## Timing boundary (main metric)

`end_to_end_ttft_ms` is the main TTFT metric.  OS page-cache conditioning is
completed first and is outside the request timer.  The timer then starts
immediately before prompt construction/tokenization and includes input
preparation, initial CPU→GPU H2D, either vision recomputation or SSD KV loading,
cache reconstruction/scatter, prefill, first-token selection, and the
first-token CUDA synchronization.  `core_ttft_ms` begins only after initial
request preparation/H2D and is retained as a diagnostic for comparison with
older runs.  `request_e2e_ms` continues through decode and the common
postprocessing boundary. Per-phase `vision_ms` is a CUDA-event device-timeline
diagnostic on GPU runs, whereas the server-side TTFT boundary is the synchronized
wall-clock request timestamp; phase sums therefore remain diagnostic.

Cold-cache experiments evict the OS page cache before request timing using
`posix_fadvise(DONTNEED)`. This cache-conditioning step is excluded from TTFT
because it is benchmark setup rather than request processing. Reads use
buffered `pread`; `O_DIRECT` is not used, and the SSD controller cache is not
forcibly flushed.  Consequently this is an OS-page-cache cold condition, not a
claim of a physically “true cold SSD”.

## Workload and controls

- Dataset: VisDial v1.0 validation, exactly 100 dialogs / 100 unique images /
  10 turns per dialog / 1,000 requests per method.
- Frozen index SHA256: `8c3dd7e983cb39e61d26362a0353b86ac84845bd7537a6331078ab7707777383`
- Frozen request-key SHA256: `395ba928a15eb45bca905aa91d1ec89617981f18b9b22338341b2a189b3f141a`
- History: gold teacher-forced; per-request history and token hashes are equal
  across methods.
- Method order: `cyclic deterministic rotation by (zero_based_dialog_index + seed) modulo number of methods; same order at every turn of a dialog`. Position balance and the exact recorded order
  are validated.
- Methods: ReComp, FullLoad, Prefix25, Prefix45. No query-dependent selector, Static+Diverse,
  SparseVLM, or MaxMin is present.

## Overall (all turns)

| Method | Quality† | End-to-end TTFT mean/p50/p95 (ms) | Core TTFT mean (ms) | Request E2E mean (ms) | SSD read/request (MB) |
|---|---:|---:|---:|---:|---:|
| ReComp | 0.452 | 537.31/547.26/593.78 | 521.95 | 569.47 | 0.00 |
| FullLoad | 0.453 | 669.62/672.03/819.84 | 667.65 | 706.40 | 1052.49 |
| Prefix25 | 0.433 | 302.46/281.92/526.91 | 300.54 | 347.65 | 276.15 |
| Prefix45 | 0.453 | 427.84/426.04/548.61 | 425.92 | 467.56 | 499.32 |

† Quality is the existing normalized generative-match auxiliary metric, not an
official VisDial MRR/R@K/Mean Rank/NDCG score.

## Turn 1 and shared persistence semantics

Every method performs normal pixel-based multimodal inference at Turn 1.  The
Prefix methods do not start with a pre-existing SSD cache. Visual KV and
penultimate image-only saliency are captured from that same Turn-1 forward;
there is no second vision forward and no separate cache-build inference. The
one persistence event per image is shared by Prefix25 and Prefix45—it is not
duplicated per budget. `saliency_extra_ms` is Turn-1 capture instrumentation
overhead and is not added again to `persist_ms`.

Only one Prefix execution per dialog retains the already-created cache object
for the physical persistence event (source counts: `{'Prefix25': 25, 'Prefix45': 75}`).
Both Prefix arms use the same saliency instrumentation, and each counterfactual
cumulative curve uses that method's own measured normal Turn-1 request plus the
one shared persistence measurement. Cache-object export occurs after the
first-token boundary; this source choice therefore cannot improve reported
TTFT, while any small post-first-token bookkeeping difference remains visible
in request E2E.

| Method | Turn-1 E2E TTFT mean | p50 | p95 | vision mean |
|---|---:|---:|---:|---:|
| ReComp | 519.42 | 525.40 | 567.87 | 38.19 |
| FullLoad | 519.56 | 527.03 | 563.79 | 38.21 |
| Prefix25 | 519.54 | 527.15 | 565.72 | 38.18 |
| Prefix45 | 520.15 | 527.62 | 563.50 | 38.37 |

| Accounting component | Boundary | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---:|---:|---:|
| saliency_extra_ms | inside Turn-1 TTFT; not added to persist | 0.01 | 0.01 | 0.01 |
| saliency_postprocess_ms | post-response persistence path | 0.08 | 0.08 | 0.09 |
| permutation_ms | post-response persistence path | 0.64 | 0.65 | 0.70 |
| repack_ms | post-response persistence path | 148.02 | 141.93 | 207.72 |
| buffered_write_ms | post-response persistence path | 445.06 | 434.92 | 547.89 |
| fsync_ms | post-response persistence path | 294.27 | 259.93 | 443.09 |
| context_open_ms | post-response persistence path | 6.34 | 6.24 | 7.90 |
| persist_ms | post-response persistence path | 979.75 | 961.25 | 1239.38 |

Mean one-time SSD write: 1191.74 MB/image. ReComp performs no SSD read or
write. The correct claim is that the proposed path spends SSD traffic to avoid
repeated vision recomputation—not that it uses less SSD I/O than ReComp.

## Reuse turns (Turn 2–10)

| Method | E2E TTFT mean (ms) | Δ vs ReComp (ms) | reduction vs ReComp | quality† | SSD read/request (MB) |
|---|---:|---:|---:|---:|---:|
| ReComp | 539.30 | 0.00 | 0.00% | 0.439 | 0.00 |
| FullLoad | 686.29 | 146.99 | -27.26% | 0.440 | 1169.43 |
| Prefix25 | 278.34 | -260.96 | 48.39% | 0.418 | 306.83 |
| Prefix45 | 417.59 | -121.71 | 22.57% | 0.440 | 554.80 |


### Per-turn end-to-end TTFT trend

| Turn | ReComp | Prefix25 | Prefix45 | ReComp−P25 | ReComp−P45 |
|---:|---:|---:|---:|---:|---:|
| 1 | 519.42 | 519.54 | 520.15 | -0.12 | -0.73 |
| 2 | 525.23 | 264.54 | 406.14 | 260.68 | 119.09 |
| 3 | 524.79 | 265.24 | 408.79 | 259.55 | 115.99 |
| 4 | 531.27 | 275.10 | 412.23 | 256.17 | 119.04 |
| 5 | 533.89 | 275.34 | 411.82 | 258.55 | 122.07 |
| 6 | 539.38 | 280.20 | 418.42 | 259.18 | 120.96 |
| 7 | 544.10 | 282.56 | 421.87 | 261.55 | 122.23 |
| 8 | 548.14 | 286.87 | 426.38 | 261.27 | 121.76 |
| 9 | 550.49 | 288.57 | 425.63 | 261.93 | 124.86 |
| 10 | 556.39 | 286.65 | 427.00 | 269.74 | 129.39 |

## Two-turn and ten-turn cumulative latency

Cumulative TTFT below is explicitly the **sum of per-request TTFTs**, not the
wall-clock completion time of a multi-request session. Cumulative E2E is also
formed as a per-request service-time sum: `worst` is the back-to-back completion
case, while `hidden` is an ideal service-latency reference and is not a measured
session wall clock (unmeasured user think time is not added). Values are
calculated per dialog first and then aggregated across dialogs. At N=1, both
Prefix scenarios leave persistence uncharged. At N≥2, `worst` adds `persist_ms`
exactly once; `hidden` excludes it.

### Cumulative end-to-end TTFT — N=2

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
| ReComp | recompute_no_persistence | 1044.64 | 1079.71 | 1133.02 | 0.00% |
| Prefix25 | worst | 1763.83 | 1759.65 | 2116.19 | -68.84% |
| Prefix45 | worst | 1906.04 | 1900.37 | 2239.96 | -82.46% |
| Prefix25 | hidden | 784.08 | 784.19 | 875.81 | 24.94% |
| Prefix45 | hidden | 926.29 | 935.93 | 1030.46 | 11.33% |

### Cumulative end-to-end TTFT — N=10

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
| ReComp | recompute_no_persistence | 5373.09 | 5586.16 | 5770.32 | 0.00% |
| Prefix25 | worst | 4004.34 | 3982.01 | 4666.47 | 25.47% |
| Prefix45 | worst | 5258.17 | 5340.91 | 5963.03 | 2.14% |
| Prefix25 | hidden | 3024.60 | 2992.01 | 3414.56 | 43.71% |
| Prefix45 | hidden | 4278.42 | 4362.02 | 4821.62 | 20.37% |

### Scenario-adjusted cumulative E2E — N=2

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
| ReComp | recompute_no_persistence | 1107.52 | 1122.38 | 1273.47 | 0.00% |
| Prefix25 | worst | 1831.58 | 1811.65 | 2156.10 | -65.38% |
| Prefix45 | worst | 1970.28 | 1963.41 | 2279.66 | -77.90% |
| Prefix25 | hidden | 851.83 | 840.89 | 1063.04 | 23.09% |
| Prefix45 | hidden | 990.54 | 990.10 | 1149.64 | 10.56% |

### Scenario-adjusted cumulative E2E — N=10

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
| ReComp | recompute_no_persistence | 5694.66 | 5808.07 | 6418.87 | 0.00% |
| Prefix25 | worst | 4456.20 | 4425.70 | 5108.28 | 21.75% |
| Prefix45 | worst | 5655.39 | 5665.23 | 6478.01 | 0.69% |
| Prefix25 | hidden | 3476.46 | 3477.11 | 4093.01 | 38.95% |
| Prefix45 | hidden | 4675.65 | 4687.80 | 5422.50 | 17.89% |

## Break-even on aggregate mean curves

The break-even turn is the smallest strict `N >= 2` for which Prefix is lower
than paired ReComp. `censored >10` means no crossing was observed within this
10-turn workload. Per-dialog distributions are preserved in the CSV files.

### TTFT sum

| Method | Scenario | break-even N | stays better through N=10 | N=2 delta (ms) | N=10 delta (ms) |
|---|---|---:|---:|---:|---:|
| Prefix25 | hidden | 2 | True | -260.56 | -2348.49 |
| Prefix25 | worst | 5 | True | 719.18 | -1368.75 |
| Prefix45 | hidden | 2 | True | -118.35 | -1094.67 |
| Prefix45 | worst | 10 | True | 861.39 | -114.92 |

### Scenario-adjusted E2E sum

| Method | Scenario | break-even N | stays better through N=10 | N=2 delta (ms) | N=10 delta (ms) |
|---|---|---:|---:|---:|---:|
| Prefix25 | hidden | 2 | True | -255.69 | -2218.20 |
| Prefix25 | worst | 5 | True | 724.06 | -1238.45 |
| Prefix45 | hidden | 2 | True | -116.98 | -1019.01 |
| Prefix45 | worst | 10 | True | 862.76 | -39.26 |

## Claim verdict

**SUPPORTED.** Machine-readable grounds: `{"aggregate_worst_case_break_even_turn": {"Prefix25": 5, "Prefix45": 10}, "ordered_quality_latency_tradeoff_25_to_45": true, "reuse_end_to_end_ttft_lower_than_recomp": {"Prefix25": true, "Prefix45": true}}`.
This verdict follows the measured data, including persistence on the
conservative path; the ideal hidden scenario is kept separate.

## Artifact semantics

- `summary.csv`: overall and reuse-only method summaries.
- `per_turn.csv` / `quality_by_turn.csv`: Turn 1…10 statistics.
- `per_dialog.csv`: paired dialog-level request sums and direct N=2/N=10 values.
- `cumulative_ttft.csv`: per-request TTFT sums after dialog-first aggregation.
- `cumulative_e2e.csv`: scenario-adjusted sums of per-request E2E after
  dialog-first aggregation (`worst` is back-to-back; `hidden` is ideal, not
  session wall clock).
- `break_even_*.csv`: dialog-level and separate aggregate-mean-curve crossings.
- `persistence_per_image.csv`: the immutable measured CSV copied byte-for-byte,
  or a lossless CSV materialization when the append-only JSONL fallback is the
  only completed source.
