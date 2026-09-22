# ConvBench final diagnostics

Validated 577/577 conversations, 6924/6924 generation requests, and 6924/6924 stage judgments.

The S1/S2/S3 denominator is fixed at the full conversation count per method. Unresolved decisions remain in the denominator and contribute zero model wins. Avg excludes any overall-conversation judgment. S1 is an official stage-wise conversation score whose prompt includes all three responses; it is not pure A1 accuracy.

## Quality (0–100)

| Method | S1 | S2 | S3 | Avg | ΔAvg vs ReComp | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 7.28 | 7.28 | 4.33 | 6.30 | +0.00 | 139 |
| FullLoad | 7.97 | 7.63 | 5.72 | 7.11 | +0.81 | 131 |
| Prefix25 | 7.97 | 6.59 | 3.29 | 5.95 | -0.35 | 131 |
| Prefix45 | 9.01 | 7.11 | 5.72 | 7.28 | +0.98 | 142 |

## System

| Method | T2 mean | T3 mean | Pooled mean | Pooled p50 | Pooled p95 | Δ TTFT vs ReComp | SSD MB/request | SSD ↓ vs FullLoad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 566.45 | 628.46 | 597.46 | 584.69 | 838.97 | +0.00% | 0.00 | 100.00% |
| FullLoad | 757.62 | 760.46 | 759.04 | 758.09 | 949.22 | -27.04% | 1152.65 | 0.00% |
| Prefix25 | 322.40 | 366.39 | 344.39 | 330.73 | 465.45 | +42.36% | 305.72 | 73.48% |
| Prefix45 | 467.43 | 511.81 | 489.62 | 481.46 | 651.53 | +18.05% | 545.20 | 52.70% |

## Turn-specific TTFT distribution (ms)

| Method | T2 p50 | T2 p95 | T3 p50 | T3 p95 |
|---|---:|---:|---:|---:|
| ReComp | 561.39 | 800.24 | 618.73 | 883.44 |
| FullLoad | 756.77 | 935.39 | 761.05 | 962.73 |
| Prefix25 | 314.04 | 403.44 | 351.64 | 498.78 |
| Prefix45 | 463.61 | 629.54 | 502.07 | 679.72 |

T2 and T3 timings are independent request start-to-first-token intervals; the pooled cache-hit statistic combines their individual observations. Persistence is a separate one-time synchronous cost and is included in the 3-turn cumulative session latency table. The OS page cache was conditioned with posix_fadvise(DONTNEED) outside the timer, using buffered pread. SSD controller cache was not explicitly flushed.

## Diagnostics

First-pass parsed: 6284; second-pass recovered: 97; unresolved: 543 (7.84%).
Turn-1 generation agreement: 577/577. Full pixel/prompt/token agreement: 577/577.
Conservative clear-repetition flags: 50 requests. Outputs were not altered or regenerated.
Inputs above 4096 tokens: 26; maximum input: 5138 tokens.

Unique images: 573; persisted conversation-store instances: 577. The latter counts independent conversation executions when an image appears more than once.
Judge checkpoint files verified against raw records: 2308.

The paired bootstrap uses complete conversation clusters, with 10,000 resamples. Quality ΔAvg and TTFT reductions are relative to ReComp. The 3-turn session table includes synchronous persistence for cache methods. Detailed per-turn TTFT, judge, repetition, context, persistence, and paired confidence interval tables are adjacent CSV files.
