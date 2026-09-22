# ConvBench evaluation analysis

Validated 577 native three-turn conversations, 6924 logical method-turn requests, and 1154 individual cache-hit requests per method.

The cache-hit TTFT is the pooled distribution of independent Turn-2 and Turn-3 request start-to-first-token intervals. It is not session cumulative latency. OS page-cache conditioning via posix_fadvise(DONTNEED) is outside each timer; buffered pread is used and SSD controller cache is not explicitly flushed.

## System results

| Method | T2 mean (ms) | T3 mean (ms) | Pooled cache-hit mean (ms) | p50 | p95 | SSD MB/request |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 566.45 | 628.46 | 597.46 | 584.69 | 838.97 | 0.00 |
| FullLoad | 757.62 | 760.46 | 759.04 | 758.09 | 949.22 | 1152.65 |
| Prefix25 | 322.40 | 366.39 | 344.39 | 330.73 | 465.45 | 305.72 |
| Prefix45 | 467.43 | 511.81 | 489.62 | 481.46 | 651.53 | 545.20 |

## One-time persistence

Mean synchronous persistence: 1159.98 ms/image. It is excluded from cache-hit TTFT and included once in the secondary session cumulative E2E table.

Generated answers form each method's later-turn history. Their lengths may differ and affect TTFT. The cache methods share one physical image-only layout captured during a designated normal Turn-1 request. The session table charges that measured capture request plus persistence to each cache-method counterfactual.

Source and cap details are in the run configs, dataset provenance, raw per-request rows, and validation.json.

The raw per-request rows are published losslessly as `raw.jsonl.gz`; restore them with
`gzip -dc raw.jsonl.gz > raw.jsonl`.

## Context cap

The nominal output cap was reduced in 444 of 6924 requests; 217 requests reached their effective cap. See context_cap_summary.csv by method and turn.

Frozen policy `native_overflow_no_truncation_v1` processed 26 inputs above the nominal 4096-token limit without truncation or RoPE configuration changes. Inputs within the nominal limit retained the dynamic output cap; longer inputs used the nominal generation cap. Successful first-token records and zero text truncation were validated for every request. See context_policy_per_request.csv for original/final input lengths and positional overflow.
