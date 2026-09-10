# Supplementary calib=1 Reorder+Prefix control

Validation: **PASS**

This is the frozen GQA 40-image / 240-question sensitivity run at a 25%
normal-chunk budget. It is supplementary and does not replace the preregistered
calib=4 primary comparison.

## Results

| Method | Accuracy | Δ vs FullLoad | TTFT mean / p50 / p95 | SSD MB/request | SSD/Full | Selector |
|---|---:|---:|---:|---:|---:|---:|
| FullLoad | 62.08% | +0.00 pp | 695.21 / 691.24 / 844.19 ms | 1165.073 | 100.00% | - |
| Reorder + Prefix 25% | 58.33% | -3.75 pp | 281.64 / 282.31 / 322.57 ms | 306.079 | 26.27% | 0.220 ms |
| Reorder + Static+Diverse 25% | 57.92% | -4.17 pp | 349.86 / 346.39 / 403.86 ms | 300.108 | 25.76% | 14.242 ms |

Static+Diverse - Prefix accuracy is **-0.42 pp**. The
primary 95% image-cluster paired-bootstrap CI is
**[-3.33, +2.50] pp** (40 image clusters, 10,000
resamples, seed 0). The supplementary question bootstrap is also recorded in
`paired_stats.json`.

Exact McNemar discordances are SD-only **5** and Prefix-only
**6** (two-sided p=1). Thus this
calib=1 run provides no observed positive selection gain: Prefix is
58.33% and Static+Diverse is 57.92%.
This statement is descriptive; a confidence interval containing zero is not
evidence of equivalence.

## Chunk overlap

Across 1280 image-layer pairs, mean/median Jaccard is
**0.5330/0.5556**. Mean
intersection, SD-only, and Prefix-only counts are
5.888, 2.638, and
2.638 chunks.

## Calibration provenance limitation

- Intended calib=1 IDs are the first question of each frozen image; their
  ordered SHA256 is `ba6b7e06c31853e2ef6009627026674b0e54e1b40d77e0def59c7c60be148438`. Evaluation remains the
  disjoint `[4:10]` slice.
- The store metadata records the resulting per-layer permutations but does not
  self-record `calibration_questions=1` or the calibration-ID hash.
- Construction provenance states that `kvstore_reorder_prefix_calib1` was
  composed from a copy of the calib=4 store and then processed for calib=1.
  It was not independently rebuilt from a freshly generated canonical raster
  store. Consequently, when importance values tie, bitwise equality of token
  tie-order with a fresh raster-to-calib1 build is **not guaranteed**.
- Therefore this is a useful sensitivity check of the observed composed
  layout, not cryptographic proof of a uniquely reproducible fresh calib=1
  layout.

## Measurement and validation

True TTFT keeps the schema-v2 boundary through synchronized first-token
determination. `validation.json` checks the exact arms and paired workload,
frozen question/gold content, store geometry and permutations, first-k Prefix
semantics, selector counters, common separator sidecar, real recorded pread
bytes/counts, budgets, and `E2E = TTFT + decode`.

Source run: `/home/dblab/hj/mllm_v2/runs/reorder_prefix_baseline/calib1_pair25`  
Store: `/home/dblab/hj/mllm_v2/kvstore_reorder_prefix_calib1`  
Source results SHA256: `598bc3384205fef6da8ae2d61c250f77c5b0363ea68333b721791bcb1893c851`
