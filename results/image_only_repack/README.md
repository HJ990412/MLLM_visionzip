# Image-only VisionZip repack analysis

This is a strict, CPU-only analysis of the frozen GQA-40 paired workload (questions[4:10], 240 requests). Source runs and historical results are read-only; this directory was published without overwrite.

## Strict correctness warning

**WARNING — strict FullLoad exactness FAILS.** Raster and repacked FullLoad prediction/first-token agreement is 99.17% / 99.17%, with 2 disagreements and a repacked-minus-raster accuracy delta of -0.83 pp. This is never labeled an exact correctness pass.

Mapped FP16 Visual-KV structural integrity is **PASS**: 1,175,453,696 elements were compared and 0 differed; `sys_kv` and `v_hidden` hashes also match. The audit classifies the two greedy flips as BF16 eager-attention reduction-order sensitivity, not a mapped-K/V corruption. See `validation.json`.

## Research-method decision

**PARTIAL** — STRONG iff system gate, the same-layout FullLoad and calib4 quality drops are both <=2 pp, and VisionZip25 is no worse than best Raster/Morton Prefix25; PARTIAL iff system gate and VisionZip25 gains >=2 pp over the best Raster/Morton Prefix25 but misses STRONG; otherwise NO-GO.

Observed VisionZip Prefix25: accuracy drop vs same-layout FullLoad 3.33 pp (vs canonical Raster FullLoad 4.17 pp); gain vs best Raster/Morton Prefix25 55.00 pp; SSD ratio vs repacked FullLoad 0.263; TTFT reduction vs repacked FullLoad 60.30%.

Thresholds are predeclared in `config.json`; they are operational decision thresholds, not learned equivalence margins.

## Main paired results

| Layout / arm | Accuracy | TTFT mean (ms) | SSD MB/request | preads (normal/sep/total) | selector mean/p95 (ms) |
|---|---:|---:|---:|---:|---:|
| Raster / FullLoad | 62.08% | 782.11 | 1165.07 | 64.0/0.0/64.0 | NA/NA |
| Raster / Prefix25 | 2.92% | 289.06 | 306.08 | 64.0/1.0/65.0 | 0.210/0.278 |
| Morton / Prefix25 | 2.92% | 289.24 | 306.08 | 64.0/1.0/65.0 | 0.226/0.286 |
| Calib4 importance legacy / Prefix25 | 60.42% | 262.16 | 306.08 | 64.0/1.0/65.0 | 0.194/0.243 |
| VisionZip image-only / Prefix25 | 57.92% | 284.74 | 306.08 | 64.0/1.0/65.0 | 0.212/0.264 |
| VisionZip image-only / Prefix50 | 60.00% | 460.86 | 603.87 | 64.0/1.0/65.0 | 0.331/0.367 |

The VisionZip-layout FullLoad arm is sanity-only. Raster and repacked FullLoad prediction/first-token agreement were both 99.2% / 99.2% over 240 requests. The optional matched-calibration separator-tail arm is included.

Paired image-cluster bootstrap (10,000 resamples, seed 0), a question-level bootstrap supplement, and exact two-sided McNemar results are embedded in `config.json`. Accuracy is binary GQA normalized exact match.

## 25% prefix importance coverage

| Layout | Importance source | Normal-token fraction | Coverage macro | Coverage global-mass |
|---|---|---:|---:|---:|
| calib4_importance_legacy | sparsevlm_calib4_analysis_only | 23.89% | 73.50% | 61.73% |
| calib4_importance_legacy | visionzip_image_saliency | 23.89% | 50.53% | 50.68% |
| calib4_importance_sep_tail | sparsevlm_calib4_analysis_only | 24.99% | 74.53% | 63.10% |
| calib4_importance_sep_tail | visionzip_image_saliency | 24.99% | 51.87% | 52.01% |
| morton | sparsevlm_calib4_analysis_only | 24.99% | 24.99% | 24.02% |
| morton | visionzip_image_saliency | 24.99% | 28.54% | 28.43% |
| raster | sparsevlm_calib4_analysis_only | 24.98% | 25.27% | 24.20% |
| raster | visionzip_image_saliency | 24.98% | 29.56% | 29.48% |
| visionzip_image_only | sparsevlm_calib4_analysis_only | 24.99% | 41.60% | 36.74% |
| visionzip_image_only | visionzip_image_saliency | 24.99% | 73.15% | 73.27% |

`importance_coverage.csv` is a byte-identical copy of the supplied long-form artifact. SparseVLM calib4 importance is analysis-only and was not used to build or serve image-only layouts.

## One-time build cost

Mean incremental VisionZip ingestion cost versus raster was 138.87 ms/image. `layout_stats.csv` reports the observed one-time components and amortization over 1, 5, 10, and 20 requests/image.

## Validation scope

Every arm has the exact same 40 images and 240 question/gold pairs; schema-v2 true TTFT, cold-cache mode, chunk size 64, sidecar separator semantics, Prefix first-k selections, zero scoring counters, measured byte/pread splits, profile provenance, profile/coverage geometry, and the independent validation artifact were checked fail-closed. No store tree or model/GPU serving is performed by this analyzer.
