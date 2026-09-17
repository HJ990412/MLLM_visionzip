# TextVQA ImageOnly-Repack generalization

## Scope and immutable workload

- Index: `/home/dblab/hj/mllm_v2/data/textvqa/index.json`
- Index SHA256: `b1e5ff0eaba2a45c6398e7cb90631cdc7387eff25ed0997a66374f25968a2f4d`
- Evaluation workload SHA256: `49fa0b15f406132162cba1b47245f28a560d28b4f3af76985c855d77a145a2a6`
- Workload: 500 images / 500 questions; fixed
  slice `questions[1:2]`
- Calibration questions used by the new layout: **0**.  The skipped question is
  retained only to match the old evaluation IDs.
- Main latency: `end_to_end_ttft_ms`; `core_ttft_ms` is diagnostic only.

## Main results

| Method | Score | Δ vs FullLoad | TTFT mean / p50 / p95 | TTFT↓ vs ReComp | E2E | SSD MB/req | SSD ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 61.93% | 0.00 pp | 538.40 / 529.44 / 716.18 ms | 0.00% | 594.51 ms | 0.00 | n/a |
| FullLoad | 61.93% | 0.00 pp | 758.09 / 739.72 / 941.62 ms | -40.80% | 818.67 ms | 1207.53 | 100.00% |
| Prefix25 | 57.87% | -4.07 pp | 289.29 / 288.75 / 350.78 ms | 46.27% | 350.46 ms | 319.03 | 26.42% |
| Prefix45 | 62.20% | 0.27 pp | 437.34 / 432.26 / 569.31 ms | 18.77% | 496.51 ms | 572.37 | 47.40% |

Prefix25 changes quality by -4.07 pp
versus same-layout FullLoad (paired image-cluster 95% CI
[-6.73,
 -1.40] pp), reduces
E2E-TTFT by 46.27% versus ReComp,
and reduces SSD bytes by 73.58%
versus FullLoad.

Prefix45 changes quality by 0.27 pp
(95% CI [-1.47,
2.07] pp), reduces E2E-TTFT
by 18.77%, and reduces SSD bytes by
52.60%.

## FullLoad numerical sanity

- Prediction agreement with ReComp: 98.00%
- First-token agreement with ReComp: 99.20%
- Score delta FullLoad−ReComp: 0.00 pp

This separates the physical-layout numerical effect (ReComp → repacked
FullLoad) from the partial-loading effect (repacked FullLoad → Prefix).

## Persistence and layout coverage

- Saliency: 39.92 ms/image
- Permutation: 0.75 ms/image
- Repack: 148.79 ms/image
- Buffered SSD write: 472.13 ms/image
- `fsync`: 663.82 ms/image
- Mapping metadata: 0.07 MB/image
- Visual KV: 1207.53 MB/image
- Prefix25/45 saliency-mass coverage: 77.45% /
  91.03%

Persistence is cache construction overhead and is not part of cache-hit TTFT.

## Category analysis

Categories use the existing ordered question-text regex heuristic: yes/no,
color, count, spatial, material/attribute, object, then fallback other. They
are not official dataset metadata. Every row belongs to the OCR-focused TextVQA workload; these lexical question-form facets are not official OCR/non-OCR annotations.
Prefix25's most negative category delta is
`yes/no`
(-8.97 pp,
n=26); Prefix45's is
`count`
(-6.67 pp).
See `category_analysis.csv` and `disagreement_analysis.csv` for all counts.

## Statistical policy and metric caveat

Quality intervals use 10,000 seeded paired
image-cluster resamples.  GQA also receives exact McNemar testing.  VQAv2 and
TextVQA retain the repository's normalized 10-answer consensus score and use
paired soft-score deltas rather than thresholded McNemar as their primary test.
The repository VQA normalizer/scorer is not byte-for-byte the official
leave-one-annotator-out evaluator; it is retained to preserve question-level
comparability with the previous cross-dataset experiment.  The repository GQA
metric also accepts a prediction beginning with the complete normalized gold
token sequence.

## Verdict

**SUPPORTED** under the explicit rule recorded in `config.json`.
Automated validation: 42/42 checks
passed.
