# GQA-large ImageOnly-Repack generalization

## Scope and immutable workload

- Index: `/home/dblab/hj/mllm_v2/data/gqa_large/index.json`
- Index SHA256: `0d50962f0c1bac3bc6e1836978289d5fde7d60b434f55cebbc4607c4c80a797c`
- Evaluation workload SHA256: `cabec1bb1035c836839b98d72ba6a04558529d246cf55f8499ca75ea2d2f290c`
- Workload: 395 images / 1185 questions; fixed
  slice `questions[1:4]`
- Calibration questions used by the new layout: **0**.  The skipped question is
  retained only to match the old evaluation IDs.
- Main latency: `end_to_end_ttft_ms`; `core_ttft_ms` is diagnostic only.

## Main results

| Method | Score | Δ vs FullLoad | TTFT mean / p50 / p95 | TTFT↓ vs ReComp | E2E | SSD MB/req | SSD ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 63.80% | 0.34 pp | 514.93 / 519.32 / 576.20 ms | 0.00% | 541.21 ms | 0.00 | n/a |
| FullLoad | 63.46% | 0.00 pp | 750.11 / 746.04 / 887.42 ms | -45.67% | 781.78 ms | 1170.08 | 100.00% |
| Prefix25 | 58.73% | -4.73 pp | 286.01 / 286.38 / 337.98 ms | 44.46% | 314.03 ms | 306.67 | 26.21% |
| Prefix45 | 62.87% | -0.59 pp | 430.85 / 428.97 / 520.35 ms | 16.33% | 458.63 ms | 553.70 | 47.32% |

Prefix25 changes quality by -4.73 pp
versus same-layout FullLoad (paired image-cluster 95% CI
[-6.67,
 -2.78] pp), reduces
E2E-TTFT by 44.46% versus ReComp,
and reduces SSD bytes by 73.79%
versus FullLoad.

Prefix45 changes quality by -0.59 pp
(95% CI [-1.86,
0.68] pp), reduces E2E-TTFT
by 16.33%, and reduces SSD bytes by
52.68%.

## FullLoad numerical sanity

- Prediction agreement with ReComp: 99.07%
- First-token agreement with ReComp: 99.07%
- Score delta FullLoad−ReComp: -0.34 pp

This separates the physical-layout numerical effect (ReComp → repacked
FullLoad) from the partial-loading effect (repacked FullLoad → Prefix).

## Persistence and layout coverage

- Saliency: 39.20 ms/image
- Permutation: 0.70 ms/image
- Repack: 140.43 ms/image
- Buffered SSD write: 454.63 ms/image
- `fsync`: 465.40 ms/image
- Mapping metadata: 0.07 MB/image
- Visual KV: 1170.08 MB/image
- Prefix25/45 saliency-mass coverage: 73.25% /
  88.85%

Persistence is cache construction overhead and is not part of cache-hit TTFT.

## Category analysis

Categories use the existing ordered question-text regex heuristic: yes/no,
color, count, spatial, material/attribute, object, then fallback other. They
are not official dataset metadata. Exploratory GQA facets: yes/no, color, count, spatial, material/attribute, object, and fallback other.
Prefix25's most negative category delta is
`color`
(-13.43 pp,
n=67); Prefix45's is
`color`
(-2.99 pp).
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
