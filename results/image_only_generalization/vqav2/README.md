# VQAv2 ImageOnly-Repack generalization

## Scope and immutable workload

- Index: `/home/dblab/hj/mllm_v2/data/vqav2/index.json`
- Index SHA256: `b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7`
- Evaluation workload SHA256: `e341b499a968c5caba4ddffc58fdf0ccafe2e0e212b6cb280ca1b11fc9f31d18`
- Workload: 250 images / 1000 questions; fixed
  slice `questions[1:5]`
- Calibration questions used by the new layout: **0**.  The skipped question is
  retained only to match the old evaluation IDs.
- Main latency: `end_to_end_ttft_ms`; `core_ttft_ms` is diagnostic only.

## Main results

| Method | Score | Δ vs FullLoad | TTFT mean / p50 / p95 | TTFT↓ vs ReComp | E2E | SSD MB/req | SSD ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 81.57% | 0.00 pp | 516.59 / 510.56 / 687.02 ms | 0.00% | 547.50 ms | 0.00 | n/a |
| FullLoad | 81.57% | 0.00 pp | 761.44 / 753.60 / 919.36 ms | -47.40% | 797.28 ms | 1176.49 | 100.00% |
| Prefix25 | 76.33% | -5.23 pp | 286.72 / 285.14 / 354.36 ms | 44.50% | 319.28 ms | 309.01 | 26.27% |
| Prefix45 | 80.50% | -1.07 pp | 434.75 / 427.99 / 555.91 ms | 15.84% | 467.40 ms | 555.70 | 47.23% |

Prefix25 changes quality by -5.23 pp
versus same-layout FullLoad (paired image-cluster 95% CI
[-7.03,
 -3.50] pp), reduces
E2E-TTFT by 44.50% versus ReComp,
and reduces SSD bytes by 73.73%
versus FullLoad.

Prefix45 changes quality by -1.07 pp
(95% CI [-2.07,
-0.07] pp), reduces E2E-TTFT
by 15.84%, and reduces SSD bytes by
52.77%.

## FullLoad numerical sanity

- Prediction agreement with ReComp: 99.20%
- First-token agreement with ReComp: 100.00%
- Score delta FullLoad−ReComp: -0.00 pp

This separates the physical-layout numerical effect (ReComp → repacked
FullLoad) from the partial-loading effect (repacked FullLoad → Prefix).

## Persistence and layout coverage

- Saliency: 39.22 ms/image
- Permutation: 0.71 ms/image
- Repack: 142.48 ms/image
- Buffered SSD write: 464.42 ms/image
- `fsync`: 538.16 ms/image
- Mapping metadata: 0.07 MB/image
- Visual KV: 1176.49 MB/image
- Prefix25/45 saliency-mass coverage: 75.02% /
  89.74%

Persistence is cache construction overhead and is not part of cache-hit TTFT.

## Category analysis

Categories use the existing ordered question-text regex heuristic: yes/no,
color, count, spatial, material/attribute, object, then fallback other. They
are not official dataset metadata. The yes/no and count buckets approximate VQAv2 yes/no and number; the remaining lexical buckets are exploratory subdivisions of the official-style other class.
Prefix25's most negative category delta is
`other`
(-9.52 pp,
n=56); Prefix45's is
`spatial`
(-2.15 pp).
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

**PARTIALLY SUPPORTED** under the explicit rule recorded in `config.json`.
Automated validation: 42/42 checks
passed.
