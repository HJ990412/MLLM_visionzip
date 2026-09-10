# VisDial v1.0 official candidate-ranking quality run

- Schema: `visdial-official-quality-v1`
- Dialogs / evaluated rounds: 2 / 20
- History: `gold_teacher_forced`
- Calibration: `caption_only_pre_dialog`
- Candidate count per round: 100
- Validation: `PASS`
- System latency measurements: none (quality-only run)

## Overall

| Method | MRR | R@1 | R@5 | R@10 | Mean Rank | NDCG | Logical KV | Touched chunks | Sparse / dense rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.6164 | 0.5500 | 0.7000 | 0.8000 | 9.45 | 0.4020 | -- | -- | 20 / 2 |
| FullLoad | 0.6151 | 0.5500 | 0.7000 | 0.8000 | 9.55 | 0.4020 | 1.0000 | 1.0000 | 20 / 2 |
| SparseVLM 25% | 0.6124 | 0.5500 | 0.6500 | 0.8000 | 9.85 | 0.4007 | 0.2462 | 0.6776 | 20 / 2 |
| Static+Diverse 25% | 0.5819 | 0.5000 | 0.6500 | 0.8000 | 9.55 | 0.3793 | 0.2425 | 0.2393 | 20 / 2 |
| Static+Diverse 50% | 0.6150 | 0.5500 | 0.7000 | 0.8000 | 9.70 | 0.4003 | 0.4864 | 0.4932 | 20 / 2 |

## Protocol

Every round uses the image, caption, and identical gold teacher-forced history. Each of the 100 source-order candidates is scored by the unnormalized sum of token log-probabilities, including EOS. The common prompt cache is cropped back to its exact branch point after every candidate. Ranks are 1-based and aligned with the original candidate order.

MRR, R@1/5/10, and Mean Rank are the standard VisDial sparse retrieval metrics. NDCG is emitted only for validation rounds carrying dense relevance annotations. Scoring, rank conversion, and NDCG were checked against the official starter at commit `5844f3d5a575e9ec1c1684feb760e7de5c912beb`.

## Artifacts

- `raw.jsonl`: scores, ranks, provenance hashes, and selection summary for every method/dialog/round.
- `official_ranks/*.json`: VisDial evaluator-schema-compatible rank fragments for the configured deterministic subset.
- `summary.csv`: overall metrics by method.
- `per_turn.csv`: metrics grouped by dialogue round.
- `per_dialog.csv`: metrics grouped by dialog and method.
- `validation.json`: completeness, rank, score, and cross-method identity checks.

## Limitations

- Candidate likelihood is a model scoring policy, not a prescribed VisDial model architecture. Because it is not length-normalized, it can prefer shorter answers; this policy is fixed across all methods.
- The rank metrics and rank-file format are official-style retrieval outputs; no claim is made that this generative checkpoint reproduces a published VisDial baseline.
- Rank files cover exactly the configured index and turn limit. Do not treat a subset file as a complete full-validation EvalAI submission.
- This run intentionally contains no TTFT, decode, E2E, or SSD timing. Those belong to the separate system runner.
- Stored arms use caption-only pre-dialog calibration. Current/future answers and candidates are excluded from selector inputs.
