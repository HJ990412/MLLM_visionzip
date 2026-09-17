# MT-GQA Full Analysis

Validated **4,061 dialogues**, three turns and four methods per dialogue. Source GQA scores were independently validated against the original scorer.

## Primary quality metric and legacy-score audit

All quality tables, confidence intervals, McNemar tests, and verdicts use **strict normalized exact match**: the normalized prediction must equal the normalized first gold answer in full. Normalization lowercases, replaces punctuation with spaces, removes a/an/the, and collapses whitespace.

The original runner stored a **prefix-tolerant** `score` and `quality_score`: it also counted a longer prediction as correct if it began with the complete gold answer. The immutable image artifacts are unchanged. In this derived analysis, `raw.jsonl` uses strict `score`/`quality_score`, and preserves the source values as `stored_legacy_score`/`legacy_quality_score`; `recomputed_score` is strict and `legacy_recomputed_score` is the validated original.

Scorer disagreements: **9 of 48,732 requests**. Counts by method: ReComp=2, FullLoad=2, ImageOnly Prefix25=3, ImageOnly Prefix45=2.

Representative legacy-correct / strict-incorrect cases:

- mtgqa_003306 T2 ReComp: prediction `Computer mouse` versus gold `computer`; legacy 1, strict 0.
- mtgqa_003306 T2 FullLoad: prediction `Computer mouse` versus gold `computer`; legacy 1, strict 0.
- mtgqa_003306 T2 ImageOnly Prefix25: prediction `Computer mouse` versus gold `computer`; legacy 1, strict 0.
- mtgqa_003306 T2 ImageOnly Prefix45: prediction `Computer mouse` versus gold `computer`; legacy 1, strict 0.
- mtgqa_003314 T2 ImageOnly Prefix25: prediction `Computer mouse` versus gold `computer`; legacy 1, strict 0.

## Quality

| Method | Acc1 | Acc2 | Acc3 | Avg | Δ Avg vs FullLoad |
|---|---:|---:|---:|---:|---:|
| ReComp | 0.6319 | 0.7365 | 0.7518 | 0.7067 | -0.0002 |
| FullLoad | 0.6319 | 0.7373 | 0.7515 | 0.7069 | +0.0000 |
| ImageOnly Prefix25 | 0.6319 | 0.7065 | 0.7323 | 0.6902 | -0.0167 |
| ImageOnly Prefix45 | 0.6319 | 0.7294 | 0.7466 | 0.7026 | -0.0043 |

## Cache-hit system performance (Turns 2–3)

| Method | TTFT ms | E2E ms | SSD MB/request |
|---|---:|---:|---:|
| ReComp | 528.469 | 556.329 | 0.000 |
| FullLoad | 737.074 | 770.453 | 1173.919 |
| ImageOnly Prefix25 | 275.612 | 304.715 | 307.816 |
| ImageOnly Prefix45 | 428.164 | 457.491 | 555.570 |

## Future-query robustness

| Method | Gap T1 | Gap T2 | Gap T3 | Gap growth T3−T1 |
|---|---:|---:|---:|---:|
| ImageOnly Prefix25 | +0.0000 | -0.0308 | -0.0192 | -0.0192 |
| ImageOnly Prefix45 | +0.0000 | -0.0079 | -0.0049 | -0.0049 |

## FullLoad and Turn-1 sanity

| Turn | Prediction agreement | First-token agreement | FullLoad−ReComp Acc |
|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | +0.0000 |
| 2 | 0.9847 | 0.9860 | +0.0007 |
| 3 | 0.9902 | 0.9906 | -0.0002 |

Turn 1 four-arm agreement versus ReComp:

- ReComp: prediction 1.0000, first-token 1.0000, accuracy gap +0.0000
- FullLoad: prediction 1.0000, first-token 1.0000, accuracy gap +0.0000
- ImageOnly Prefix25: prediction 1.0000, first-token 1.0000, accuracy gap +0.0000
- ImageOnly Prefix45: prediction 1.0000, first-token 1.0000, accuracy gap +0.0000

## Direct answers

### Q1

"MT-GQA-reconstructed; source and hashes are recorded in the immutable run config."

### Q2

"Three questions from one image; Turn 2 receives Q1/gold A1/Q2 and Turn 3 receives Q1/gold A1/Q2/gold A2/Q3."

### Q3

{"recompute": {"acc1": 0.6318640728884511, "acc2": 0.7365180989903964, "acc3": 0.7517852745629156, "avg": 0.7067224821472544}, "fullload": {"acc1": 0.6318640728884511, "acc2": 0.7372568332922925, "acc3": 0.7515390297956168, "avg": 0.7068866453254535}, "prefix25": {"acc1": 0.6318640728884511, "acc2": 0.7064762373799557, "acc3": 0.7323319379463187, "avg": 0.6902240827382418}, "prefix45": {"acc1": 0.6318640728884511, "acc2": 0.7293770007387343, "acc3": 0.746614134449643, "avg": 0.702618402692276}}

### Q4

{"gap_growth_t3_minus_t1": -0.019207091849298204, "gap_worsened": true}

### Q5

{"gap_growth_t3_minus_t1": -0.004924895345973898, "gap_worsened": true}

### Q6

"Future-query robustness verdict: SUPPORTED."

### Q7

{"prefix25_ttft_reduction_vs_recomp_pct": 47.847041076127624}

### Q8

{"prefix45_ttft_reduction_vs_recomp_pct": 18.980295810103403}

### Q9

{"fullload_is_faster_than_recomp": false}

### Q10

{"partial_loading_needed_for_best_measured_ttft": true}

## Verdict

- MT-GQA Quality: **SUPPORTED**
- MT-GQA Efficiency: **SUPPORTED**

SUPPORTED iff both prefixes have Avg gap >= -2pp and T3-vs-T1 gap growth >= -2pp; PARTIAL iff one does.
SUPPORTED iff both prefixes beat ReComp cache-hit TTFT and FullLoad SSD bytes; PARTIAL iff one does.

## Protocol and storage conditions

The source paper specifies GQA testdev-balanced, 4,061 three-turn dialogues, but the exact dialogue artifact was unavailable at evaluation time. We therefore use a deterministic reconstruction and do not claim exact benchmark identity.

- History: gold teacher-forced; no generated answer is fed to a later turn.
- Turn 1: every arm performs normal pixel inference (vision forward 1, SSD read 0).
- Persistence is a one-time post-Turn-1 cost and is not added to cache-hit TTFT.
- Cache hits use OS-page-cache-cold buffered `pread`; O_DIRECT is false.
- SSD controller-cache flush is false/not performed.
- Run cache condition: `OS-page-cache-cold`.
- Main TTFT field: `end_to_end_ttft_ms`.

## Artifacts

- `dataset_provenance.json`
- `dialogues.json`
- `dataset_stats.json`
- `config.json`
- `raw.jsonl.gz` — lossless gzip of the 337 MB local `raw.jsonl`; run `gzip -cd raw.jsonl.gz > raw.jsonl` before tools that require the uncompressed path. Uncompressed SHA256: `fa1fef2408aaed2962e558aa580a4e204675b0b40cd8231ec1bb5fdac4cf34a5`.
- `per_turn.csv`
- `per_dialog.csv`
- `quality_by_turn.csv`
- `latency_by_turn.csv`
- `io_summary.csv`
- `persistence_overhead.csv`
- `statistical_analysis.json`
- `validation.json`
