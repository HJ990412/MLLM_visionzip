# MT-GQA Full Analysis

Validated **10 dialogues**, three turns and four methods per dialogue. GQA scores were independently recomputed.

## Quality

| Method | Acc1 | Acc2 | Acc3 | Avg | Δ Avg vs FullLoad |
|---|---:|---:|---:|---:|---:|
| ReComp | 0.6000 | 1.0000 | 1.0000 | 0.8667 | +0.0000 |
| FullLoad | 0.6000 | 1.0000 | 1.0000 | 0.8667 | +0.0000 |
| ImageOnly Prefix25 | 0.6000 | 0.9000 | 1.0000 | 0.8333 | -0.0333 |
| ImageOnly Prefix45 | 0.6000 | 1.0000 | 1.0000 | 0.8667 | +0.0000 |

## Cache-hit system performance (Turns 2–3)

| Method | TTFT ms | E2E ms | SSD MB/request |
|---|---:|---:|---:|
| ReComp | 615.315 | 651.130 | 0.000 |
| FullLoad | 892.008 | 929.244 | 1333.789 |
| ImageOnly Prefix25 | 327.742 | 365.502 | 360.710 |
| ImageOnly Prefix45 | 520.238 | 558.295 | 629.146 |

## Future-query robustness

| Method | Gap T1 | Gap T2 | Gap T3 | Gap growth T3−T1 |
|---|---:|---:|---:|---:|
| ImageOnly Prefix25 | +0.0000 | -0.1000 | +0.0000 | +0.0000 |
| ImageOnly Prefix45 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## FullLoad and Turn-1 sanity

| Turn | Prediction agreement | First-token agreement | FullLoad−ReComp Acc |
|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | +0.0000 |
| 2 | 1.0000 | 1.0000 | +0.0000 |
| 3 | 1.0000 | 1.0000 | +0.0000 |

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

{"recompute": {"acc1": 0.6, "acc2": 1.0, "acc3": 1.0, "avg": 0.8666666666666666}, "fullload": {"acc1": 0.6, "acc2": 1.0, "acc3": 1.0, "avg": 0.8666666666666666}, "prefix25": {"acc1": 0.6, "acc2": 0.9, "acc3": 1.0, "avg": 0.8333333333333334}, "prefix45": {"acc1": 0.6, "acc2": 1.0, "acc3": 1.0, "avg": 0.8666666666666666}}

### Q4

{"gap_growth_t3_minus_t1": 0.0, "gap_worsened": false}

### Q5

{"gap_growth_t3_minus_t1": 0.0, "gap_worsened": false}

### Q6

"Future-query robustness verdict: PARTIALLY SUPPORTED."

### Q7

{"prefix25_ttft_reduction_vs_recomp_pct": 46.73591165707407}

### Q8

{"prefix45_ttft_reduction_vs_recomp_pct": 15.45177433614185}

### Q9

{"fullload_is_faster_than_recomp": false}

### Q10

{"partial_loading_needed_for_best_measured_ttft": true}

## Verdict

- MT-GQA Quality: **PARTIALLY SUPPORTED**
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
- `raw.jsonl`
- `per_turn.csv`
- `per_dialog.csv`
- `quality_by_turn.csv`
- `latency_by_turn.csv`
- `io_summary.csv`
- `persistence_overhead.csv`
- `statistical_analysis.json`
- `validation.json`
