# MT-GQA Gold + Generated History — Four-Arm Evaluation

## Gold-History

| Method | Acc1 | Acc2 | Acc3 | Avg |
|---|---:|---:|---:|---:|
| ReComp | 63.19% | 73.65% | 75.18% | 70.67% |
| FullLoad | 63.19% | 73.77% | 75.20% | 70.72% |
| QA-Chunk25 | 63.19% | 71.14% | 72.47% | 68.93% |
| Ours25 | 63.19% | 70.65% | 73.23% | 69.02% |

## Generated-History

| Method | Acc1 | Acc2 | Acc3 | Avg |
|---|---:|---:|---:|---:|
| ReComp | 63.19% | 68.11% | 68.85% | 66.72% |
| FullLoad | 63.19% | 68.06% | 68.70% | 66.65% |
| QA-Chunk25 | 63.19% | 65.77% | 65.85% | 64.93% |
| Ours25 | 63.19% | 65.48% | 66.73% | 65.13% |

## Gold vs Generated

| Method | Gold Avg | Generated Avg | Δ Avg | Gold Acc3 | Generated Acc3 | Δ Acc3 |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 70.67% | 66.72% | -3.96 pp | 75.18% | 68.85% | -6.33 pp |
| FullLoad | 70.72% | 66.65% | -4.07 pp | 75.20% | 68.70% | -6.50 pp |
| QA-Chunk25 | 68.93% | 64.93% | -4.00 pp | 72.47% | 65.85% | -6.62 pp |
| Ours25 | 69.02% | 65.13% | -3.89 pp | 73.23% | 66.73% | -6.50 pp |

## Input/output token lengths

### Gold-History

| Method | T1 generated | T2 generated | T2 input | T3 input |
|---|---:|---:|---:|---:|
| ReComp | 2.40 | 2.44 | 55.68 | 75.70 |
| FullLoad | 2.40 | 2.45 | 55.68 | 75.70 |
| QA-Chunk25 | 2.40 | 2.43 | 55.68 | 75.70 |
| Ours25 | 2.40 | 2.45 | 55.68 | 75.70 |

### Generated-History

| Method | T1 generated | T2 generated | T2 input | T3 input |
|---|---:|---:|---:|---:|
| ReComp | 2.40 | 2.43 | 55.74 | 75.84 |
| FullLoad | 2.40 | 2.43 | 55.74 | 75.84 |
| QA-Chunk25 | 2.40 | 2.42 | 55.74 | 75.83 |
| Ours25 | 2.40 | 2.42 | 55.74 | 75.84 |

Generated-History T3 input lengths are method-specific because each method propagates its own decoded responses: ReComp 75.84, FullLoad 75.84, QA-Chunk25 75.83, Ours25 75.84 tokens on average. TTFT differences therefore combine Visual-KV path effects with history-length effects; the report does not conflate the two.

## Gold-History system performance

Main population is cache-hit Turns 2–3. TTFT includes prompt/token preparation, H2D, online selection, SSD I/O, scatter, prefill, and the synchronized first-token decision.

| Method | T2 TTFT mean (p50/p95) | T3 TTFT mean (p50/p95) | T2–3 mean (p50/p95) | SSD MB | Selector ms | Preads |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 525.05 (528.25/590.45) | 532.48 (536.16/597.63) | 528.76 (530.53/596.67) | 0.00 | 0.00 | 0.00 |
| FullLoad | 691.61 (687.21/849.94) | 692.38 (688.86/848.94) | 692.00 (688.17/849.73) | 1173.92 | 0.00 | 64.00 |
| QA-Chunk25 | 450.46 (453.18/534.12) | 449.62 (451.20/533.34) | 450.04 (452.33/533.59) | 349.91 | 96.51 | 303.41 |
| Ours25 | 277.26 (277.94/328.49) | 279.00 (279.17/328.81) | 278.13 (278.54/328.75) | 307.82 | 0.21 | 65.00 |

## Generated-History system performance

Main population is cache-hit Turns 2–3. TTFT includes prompt/token preparation, H2D, online selection, SSD I/O, scatter, prefill, and the synchronized first-token decision.

| Method | T2 TTFT mean (p50/p95) | T3 TTFT mean (p50/p95) | T2–3 mean (p50/p95) | SSD MB | Selector ms | Preads |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 525.38 (528.59/594.48) | 533.09 (536.35/596.76) | 529.23 (530.98/596.22) | 0.00 | 0.00 | 0.00 |
| FullLoad | 687.67 (685.73/827.66) | 690.02 (689.40/829.70) | 688.85 (687.48/828.88) | 1173.92 | 0.00 | 64.00 |
| QA-Chunk25 | 451.20 (453.50/537.78) | 451.19 (454.43/534.41) | 451.20 (453.96/536.63) | 349.92 | 96.68 | 303.06 |
| Ours25 | 278.59 (278.60/328.01) | 279.74 (279.82/328.71) | 279.16 (279.37/328.11) | 307.82 | 0.21 | 65.00 |

## Detailed cache-hit I/O and selector costs

### Gold-History pooled T2–T3 I/O

| Method | SSD MB | Probe MB | Selected KV MB | Separator MB | Preads | Runs/layer | SSD read ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FullLoad | 1173.92 | 0.00 | 1173.92 | 0.00 | 64.00 | 1.00 | 419.98 |
| QA-Chunk25 | 349.91 | 55.03 | 275.46 | 19.43 | 303.41 | 4.23 | 238.78 |
| Ours25 | 307.82 | 0.00 | 288.39 | 19.43 | 65.00 | 1.00 | 144.09 |

### Gold-History QA-Chunk25 selector by turn

| Turn | Raters | Rater ms | Projection ms | Probe I/O ms | Query score ms | Chunk aggregation ms | Top-k ms | Selector ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 23.84 | 1.75 | 12.85 | 33.20 | 8.13 | 9.01 | 5.68 | 96.99 |
| 3 | 32.99 | 1.76 | 12.37 | 33.24 | 8.17 | 9.09 | 5.72 | 96.03 |

### Generated-History pooled T2–T3 I/O

| Method | SSD MB | Probe MB | Selected KV MB | Separator MB | Preads | Runs/layer | SSD read ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FullLoad | 1173.92 | 0.00 | 1173.92 | 0.00 | 64.00 | 1.00 | 419.33 |
| QA-Chunk25 | 349.92 | 55.03 | 275.46 | 19.43 | 303.06 | 4.22 | 239.86 |
| Ours25 | 307.82 | 0.00 | 288.39 | 19.43 | 65.00 | 1.00 | 144.92 |

### Generated-History QA-Chunk25 selector by turn

| Turn | Raters | Rater ms | Projection ms | Probe I/O ms | Query score ms | Chunk aggregation ms | Top-k ms | Selector ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 23.79 | 1.75 | 12.65 | 33.32 | 8.13 | 8.98 | 5.66 | 96.94 |
| 3 | 32.90 | 1.76 | 12.36 | 33.44 | 8.20 | 9.11 | 5.73 | 96.41 |

## Generated-minus-Gold degradation

| Method | ΔAcc2 | ΔAcc3 | ΔAvg |
|---|---:|---:|---:|
| ReComp | -5.54 pp | -6.33 pp | -3.96 pp |
| FullLoad | -5.71 pp | -6.50 pp | -4.07 pp |
| QA-Chunk25 | -5.37 pp | -6.62 pp | -4.00 pp |
| Ours25 | -5.17 pp | -6.50 pp | -3.89 pp |

## Paired QA-Chunk25 vs Ours25

### Gold-History

| Turn | Both correct | QA only | Ours only | Both wrong | QA−Ours | Exact McNemar p | Dialogue-bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2566 | 0 | 0 | 1495 | +0.00 pp | 1 | [+0.00, +0.00] pp |
| 2 | 2620 | 269 | 249 | 923 | +0.49 pp | 0.403847 | [-0.59, +1.60] pp |
| 3 | 2742 | 201 | 232 | 886 | -0.76 pp | 0.1493 | [-1.77, +0.27] pp |
| Avg | — | — | — | — | -0.09 pp | — | [-0.57, +0.42] pp |

### Generated-History

| Turn | Both correct | QA only | Ours only | Both wrong | QA−Ours | Exact McNemar p | Dialogue-bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2566 | 0 | 0 | 1495 | +0.00 pp | 1 | [+0.00, +0.00] pp |
| 2 | 2415 | 256 | 244 | 1146 | +0.30 pp | 0.622809 | [-0.79, +1.38] pp |
| 3 | 2441 | 233 | 269 | 1118 | -0.89 pp | 0.11817 | [-1.97, +0.20] pp |
| Avg | — | — | — | — | -0.20 pp | — | [-0.73, +0.35] pp |

## Generated-history error propagation

| Method | T1C/T2C | T1C/T2W | T1W/T2C | T1W/T2W |
|---|---:|---:|---:|---:|
| ReComp | 1885 | 681 | 881 | 614 |
| FullLoad | 1883 | 683 | 881 | 614 |
| QA-Chunk25 | 1818 | 748 | 853 | 642 |
| Ours25 | 1815 | 751 | 844 | 651 |

T3 accuracy for each exact prior-turn correctness pattern:

| Method | CC | CW | WC | WW |
|---|---:|---:|---:|---:|
| ReComp | 76.45% (1885) | 60.35% (681) | 70.37% (881) | 52.77% (614) |
| FullLoad | 76.53% (1883) | 59.88% (683) | 70.94% (881) | 51.30% (614) |
| QA-Chunk25 | 73.65% (1818) | 57.62% (748) | 67.64% (853) | 50.93% (642) |
| Ours25 | 74.16% (1815) | 59.65% (751) | 69.19% (844) | 51.00% (651) |

Conditional summaries:

| Method | T2 given T1 correct (n) | T2 given T1 wrong (n) | T3 given T1,T2 correct (n) | T3 given prior error (n) |
|---|---:|---:|---:|---:|
| ReComp | 73.46% (2566) | 58.93% (1495) | 76.45% (1885) | 62.27% (2176) |
| FullLoad | 73.38% (2566) | 58.93% (1495) | 76.53% (1883) | 61.94% (2178) |
| QA-Chunk25 | 70.85% (2566) | 57.06% (1495) | 73.65% (1818) | 59.52% (2243) |
| Ours25 | 70.73% (2566) | 56.45% (1495) | 74.16% (1815) | 60.73% (2246) |

## Direct answers to Q1–Q8

### Q1 — Gold-history QA-Chunk25 vs Ours25

The QA−Ours gaps for Acc1/Acc2/Acc3/Avg are +0.00 pp/+0.49 pp/-0.76 pp/-0.09 pp.

### Q2 — Generated-history QA-Chunk25 vs Ours25

The QA−Ours gaps for Acc1/Acc2/Acc3/Avg are +0.00 pp/+0.30 pp/-0.89 pp/-0.20 pp.

### Q3 — Generated minus Gold

- ReComp: ΔAcc2 -5.54 pp, ΔAcc3 -6.33 pp, ΔAvg -3.96 pp.
- FullLoad: ΔAcc2 -5.71 pp, ΔAcc3 -6.50 pp, ΔAvg -4.07 pp.
- QA-Chunk25: ΔAcc2 -5.37 pp, ΔAcc3 -6.62 pp, ΔAvg -4.00 pp.
- Ours25: ΔAcc2 -5.17 pp, ΔAcc3 -6.50 pp, ΔAvg -3.89 pp.

### Q4 — Method-specific error propagation

Ours changes 0.11 pp more favorably than QA in Avg. The full ΔAcc2/ΔAcc3/ΔAvg and conditional T2/T3 tables above show the method-specific pattern. This is a descriptive association under each method's own generated history, not an equivalence test or a causal estimate.

### Q5 — Does the adaptive-selection quality gain persist?

QA−Ours Avg is -0.09 pp with Gold history and -0.20 pp with Generated history. Per-turn paired cells, exact McNemar p-values, and dialogue-cluster bootstrap CIs are in `paired_quality.json`.

### Q6 — Cost of that gain

In Gold T2–3, QA−Ours mean TTFT is +171.91 ms, selector cost is +96.30 ms, SSD traffic is +42.10 MB/request, and preads differ by +238.41/request.

In Generated T2–3, QA−Ours mean TTFT is +172.03 ms, selector cost is +96.46 ms, SSD traffic is +42.10 MB/request, and preads differ by +238.06/request.

### Q7 — QA T2↔T3 selection

Mean layer-tagged chunk Jaccard is 0.847407 for Gold and 0.846865 for Generated history (difference -0.000542). Cross-protocol same-turn comparisons are recorded in `selection_analysis.json`.

### Q8 — Ours fixed-prefix invariance

Ours selected physical prefix IDs are identical across Gold/Generated, T2/T3, and all queries for the same image: **YES**. The underlying image-only storage permutation is also identical across protocols: **YES**.

## Paired quality and error propagation

Exact paired cells and McNemar tests use 4,061 dialogues per turn. The bootstrap uses 10,000 dialogue-cluster resamples with seed 1234. Conditional error-propagation counts and accuracies are in `error_propagation.csv`.

## Protocol interpretation

Gold-History is a controlled multi-turn evaluation that separates Visual-KV retrieval effects from prior-generation error propagation. Generated-History propagates each method's own previous outputs and captures realistic conversational error propagation. They answer different questions; neither is labeled the uniquely correct metric.

## Limitations

- This is MT-GQA-reconstructed with MetaCompress-compatible Acc1/Acc2/Acc3/Avg reporting; exact MetaCompress protocol identity is not claimed.
- Dialogue-cluster bootstrap follows the requested unit and does not additionally cluster dialogues that share an image.
- OS page cache is conditioned, but SSD controller cache is not flushed.
- FullLoad uses persisted FP16 Visual-KV and the stored-KV manual decode path, while ReComp uses the ordinary pixel/HF generation path; no byte-identical or output-equivalence claim is made between them.
- No equivalence claim is made; the report provides differences, CIs, and McNemar tests.

## Completion

```text
IMPLEMENTATION: PASS
GOLD-HISTORY RUN: PASS
GENERATED-HISTORY RUN: PASS
VALIDATION: PASS
```

- Workload index: `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`
- Logical rows: 97,464
- Failed requests: 0
- Duplicate cells: 0

MT-GQA GOLD + GENERATED HISTORY 4-ARM EVALUATION VALIDATED: YES
