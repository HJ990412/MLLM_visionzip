# MT-VQA-v2 Generated-History 4-Arm Analysis

### MT-VQA-v2 Generated-History Quality

| Method | Acc1 | Acc2 | Acc3 | Avg |
|---|---:|---:|---:|---:|
| ReComp | 79.73% | 78.80% | 83.20% | 80.58% |
| FullLoad | 79.73% | 78.93% | 82.80% | 80.49% |
| QA-Chunk25 | 79.73% | 76.93% | 80.93% | 79.20% |
| Ours25 | 79.73% | 74.93% | 80.13% | 78.27% |

### MT-VQA-v2 Cache-hit TTFT

| Method | T2 mean (p50/p95) | T3 mean (p50/p95) | T2–T3 pooled |
|---|---:|---:|---:|
| ReComp | 526.99 (519.19/699.82) | 532.67 (523.48/700.38) | 529.83 (522.23/700.54) |
| FullLoad | 698.34 (688.73/869.67) | 717.02 (703.92/905.17) | 707.68 (698.71/893.92) |
| QA-Chunk25 | 458.72 (459.31/555.99) | 466.50 (464.56/591.13) | 462.61 (462.89/575.44) |
| Ours25 | 280.73 (279.29/336.23) | 292.09 (288.82/358.19) | 286.41 (284.69/349.87) |

### Quality–Efficiency Summary

| Method | Avg Acc | T2–T3 TTFT | SSD MB/request | Preads/request |
|---|---:|---:|---:|---:|
| ReComp | 80.58% | 529.83 ms | 0.000 | 0.00 |
| FullLoad | 80.49% | 707.68 ms | 1176.494 | 64.00 |
| QA-Chunk25 | 79.20% | 462.61 ms | 351.140 | 308.72 |
| Ours25 | 78.27% | 286.41 ms | 309.011 | 65.00 |

## Generated-history input lengths

| Method | T2 input tokens | T3 input tokens | T2 generated tokens | T3 generated tokens |
|---|---:|---:|---:|---:|
| ReComp | 50.53 | 67.96 | 2.63 | 2.64 |
| FullLoad | 50.53 | 67.96 | 2.63 | 2.64 |
| QA-Chunk25 | 50.53 | 67.98 | 2.65 | 2.59 |
| Ours25 | 50.53 | 67.96 | 2.63 | 2.66 |

## TTFT reductions

- Ours vs ReComp: **45.94%**.
- Ours vs FullLoad: **59.53%**.
- Ours vs QA-Chunk25: **38.09%**.
- T2–T3 is the comparison population. Stored methods are cache hits; ReComp intentionally reprocesses pixels.

## Selector, SSD I/O, and locality

QA component timings may overlap. `selector_wall_ms` and request TTFT are the authoritative wall clocks; component means must not be summed.

| Method | SSD MB | ratio vs FullLoad | probe MB | selected K/V MB | separator MB | preads | read ms | runs/layer | mean run | max run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.000 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | N/A | 0 |
| FullLoad | 1176.494 | 1.0000 | 0.000 | 1176.494 | 0.000 | 64.00 | 427.73 | 1.00 | 35.49 | 46 |
| QA-Chunk25 | 351.140 | 0.2985 | 55.148 | 276.623 | 19.369 | 308.72 | 243.16 | 4.31 | 2.00 | 12 |
| Ours25 | 309.011 | 0.2627 | 0.000 | 289.642 | 19.369 | 65.00 | 151.56 | 1.00 | 8.63 | 12 |

QA pooled selector wall: 98.91 ms; rater count: 25.55; probe I/O: 32.68 ms.

## QA selection change

Across 250 dialogues, T2↔T3 layer-tagged chunk Jaccard is mean **0.851793**, median **0.852090**; identical rate **0.00%**; mean changed chunks **44.25**.
Ours fixed-prefix T2/T3 invariance: **PASS**.

## Paired soft quality

QA−Ours Acc1/Acc2/Acc3/Avg: +0.00/+2.00/+0.80/+0.93 pp.
The primary 10,000-resample image-cluster bootstrap Avg CI is [-0.84, +2.76] pp (seed 1234). Full-credit four-cells and McNemar values are diagnostic only.

## Error propagation

Conditioning uses full credit (`score == 1`), while every conditional outcome is mean soft VQA score. Populations are method-specific, so these are descriptive associations, not causal effects.

| Method | T2 given T1 full | T2 given T1 non-full | T3 given both prior full | T3 given any prior non-full |
|---|---:|---:|---:|---:|
| ReComp | 79.60% | 76.62% | 83.21% | 83.19% |
| FullLoad | 80.33% | 75.12% | 83.45% | 81.98% |
| QA-Chunk25 | 78.87% | 71.64% | 80.54% | 81.42% |
| Ours25 | 76.68% | 70.15% | 79.20% | 81.20% |

## Persistence and derived session latency

Persistence is excluded from cache-hit TTFT. QA's raster persistence in the secondary session table is a derived standalone-equivalent attribution of the shared FullLoad-captured raster store, not a separately measured QA write.

| Store | persist mean (p50/p95) | write MB/image |
|---|---:|---:|
| raster | 1089.89 (1051.86/1412.83) | 1272.041 |
| image_only | 1062.27 (1043.67/1320.23) | 1198.556 |

## MT-GQA Generated-History comparison

| Dataset | ReComp Avg | FullLoad Avg | QA Avg | Ours Avg | ReComp TTFT | FullLoad TTFT | QA TTFT | Ours TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MT-GQA reconstructed | 66.72% | 66.65% | 64.93% | 65.13% | 529.23 | 688.85 | 451.20 | 279.16 |
| MT-VQA-v2 reconstructed | 80.58% | 80.49% | 79.20% | 78.27% | 529.83 | 707.68 | 462.61 | 286.41 |

## Direct answers (Q1–Q9)

### Q1 — QA vs Ours quality gap

QA−Ours Acc1/Acc2/Acc3/Avg is +0.00/+2.00/+0.80/+0.93 pp.

### Q2 — Clear quality gain from per-turn adaptation?

No clear positive gain is established: the primary Avg bootstrap CI includes zero. This is not evidence of equivalence.

### Q3 — Ours TTFT reduction vs QA

**38.09%**.

### Q4 — Ours TTFT reduction vs ReComp

**45.94%**.

### Q5 — Is FullLoad faster than ReComp?

FullLoad is slower by 177.85 ms on pooled T2–T3 mean TTFT.

### Q6 — How much does QA selection change?

Mean/median Jaccard is 0.851793/0.852090, identical rate 0.00%, and mean symmetric-difference count 44.25.

### Q7 — Does selection change yield quality gain?

When QA selection changed, the descriptive T3 QA−Ours soft-score delta is 0.007999999999999998 (n=250); when identical it is N/A (n=0). This split is not randomized, so selection change alone does not identify a causal quality gain.

### Q8 — Is Ours unusually vulnerable to error propagation?

The descriptive T3 drop from both-prior-full to any-prior-non-full is -2.00% for Ours and -0.88% for QA. Because method-specific conditioning sets differ, this comparison alone cannot establish that Ours is specially vulnerable.

### Q9 — Same qualitative conclusion as MT-GQA?

Yes under the preregistered qualitative criterion (no clear positive QA quality gain plus a positive Ours-vs-QA TTFT reduction). MT-GQA showed QA−Ours Avg −0.20 pp and Ours-vs-QA TTFT reduction 38.13%; MT-VQA-v2 shows +0.93 pp with CI [-0.84, +2.76] pp and 38.09% TTFT reduction. No cross-dataset equivalence is claimed.

## Limitations

- No official/released MT-VQA-v2 dialogue artifact was available locally. This is a deterministic 250-image `MT-VQA-v2-reconstructed` subset and does not claim official benchmark or exact MetaCompress identity.
- The repository scorer preserves its established `min(matches/3, 1)` semantics but is not claimed byte-identical to the official VQA evaluation package.
- The frozen local VQAv2 index establishes this workload, but its original upstream stream-prefix length is not recoverable from the local config.
- OS page cache is conditioned; SSD controller cache is not flushed.
- ReComp and persisted Visual-KV paths are operationally different; no byte/output-equivalence claim is made.
- Full-credit conditioning and McNemar are diagnostic; primary quality and inference remain soft-score means and image-cluster bootstrap.

## Completion

```text
DATASET CONSTRUCTION: PASS
IMPLEMENTATION: PASS
FULL RUN: PASS
VALIDATION: PASS
```

- Images: 250
- Dialogues: 250
- Total requests: 3000
- Failed / duplicates: 0 / 0
- Index SHA256: `89719b2a1187c07e3228cc76cf1e473b3c713a0dcb3d65da0c596ae81898d6ea`

MT-VQA-v2 GENERATED-HISTORY 4-ARM EVALUATION VALIDATED: YES
