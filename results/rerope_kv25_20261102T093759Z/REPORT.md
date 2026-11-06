# Ours-ReRoPE and Pruned-T1 (LLaVA, GQA Stage 1)

Run: `runs/rerope_kv25_20261102T093759Z`. Contract: [CONTRACT.md](CONTRACT.md) (frozen before GPU execution; G-a/G-b tolerances fixed from the rev.-3 G4 stores, evidence [dev/g_ab_evidence.json](dev/g_ab_evidence.json)). Rev.-3 arms Orig and Compact were re-run in this run; the rev.-3 run is only used for the reproduction gate G-e.

**All conclusions are exploratory**: the 40-image GQA development set has been used repeatedly (rev. 3 and earlier runs) and the ReRoPE arm was motivated by the rev.-3 result on the same images. Differences below 1 pp are not interpreted as established (BF16 shape rounding can flip boundary first tokens; rev.-3 noise floor).

Workload: 40 GQA images × `questions[4:10]` (manifest `40e23fa7…`), T1 = `questions[4]`, 200 cache hits (T2–T6) per stored arm; Pruned-T1 answers all 240 questions as standalone pruned first turns. LLaVA-NeXT-Vicuna-7B NF4/BF16/eager, FP16 KV, 64-token chunks, separator sidecar, greedy max 16, cold page cache (DONTNEED) outside timers. ReRoPE shares the Ours T1 capture (its T1 row is the Ours T1).

## Gates

| Gate | Result |
|---|---:|
| G1 selection/bytes/embeds/T1 identity (rev. 3) | PASS |
| G2 rev. 3 structural bit-identity + layer-0 S/sep (Ours/Orig/Compact) | PASS |
| G-a round trip ReRoPE K -> Ours K (all layers) | PASS (40/40; max rel L2 2.16e-04) |
| G-b layer-0 ReRoPE K vs Compact K; V/sys bit-identity; negative control | PASS (max rel L2 0.0046 ≤ 0.01; oracle ratio ≤ 1.030; un-rotated ≥ 1.12; V/sys bit-identical True) |
| G-c text start / positions (all requests) | PASS |
| G-d identity, equal hit reads (4 KV25 arms), Pruned S identity, zero reads | PASS |
| G-e reproduction vs rev. 3 (report-only) | Ours-KV25 100.00%; Recomp-KV25-Orig 100.00%; Recomp-KV25-Compact 100.00%; FullLoad 100.00%; ReComp 100.00% |
| G-f Pruned-T1 vs Compact hits (report-only) | first token 99.00%, prediction 99.00%, 2 mismatches |
| G4 GPU correctness (rev. 3 (a)–(e) + extension (f)–(h)) | PASS / PASS |
| G5 protection before/after | PASS |
| G6 Ours vs 2026-10-01 reference | 100.00% sequence agreement |
| G7 sentinel non-leakage (smoke) | PASS |

## 2×2: KV context × position (cache-hit accuracy, T2–T6)

| Context \ Position | Original positions | Contiguous positions |
|---|---:|---:|
| Full T1 context | 56.50% (Ours-KV25) | 57.00% (Ours-ReRoPE) |
| Reduced prefill context | 56.00% (Recomp-KV25-Orig) | 60.50% (Recomp-KV25-Compact) |

Baselines: FullLoad 63.00%, ReComp 63.00%. Prediction agreement with FullLoad: Ours-KV25 82.00%, Recomp-KV25-Orig 77.50%, Recomp-KV25-Compact 84.50%, Ours-ReRoPE 85.00%.

### Paired differences (image-cluster bootstrap, 10,000 resamples, seed 1234)

| Difference | Mean | 95% CI | a-only / b-only correct | Pre-registered reading |
|---|---:|---:|---:|---:|
| Ours-ReRoPE − Ours-KV25 | +0.50 pp | [-4.50 pp, +5.50 pp] | 12 / 11 | difference not shown (CI includes 0; not equivalence) |
| Ours-ReRoPE − Recomp-KV25-Compact | -3.50 pp | [-8.00 pp, +1.00 pp] | 5 / 12 | difference not shown (CI includes 0; not equivalence) |
| Ours-KV25 − Recomp-KV25-Orig | +0.50 pp | [-2.50 pp, +3.50 pp] | 7 / 6 | difference not shown (CI includes 0; not equivalence) |
| Ours-KV25 − Recomp-KV25-Compact | -4.00 pp | [-8.00 pp, +0.00 pp] | 6 / 14 | difference not shown (CI includes 0; not equivalence) |
| Recomp-KV25-Orig − Recomp-KV25-Compact | -4.50 pp | [-8.00 pp, -0.50 pp] | 7 / 16 | Recomp-KV25-Compact higher (CI upper bound < 0) |

### Factor effects (joint image-level bootstrap, same resamples for all three)

| Effect | Contrast | Positive means | Estimate | 95% CI | Reading |
|---|---:|---:|---:|---:|---:|
| context | ((Ours + ReRoPE) − (Orig + Compact)) / 2 | full context higher than reduced (averaged over positions) | -1.50 pp | [-4.25 pp, +1.50 pp] | not shown (CI includes 0) |
| position | ((Compact + ReRoPE) − (Ours + Orig)) / 2 | contiguous positions higher than original (averaged over contexts) | +2.50 pp | [-1.00 pp, +6.00 pp] | not shown (CI includes 0) |
| interaction | (Ours − ReRoPE) − (Orig − Compact) | original-minus-contiguous gap larger in full context than in reduced context | +4.00 pp | [-1.00 pp, +9.00 pp] | not shown (CI includes 0) |

## Decisions (rules fixed in the contract before results)

- **ReRoPE − Ours**: difference not shown (CI includes 0; not equivalence) (mean +0.50 pp, CI [-4.50 pp, +5.50 pp]).
- **ReRoPE − Compact**: difference not shown (CI includes 0; not equivalence) (mean -3.50 pp, CI [-8.00 pp, +1.00 pp]).
- **Context effect** not shown (CI includes 0); **position effect** not shown (CI includes 0); **interaction** not shown (CI includes 0).
- ReRoPE needs no extra prefill: its cost is 7.31 ms re-rotation inside persistence (vs 87.66 ms reduced prefill for Recomp).

## Pruned-T1 (VisionZip-style first turn, no cache) vs full-token standalone inference (ReComp)

| Questions | n | Pruned-T1 | ReComp | Pruned − ReComp | 95% CI | Reading |
|---|---:|---:|---:|---:|---:|---:|
| (a) T1 | 40 | 55.00% | 60.00% | -5.00 pp | [-15.00 pp, +5.00 pp] | not shown (CI includes 0) |
| (b) T2–T6 | 200 | 61.50% | 63.00% | -1.50 pp | [-6.50 pp, +3.00 pp] | not shown (CI includes 0) |
| (c) all | 240 | 60.42% | 62.50% | -2.08 pp | [-6.25 pp, +2.08 pp] | not shown (CI includes 0) |

G-f: Pruned-T1 T2–T6 vs the Compact hit of the same question: first-token agreement 99.00%, prediction agreement 99.00% (2 of 200 differ; the pruned prefill contains the question tokens in the same GEMM and the Compact hit goes through the FP16 store and the masked full-size buffer).

### Dialogue-level expected accuracy (1 T1 + 5 hits per image)

| Design | Mean accuracy | 95% CI |
|---|---:|---:|
| T1 full-token + Ours-KV25 hits | 57.08% | [50.00%, 64.17%] |
| T1 full-token + Ours-ReRoPE hits | 57.50% | [50.42%, 64.58%] |
| All Pruned-T1 | 60.42% | [53.74%, 67.08%] |

| Difference | Mean | 95% CI | Reading |
|---|---:|---:|---:|
| T1 full + Ours − T1 full + ReRoPE | -0.42 pp | [-4.58 pp, +3.75 pp] | not shown (CI includes 0) |
| T1 full + Ours − All Pruned | -3.33 pp | [-7.08 pp, +0.42 pp] | not shown (CI includes 0) |
| T1 full + ReRoPE − All Pruned | -2.92 pp | [-7.08 pp, +0.83 pp] | not shown (CI includes 0) |

"T1 full-token" is the ReComp T1 row; every full-token T1 (ReComp, FullLoad, Ours, Orig, Compact) produced identical first tokens and sequences: True.

## Cache-hit latency and SSD reads

| Arm | TTFT mean / p50 / p95 (ms) | E2E mean (ms) | Read MB/hit | preads/hit |
|---|---:|---:|---:|---:|
| ReComp | 519.13 / 527.17 / 589.39 | 544.90 | 0.000 | 0.0 |
| FullLoad | 732.74 / 727.75 / 855.52 | 759.74 | 1165.073 | 64.0 |
| Ours-KV25 | 281.78 / 279.84 / 346.27 | 308.67 | 306.289 | 65.0 |
| Recomp-KV25-Orig | 278.58 / 276.72 / 328.44 | 305.87 | 306.289 | 65.0 |
| Recomp-KV25-Compact | 271.96 / 265.49 / 345.13 | 298.92 | 306.289 | 65.0 |
| Ours-ReRoPE | 278.10 / 275.96 / 330.86 | 304.67 | 306.289 | 65.0 |

The four KV25 arms read identical bytes and preads on every hit: yes. Recomp-KV25-Orig − Ours TTFT -3.20 ms [-7.06, +0.59]; Recomp-KV25-Compact − Ours TTFT -9.83 ms [-15.67, -4.06]; Ours-ReRoPE − Ours TTFT -3.68 ms [-6.65, -0.95].

Pruned-T1 request time (mean over 240 requests): TTFT 166.9 ms = image decode 5.2 + processor/H2D 14.9 + vision/projector/merge 43.2 + saliency/order/plan 1.0 + gather 0.0 + pruned prefill 102.4. **This is a composed measurement (the full-token forward is aborted after the multimodal merge, not the schema-v2 server path) and is not used for any efficiency comparison.**

## Persistence cost per image (mean ms; not part of cache-hit TTFT)

| Arm | Saliency | Permutation | Plan | Embed gather | Reduced prefill | Synthetic assembly | **Re-rotation** | Repack | Write | fsync | Seal | Total | File MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 421.18 | 408.48 | 649.04 | 1678.79 | 1260.58 |
| Ours-KV25 | 0.02 | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 136.73 | 106.91 | 197.26 | 184.48 | 650.97 | 309.02 |
| Recomp-KV25-Orig | 0.02 | 0.60 | 0.92 | 0.09 | 87.66 | 2.26 | 0.00 | 136.27 | 106.86 | 185.29 | 181.96 | 726.69 | 309.02 |
| Recomp-KV25-Compact | 0.02 | 0.60 | 0.92 | 0.09 | 87.77 | 2.26 | 0.00 | 139.42 | 107.42 | 180.98 | 183.44 | 727.78 | 309.02 |
| Ours-ReRoPE | 0.02 | 0.59 | 0.96 | 0.00 | 0.00 | 0.00 | 7.31 | 135.08 | 105.94 | 193.88 | 184.54 | 652.73 | 309.02 |

| Arm | Primary extra cost | mean / p95 (ms) | ÷1 | ÷3 | ÷5 | ÷10 | Full Δ vs Ours [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recomp-KV25-Orig | prefill_only_ms | 87.75 / 114.69 | 87.75 | 29.25 | 17.55 | 8.78 | +75.7 [+64.0, +87.7] |
| Recomp-KV25-Compact | prefill_only_ms | 87.86 / 114.77 | 87.86 | 29.29 | 17.57 | 8.79 | +76.8 [+63.8, +90.5] |
| Ours-ReRoPE | rerope_ms | 7.31 / 8.10 | 7.31 | 2.44 | 1.46 | 0.73 | +1.8 [-6.2, +9.0] |

ReRoPE shares the Ours T1 capture; its persistence re-runs saliency/permutation on that capture, so the full Δ also contains writer wall-time noise. The re-rotation (`rerope_ms`, FP64, 32 layers) is the only compute the ReRoPE store needs beyond Ours.

## Per-layer KV difference (mean over 40 images, S rows, rel L2)

| Layer | Orig vs Ours K / V (context, original pos.) | ReRoPE vs Compact K / V (context, contiguous pos.) | ReRoPE vs Ours K (realignment only) | Compact realigned vs Ours K |
|---|---:|---:|---:|---:|
| 0 | 0.0023 / 0.0021 | 0.0044 / 0.0021 | 1.1738 | 0.0043 |
| 1 | 0.0308 / 0.0244 | 0.0268 / 0.0194 | 1.0353 | 0.0262 |
| 2 | 0.0275 / 0.0340 | 0.0267 / 0.0326 | 0.9437 | 0.0262 |
| 4 | 0.0678 / 0.0972 | 0.0463 / 0.1069 | 0.9765 | 0.0461 |
| 8 | 0.1232 / 0.2171 | 0.2111 / 0.3959 | 0.9755 | 0.2145 |
| 12 | 0.2103 / 0.3801 | 0.3208 / 0.5381 | 0.9392 | 0.3233 |
| 16 | 0.2280 / 0.3648 | 0.3564 / 0.4862 | 0.9305 | 0.3535 |
| 20 | 0.2162 / 0.3380 | 0.3202 / 0.4347 | 0.8849 | 0.3191 |
| 24 | 0.2112 / 0.3573 | 0.2967 / 0.4545 | 0.8731 | 0.2959 |
| 28 | 0.2173 / 0.4015 | 0.2724 / 0.5063 | 0.9073 | 0.2739 |
| 31 | 0.2158 / 0.4184 | 0.2607 / 0.5305 | 1.0225 | 0.2590 |

Full table: [kv_layer_diff.csv](kv_layer_diff.csv).

## Reproduction (G-e vs rev. 3; G6 vs 2026-10-01)

| New arm | Reference | n | First-token agr. | Prediction agr. | Sequence agr. |
|---|---:|---:|---:|---:|---:|
| Ours-KV25 | rev3 / ours | 240 | 100.00% | 100.00% | 100.00% |
| Recomp-KV25-Orig | rev3 / orig | 240 | 100.00% | 100.00% | 100.00% |
| Recomp-KV25-Compact | rev3 / compact | 240 | 100.00% | 100.00% | 100.00% |
| FullLoad | rev3 / fullload | 240 | 100.00% | 100.00% | 100.00% |
| ReComp | rev3 / recompute | 240 | 100.00% | 100.00% | 100.00% |
| Ours-KV25 | prefix25_20261001 / B | 240 | 100.00% | 100.00% | 100.00% |

G-f Pruned-T1 vs Compact mismatches (2), with first-token top-2 logit margins:
- n272098/201535605 T4: pruned `No` (1939, margin 0.000) vs compact `Yes` (3869, margin 0.125), gold `no`
- n450919/2093757 T2: pruned `Small` (18285, margin 0.125) vs compact `Large` (8218, margin 0.000), gold `small`

## Limits

- Exploratory: 40 image clusters of a development set used several times; the ReRoPE hypothesis came from the rev.-3 result on the same images; 200 hits per arm; Stage 2 not run.
- LLaVA only. Recomp arms and Pruned-T1 use image-only saliency without contextual merge, so they isolate KV context and position rule, not the full VisionZip system.
- Pruned-T1 TTFT is a composed measurement and is not an efficiency result.
- Latency is single-GPU, batch 1, page-cache DONTNEED; no NAND/controller coldness claim.
- G-a/G-b tolerances were fixed from rev.-3 stores before this run; G-e and G-f are report-only.

