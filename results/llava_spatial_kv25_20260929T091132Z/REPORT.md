# LLaVA-NeXT GQA SpatialUniform versus IndexUniform Visual-KV25

This is the previously used 40-image development workload. The Spatial method samples original decoder KV rows on the model input patch grids; it does not merge features or KV values. The 54:10 arm changes only the dominant/contextual integer allocation within the existing original-token contextual selector, and is not a reproduction of VisionZip feature merging.

| Method | D/aux mean tokens | All accuracy | Hit accuracy | Δ hit vs D25 (pp) | Δ hit vs Index (pp) | Hit TTFT (ms) | SSD MB/hit | Persistence ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | N/A | 0.6250 | 0.6300 | +6.50 | +5.50 | 523.23 | 0.000 | N/A |
| FullLoad | N/A | 0.6250 | 0.6300 | +6.50 | +5.50 | 723.36 | 1165.073 | 1106.85 |
| D25+C0 | 546.0/0.0 | 0.5708 | 0.5650 | +0.00 | -1.00 | 264.17 | 319.501 | 1125.91 |
| D20+Context5 | 437.15/108.85 | 0.5792 | 0.5750 | +1.00 | +0.00 | 265.98 | 319.501 | 1091.24 |
| D20+IndexUniform5 | 437.15/108.85 | 0.5792 | 0.5750 | +1.00 | +0.00 | 266.83 | 319.501 | 1072.96 |
| D20+SpatialUniform5 | 437.15/108.85 | 0.5458 | 0.5350 | -3.00 | -4.00 | 266.23 | 319.501 | 1087.09 |
| D20+Random5 | 437.15/108.85 | 0.5625 | 0.5550 | -1.00 | -2.00 | 266.54 | 319.501 | 1100.94 |
| D21.1+C3.9 [54:10 reference] | 460.95/85.05 | 0.5708 | 0.5650 | +0.00 | -1.00 | 265.44 | 319.501 | 1115.50 |

54:10 integer allocation is k_aux=floor(10k/64), k_dominant=k-k_aux. Across 40 images, its actual mean dominant/content fraction was 0.2111 and auxiliary/content fraction 0.0389; the displayed 21.1/3.9 percentages are nominal. Average counts were 460.95/85.05.

## Fixed comparisons and paired disagreement

Primary metric: T2–T6 accuracy, 200 paired hits across 40 image clusters. One percentage point equals two answers. Intervals use 10,000 image-cluster bootstrap resamples with seed 1234; every question from one image stays in its cluster. A confidence interval containing zero establishes neither improvement nor equivalence.

| Pair (A − B) | Role | Δ hit accuracy (pp), 95% CI | Δ hit TTFT (ms), 95% CI | TTFT ratio, 95% CI | Both right / both wrong / A only / B only | Prediction agreement |
|---|---|---:|---:|---:|---:|---:|
| D20+SpatialUniform5 − D20+IndexUniform5 | primary | -4.00 [-9.50, +1.00] | -0.60 [-4.17, +3.21] | 0.998 [0.985, 1.012] | 100 / 78 / 7 / 15 | 0.845 |
| D20+SpatialUniform5 − D25+C0 | practical_baseline | -3.00 [-9.00, +3.50] | +2.07 [-2.61, +6.77] | 1.008 [0.990, 1.026] | 94 / 74 / 13 / 19 | 0.770 |
| D20+Context5 − D20+IndexUniform5 | secondary | +0.00 [-3.00, +3.00] | -0.85 [-4.27, +2.53] | 0.997 [0.984, 1.010] | 111 / 81 / 4 / 4 | 0.960 |
| D20+SpatialUniform5 − D20+Context5 | secondary | -4.00 [-10.00, +2.00] | +0.25 [-3.41, +4.03] | 1.001 [0.987, 1.015] | 99 / 77 / 8 / 16 | 0.835 |
| D20+SpatialUniform5 − D20+Random5 | secondary | -2.00 [-8.00, +3.50] | -0.31 [-3.82, +3.17] | 0.999 [0.986, 1.012] | 98 / 80 / 9 / 13 | 0.820 |
| D21.1+C3.9 [54:10 reference] − D25+C0 | secondary | +0.00 [-4.00, +3.50] | +1.27 [-3.96, +6.38] | 1.005 [0.985, 1.024] | 106 / 80 / 7 / 7 | 0.890 |
| D21.1+C3.9 [54:10 reference] − D20+Context5 | secondary | -1.00 [-4.50, +2.50] | -0.54 [-4.73, +3.63] | 0.998 [0.982, 1.014] | 106 / 78 / 7 / 9 | 0.890 |

The paired counts compare the same image and question. Equal mean accuracy can still hide different correct and incorrect answers. Examples are in pilot_prediction_flips.json.

## Spatial coverage on a separate fixed 8×8 diagnostic grid

The grid below measures occupied cells in each model-input patch branch. coverage.csv also records row-major 64-bin selected-token count histograms for auxiliary IDs and the full dominant plus auxiliary set on this fixed grid. It is separate from the Spatial selector's own regions and is not a semantic-coverage measure.

| Branch | Method | Auxiliary occupied fraction | Dominant ∪ auxiliary occupied fraction |
|---|---|---:|---:|
| base | D20+IndexUniform5 | 0.4129 | 0.8539 |
| base | D20+SpatialUniform5 | 0.4543 | 0.8562 |
| high | D20+IndexUniform5 | 0.7488 | 0.9742 |
| high | D20+SpatialUniform5 | 0.9223 | 0.9941 |

Mean full selected-ID Jaccard (dominant ∪ auxiliary) between Index and Spatial was 0.6849; auxiliary-only Jaccard was 0.0320. The full-set measure includes the shared dominant IDs. Per-branch mean full/auxiliary Jaccard: base 0.6645/0.0291; high 0.6919/0.0330.

Mean originally empty regions/fallback selections per image: base 0.00/0.00, high 0.05/0.05. Mean empty-region/fallback fractions by branch: base 0.0000/0.0000; high 0.0007/0.0007. Fractions are count divided by that branch's quota (zero when quota is zero). Per-image and per-branch quotas, added/removed IDs, and coverage are in pilot_coverage.csv and selection JSON artifacts.

## Turn-level quality and measured costs

| Method | T1 | T2 | T3 | T4 | T5 | T6 | Hit TTFT p50/p95 (ms) | Hit E2E (ms) | Content / structural inclusive retention | Preads/hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.6000 | 0.6250 | 0.6500 | 0.7000 | 0.4750 | 0.7000 | 528.79/604.83 | 549.40 | 1.0000/1.0000 | 0.0 |
| FullLoad | 0.6000 | 0.5750 | 0.6500 | 0.7250 | 0.4750 | 0.7250 | 700.42/931.17 | 754.36 | 1.0000/1.0000 | 64.0 |
| D25+C0 | 0.6000 | 0.6250 | 0.6000 | 0.4750 | 0.4250 | 0.7000 | 265.32/309.56 | 291.16 | 0.2500/0.2629 | 65.0 |
| D20+Context5 | 0.6000 | 0.5750 | 0.5500 | 0.5750 | 0.4750 | 0.7000 | 264.86/317.38 | 294.22 | 0.2500/0.2629 | 65.0 |
| D20+IndexUniform5 | 0.6000 | 0.5750 | 0.6000 | 0.5750 | 0.4750 | 0.6500 | 267.17/326.53 | 295.19 | 0.2500/0.2629 | 65.0 |
| D20+SpatialUniform5 | 0.6000 | 0.5750 | 0.5500 | 0.4500 | 0.4750 | 0.6250 | 266.32/320.06 | 297.47 | 0.2500/0.2629 | 65.0 |
| D20+Random5 | 0.6000 | 0.6000 | 0.5500 | 0.5250 | 0.4500 | 0.6500 | 266.51/315.99 | 295.10 | 0.2500/0.2629 | 65.0 |
| D21.1+C3.9 [54:10 reference] | 0.6000 | 0.5500 | 0.6000 | 0.5500 | 0.4500 | 0.6750 | 264.46/319.63 | 293.14 | 0.2500/0.2629 | 65.0 |

Persistence is an image-level one-time cost outside hit TTFT. The T1 TTFT includes image read/decode, processor, vision, prefill, first-token materialization, and CUDA synchronization. Hit TTFT includes prompt preparation, real SSD pread, transfer, prefill, and first-token materialization. The per-component capture, geometry, spatial selection, clustering, KV repack, write, and fsync timings are in pilot_persistence.csv. Components can overlap and are not added into a fabricated critical path. Metadata activation and DONTNEED page-cache conditioning are separately recorded; DONTNEED does not prove cold SSD controller or NAND state. A Python os.pread wrapper records each returned range in the timed stored-hit path, with the same instrumentation on every stored arm.

## Answers to the eight experiment questions

1. Spatial versus Index hit quality: INCONCLUSIVE. Paired difference -4.00 pp, 95% CI [-9.50, +1.00].
2. Spatial versus D25: INCONCLUSIVE. Paired difference -3.00 pp, 95% CI [-9.00, +3.50].
3. 54:10 versus Context5 allocation: INCONCLUSIVE. Paired difference -1.00 pp, 95% CI [-4.50, +2.50].
4. Spatial and Index both right 100, both wrong 78, Spatial only right 7, Index only right 15; exact prediction agreement 0.845.
5. Fixed 8×8 occupied-cell change: base auxiliary +4.14 pp and dominant plus auxiliary +0.23 pp; high auxiliary +17.34 pp and dominant plus auxiliary +1.99 pp. These geometric diagnostics do not establish why accuracy changed.
6. Every selective arm used exactly ceil(0.25 × N) original content rows. The four D20 arms share dominant IDs; Spatial matches Index base/high auxiliary quotas image by image. The same structural sidecar policy and selective SSD bytes/pread counts were audited on paired hits.
7. Spatial minus Index: T1 TTFT -2.47 ms, persistence +14.13 ms/image, hit TTFT -0.60 ms; exact component costs are in persistence.csv.
8. This reused development pilot cannot establish independent generalization. A favorable observed arm should be tested on a genuinely independent workload before considering a main method change; no main method is changed here.

## Scope and audit

Checkpoint revision: c916e6cdcd760b4cecd1dd4907f84ac649f93b23. NF4 model, BF16 compute, eager attention, FP16 SSD KV, 64-token chunks, seed 1234, greedy max_new_tokens=16.
Runner audit: PASS; 1920/1920 requests. Strict independent audit: PASS; all 1,920 GQA answers independently rescored, 290,176 core selection/geometry/I/O/protection checks and 1,098 statistics/persistence checks passed, with no missing or duplicate requests. The independent receipt is `strict_external_pilot_audit.json`, and its exact re-runnable code is `strict_external_audit.py` in this results directory. The earlier failed smoke run and all prior source/results/stores remain protected. After scoped cleanup, SSD replay requires rebuilding stores from each normal Turn 1.

IMPLEMENTATION: PASS
GEOMETRY VALIDATION: PASS
GPU CORRECTNESS: PASS
PILOT: COMPLETE
SPATIAL VS INDEX QUALITY: INCONCLUSIVE
54:10 ALLOCATION SIGNAL: INCONCLUSIVE
INDEPENDENT HOLDOUT: NOT RUN
QWEN / FULL MT-GQA: NOT RUN
