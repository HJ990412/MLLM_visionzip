# LLaVA KV25 migration pilot

Run: `/home/dblab/hj/mllm_v2/runs/llava_kv25_migration_20260929T032642Z`

The GQA pilot uses 40 fixed images and questions[4:10]. The MT-GQA smoke selection is 4 distinct images; any executed MT phase is not the full 4,061-dialogue experiment.

All T1 requests use normal full-image inference. The old and new Ours hit paths share one immutable image-only physical store but use separate serving contexts. One physical write was measured; the same measured cost is attributed separately to either method under independent deployment.

## gqa

Validation: **VALID**

| Method | Budget unit | Hit content retention | Normal read MB/hit | Total read MB/hit | Hit TTFT ms | Hit quality |
|---|---|---:|---:|---:|---:|---:|
| ReComp | none | N/A | 0.000 | 0.000 | 512.19 | 0.630 |
| FullLoad | full_visual_kv | 1.0000 | 1165.073 | 1165.073 | 691.73 | 0.630 |
| Ours-Chunk25-Legacy | chunk | 0.2499 | 286.052 | 306.079 | 256.59 | 0.575 |
| Ours-KV25-New | visual_kv | 0.2500 | 299.473 | 319.501 | 257.51 | 0.565 |

All-question quality and each turn's quality:

| Method | All questions | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.625 | 0.600 | 0.625 | 0.650 | 0.700 | 0.475 | 0.700 |
| FullLoad | 0.625 | 0.600 | 0.575 | 0.650 | 0.725 | 0.475 | 0.725 |
| Ours-Chunk25-Legacy | 0.579 | 0.600 | 0.625 | 0.600 | 0.500 | 0.450 | 0.700 |
| Ours-KV25-New | 0.571 | 0.600 | 0.625 | 0.600 | 0.475 | 0.425 | 0.700 |

Hit retention and physical reads use distinct denominators. Image mean is the mean of request-level ratios; pooled is the ratio of summed tokens or bytes.

| Method | Content mean / pooled | Visual incl. structural mean / pooled | Normal read / FullLoad mean / pooled | Total read / FullLoad mean / pooled | Unused real rows / MB per hit | Structural normal MB/hit | Cross-file structural duplicate MB/hit | Same-file duplicate MB/hit | Padding MB/hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | N/A | N/A | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.00 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FullLoad | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.00 / 0.000 | 20.028 | 0.000 | 0.000 | 0.000 |
| Ours-Chunk25-Legacy | 0.2499 / 0.2498 | 0.2628 / 0.2627 | 0.2456 / 0.2455 | 0.2628 / 0.2627 | 0.00 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Ours-KV25-New | 0.2500 / 0.2500 | 0.2629 / 0.2629 | 0.2575 / 0.2570 | 0.2747 / 0.2742 | 25.20 / 13.212 | 0.000 | 0.000 | 0.000 | 0.000 |

T1 and persistence are outside cache-hit means; context activation is outside request TTFT.

| Method | T1 TTFT ms | T1 E2E ms | Persistence ms attributed independently | Physical writes in run | Activation ms |
|---|---:|---:|---:|---:|---:|
| ReComp | 512.82 | 540.63 | N/A | 0 | N/A |
| FullLoad | 513.14 | 540.81 | 1178.46 | 40 | 1.00 |
| Ours-Chunk25-Legacy | 514.25 | 542.06 | 1088.09 | 40 | 11.17 |
| Ours-KV25-New | 512.94 | 540.65 | 1088.09 | 0 | 10.79 |

Old/new same selected set: 16 images; different: 24.
New chunk count compared with old: {'increase': 16, 'same': 24}.
Paired hit quality (new − old): {'mean_difference_new_minus_old': -0.009999999999999998, 'ci95': [-0.034999999999999996, 0.010000000000000005], 'image_clusters': 40, 'bootstrap_resamples': 10000, 'seed': 1234}.
Paired hit TTFT ms (new − old): {'mean_difference_new_minus_old': 0.9238392952829599, 'ci95': [-3.5969851164263673, 5.5414397452841495], 'image_clusters': 40, 'bootstrap_resamples': 10000, 'seed': 1234}.

## mt_smoke

Validation: **VALID**

| Method | Budget unit | Hit content retention | Normal read MB/hit | Total read MB/hit | Hit TTFT ms | Hit quality |
|---|---|---:|---:|---:|---:|---:|
| ReComp | none | N/A | 0.000 | 0.000 | 532.99 | 0.500 |
| FullLoad | full_visual_kv | 1.0000 | 1188.299 | 1188.299 | 617.54 | 0.500 |
| Ours-Chunk25-Legacy | chunk | 0.2508 | 293.601 | 311.689 | 277.80 | 0.500 |
| Ours-KV25-New | visual_kv | 0.2500 | 301.990 | 320.078 | 262.67 | 0.500 |

All-question quality and each turn's quality:

| Method | All questions | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.583 | 0.750 | 0.500 | 0.500 | N/A | N/A | N/A |
| FullLoad | 0.583 | 0.750 | 0.500 | 0.500 | N/A | N/A | N/A |
| Ours-Chunk25-Legacy | 0.583 | 0.750 | 0.500 | 0.500 | N/A | N/A | N/A |
| Ours-KV25-New | 0.583 | 0.750 | 0.500 | 0.500 | N/A | N/A | N/A |

Hit retention and physical reads use distinct denominators. Image mean is the mean of request-level ratios; pooled is the ratio of summed tokens or bytes.

| Method | Content mean / pooled | Visual incl. structural mean / pooled | Normal read / FullLoad mean / pooled | Total read / FullLoad mean / pooled | Unused real rows / MB per hit | Structural normal MB/hit | Cross-file structural duplicate MB/hit | Same-file duplicate MB/hit | Padding MB/hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | N/A | N/A | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.00 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FullLoad | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.00 / 0.000 | 18.088 | 0.000 | 0.000 | 0.000 |
| Ours-Chunk25-Legacy | 0.2508 / 0.2509 | 0.2622 / 0.2623 | 0.2470 / 0.2471 | 0.2622 / 0.2623 | 0.00 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Ours-KV25-New | 0.2500 / 0.2500 | 0.2614 / 0.2614 | 0.2545 / 0.2541 | 0.2697 / 0.2694 | 18.00 / 9.437 | 0.000 | 0.000 | 0.000 | 0.000 |

T1 and persistence are outside cache-hit means; context activation is outside request TTFT.

| Method | T1 TTFT ms | T1 E2E ms | Persistence ms attributed independently | Physical writes in run | Activation ms |
|---|---:|---:|---:|---:|---:|
| ReComp | 513.18 | 537.02 | N/A | 0 | N/A |
| FullLoad | 513.07 | 537.02 | 1409.33 | 4 | 1.08 |
| Ours-Chunk25-Legacy | 516.89 | 540.68 | 1449.68 | 4 | 11.31 |
| Ours-KV25-New | 515.29 | 539.19 | 1449.68 | 0 | 11.01 |

Old/new same selected set: 2 images; different: 2.
New chunk count compared with old: {'increase': 1, 'same': 3}.
Paired hit quality (new − old): {'mean_difference_new_minus_old': 0.0, 'ci95': [0.0, 0.0], 'image_clusters': 4, 'bootstrap_resamples': 10000, 'seed': 1234}.
Paired hit TTFT ms (new − old): {'mean_difference_new_minus_old': -15.132999222259969, 'ci95': [-42.19569097040221, 8.990402333438396], 'image_clusters': 4, 'bootstrap_resamples': 10000, 'seed': 1234}.

## Timing and limits

`end_to_end_ttft_ms` starts before prompt and token preparation and ends after the first token decision and CUDA synchronization. OS page-cache conditioning via `posix_fadvise_DONTNEED` is outside this timer; it does not guarantee a cold SSD controller or NAND. Persistence and context activation are separately recorded in each image artifact. Existing historic results were not relabeled or reused as same-run latency.

Canonical manifest hash: `dad64b5b7e0dcbb9858d708b83575c8e43dfc639e2995a9dc2838c18124e9a98`.

## Interpretation and preserved artifacts

ReComp performs a normal multimodal forward, while stored methods reconstruct a cached prefix with the existing FP16 store and BF16 serving conversion. ReComp versus cached output differences therefore have a separate precision/shape interpretation. The KV25 correctness gate compared SSD KV25 with a matched-computation in-memory reference; disagreement with ReComp alone is not evidence of a KV25 error.

The reported read bytes are bytes returned by OS preads, not SSD controller or NAND traffic. During measurement, the source Visual KV and all newly built run-local full Visual KV stores were on SSD. The completed pilot's run-local copies were later removed as recorded below. A 95% paired bootstrap interval containing zero is not an equivalence claim. QA-Chunk25 and ReKV-Chunk25 retain their own chunk-budget semantics and were not changed or rerun here.

Reproduction: [REPRODUCE.md](REPRODUCE.md). Independent raw audit: [independent_audit.json](independent_audit.json). Future QA-Chunk25/ReKV-Chunk25 budget checks: [BASELINE_BUDGET_AUDIT.md](BASELINE_BUDGET_AUDIT.md).

Qwen GPU inference and the full 4,061-dialogue MT-GQA run were not performed in this pilot.

## Main rerun readiness

Implementation and validation gates passed. At initial report finalization, about 35 GiB was free and the run-local full Visual KV stores occupied about 101 GiB. Following the post-run cleanup below, about 135 GiB was free. A larger rerun that retains every new full store still needs a storage plan; regenerating and removing completed run-local stores is possible.

## Post-run cleanup

At the user's request on 2026-09-29T04:13:36.916147+00:00, only `runs/llava_kv25_migration_20260929T032642Z/stores/` was removed (100.51 GiB actual disk usage). Run and results raw JSONL, summaries, manifest, source hashes, CPU/GPU receipts, validation, and report remain. The historical pilot validation was completed before cleanup; store-existence checks require rebuilding the store to rerun. See `runs/llava_kv25_migration_20260929T032642Z/postrun_cleanup.json`.
