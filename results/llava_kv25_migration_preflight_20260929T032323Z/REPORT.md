# LLaVA KV25 migration pilot

Run: `/home/dblab/hj/mllm_v2/runs/llava_kv25_migration_preflight_20260929T032323Z`

GQA runner preflight uses only the first frozen image; it is not the 40-image pilot. The MT-GQA smoke selection is 4 distinct images; any executed MT phase is not the full 4,061-dialogue experiment.

All T1 requests use normal full-image inference. The old and new Ours hit paths share one immutable image-only physical store but use separate serving contexts. One physical write was measured; the same measured cost is attributed separately to either method under independent deployment.

## gqa

Validation: **VALID**

| Method | Budget unit | Hit content retention | Normal read MB/hit | Total read MB/hit | Hit TTFT ms | Hit quality |
|---|---|---:|---:|---:|---:|---:|
| ReComp | none | N/A | 0.000 | 0.000 | 497.96 | 0.200 |
| FullLoad | full_visual_kv | 1.0000 | 1124.073 | 1124.073 | 878.09 | 0.400 |
| Ours-Chunk25-Legacy | chunk | 0.2424 | 268.435 | 285.213 | 298.64 | 0.400 |
| Ours-KV25-New | visual_kv | 0.2500 | 301.990 | 318.767 | 312.89 | 0.400 |

Old/new same selected set: 0 images; different: 1.
New chunk count compared with old: {'increase': 1}.
Paired hit quality (new − old): {'mean_difference_new_minus_old': 0.0, 'ci95': [0.0, 0.0], 'image_clusters': 1, 'bootstrap_resamples': 10000, 'seed': 1234}.
Paired hit TTFT ms (new − old): {'mean_difference_new_minus_old': 14.24590377137065, 'ci95': [14.24590377137065, 14.24590377137065], 'image_clusters': 1, 'bootstrap_resamples': 10000, 'seed': 1234}.

## Timing and limits

`end_to_end_ttft_ms` starts before prompt and token preparation and ends after the first token decision and CUDA synchronization. OS page-cache conditioning via `posix_fadvise_DONTNEED` is outside this timer; it does not guarantee a cold SSD controller or NAND. Persistence and context activation are separately recorded in each image artifact. Existing historic results were not relabeled or reused as same-run latency.

Frozen manifest SHA256: `7fcc1526dc6d35e12e10d127cc51e70c2ed56d2a0e1c5b4378c89b67214bac75`.
