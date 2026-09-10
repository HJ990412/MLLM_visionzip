# VisDial v1.0 multi-turn system run

- Schema: `multiturn-results-v1-true-ttft`
- Dialogs / turns: 2 / 20
- History: `gold_teacher_forced`
- Calibration: `caption_only_pre_dialog`
- Page cache: `cold`
- Max new tokens: 16
- Validation: `PASS`

## Overall

| Method | Quality† | TTFT mean / p50 / p95 (ms) | SSD read (MB) | Selector (ms) | Logical kept tokens | SSD total / Full KV |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.550 | 546.8/531.2/584.3 | 0.0 | 0.00 | —% | 0.0% |
| FullLoad | 0.550 | 655.2/665.0/734.1 | 1175.5 | 0.00 | 100.0% | 100.0% |
| SparseVLM 25% | 0.500 | 683.2/698.9/756.1 | 860.3 | — | 24.6% | 73.2% |
| Static+Diverse 25% | 0.500 | 325.4/319.9/363.8 | 301.1 | 15.35 | 24.2% | 25.6% |
| Static+Diverse 50% | 0.500 | 533.6/536.2/577.6 | 598.0 | 21.95 | 48.6% | 50.9% |

† VisDial system run의 quality는 normalized generative match 보조지표이며 공식 VisDial 점수가 아니다.
`ssd_read_chunk_units`는 K/V/probe 파일별 chunk-equivalent 합이며 unique selected chunk 수가 아니다.

## Hypothesis checks

The machine-readable values are in `analysis.json`. The run records every turn separately; `per_turn.csv` and `per_active_images.csv` preserve the two workload axes.

## Timing boundary

`request start → selector → SSD pread → reconstruction/scatter → prompt prefill → first output token argmax → CUDA synchronize` is TTFT. Later autoregressive generation is `decode_ms`; `e2e_ms ≈ ttft_ms + decode_ms` is validated row by row.

## Limitations

- Static+Diverse는 SSD에서 읽는 payload를 줄이지만 현재 PrefixCache는 full-length GPU KV tensor를 할당한다. selected KV bytes를 실제 GPU allocation 절감으로 해석하면 안 된다.
- VisDial generative match는 보조 지표다. candidate conditional-likelihood 기반 MRR/R@K/Mean Rank/NDCG와 구분한다.
- caption-only importance calibration만 사용했으며 평가 turn/future answer는 reorder나 selector 입력에 들어가지 않는다.
