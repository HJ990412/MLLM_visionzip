# VisDial v1.0 multi-turn system run

- Schema: `multiturn-results-v1-true-ttft`
- Dialogs / turns: 2 / 20
- History: `gold_teacher_forced`
- Calibration: `caption_only_pre_dialog`
- Page cache: `cold`
- Max new tokens: 16
- Validation: `PASS`

## Overall

| Method | Quality† | TTFT mean / p50 / p95 (ms) | SSD read (MB) | Selector (ms) | Selected KV |
|---|---:|---:|---:|---:|---:|
| ReComp | 0.550 | 549.8/534.5/584.5 | 0.0 | 0.00 | —% |
| FullLoad | 0.550 | 631.0/620.4/697.4 | 1175.5 | 0.00 | 100.0% |
| SparseVLM 25% | 0.500 | 684.4/686.4/757.0 | 860.3 | 605.30 | 24.6% |
| Static+Diverse 25% | 0.500 | 323.8/319.3/361.7 | 301.1 | 15.74 | 24.2% |
| Static+Diverse 50% | 0.500 | 529.3/528.5/572.3 | 598.0 | 21.92 | 48.6% |

† VisDial system run의 quality는 normalized generative match 보조지표이며 공식 VisDial 점수가 아니다.

## Hypothesis checks

The machine-readable values are in `analysis.json`. The run records every turn separately; `per_turn.csv` and `per_active_images.csv` preserve the two workload axes.

## Timing boundary

`request start → selector → SSD pread → reconstruction/scatter → prompt prefill → first output token argmax → CUDA synchronize` is TTFT. Later autoregressive generation is `decode_ms`; `e2e_ms ≈ ttft_ms + decode_ms` is validated row by row.

## Limitations

- Static+Diverse는 SSD에서 읽는 payload를 줄이지만 현재 PrefixCache는 full-length GPU KV tensor를 할당한다. selected KV bytes를 실제 GPU allocation 절감으로 해석하면 안 된다.
- VisDial generative match는 보조 지표다. candidate conditional-likelihood 기반 MRR/R@K/Mean Rank/NDCG와 구분한다.
- caption-only importance calibration만 사용했으며 평가 turn/future answer는 reorder나 selector 입력에 들어가지 않는다.
