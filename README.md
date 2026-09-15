# Storage-aware Visual KV Reuse for Multi-turn MLLM Serving

LLaVA-NeXT의 image-prefix Visual KV를 첫 요청에서 한 번 만들고 SSD에 저장한 뒤,
후속 요청에서 필요한 연속 prefix만 읽어 vision recomputation과 SSD traffic을 줄이는
실험 구현이다. 현재 경로는 calibration question이나 온라인 selector 없이
Vision Encoder의 image-only saliency로 KV를 한 번 재배치하고, cache-hit 요청에서는
각 layer의 physical first-k chunk를 순차적으로 읽는다.

- 모델: `llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4, eager attention
- 저장 단위: 64-token chunk, separator KV sidecar
- 주요 데이터: GQA 40 images / 240 questions, VisDial v1.0 100 dialogs × 10 turns
- 주요 지표: request 시작부터 first output token CUDA synchronization까지의 end-to-end TTFT
- 환경: `conda activate mllm_ft` (torch 2.5.1, transformers 4.57.6), RTX 4090 24GB

핵심 구현은 `mmimpress/store.py`, `mmimpress/serve.py`, `mmimpress/piggyback.py`에,
실험·분석은 `scripts/`, 검증된 표와 raw-derived artifact는 `results/`에 있다.

## 1. Calibration-free ImageOnly VisionZip repack + Prefix (2026-09-11)

Vision Encoder penultimate layer의 CLS-to-patch attention을 head 방향으로 합산해
image-only saliency를 만들고, real patch만 stable descending sort한 하나의 global
permutation을 모든 LLM KV layer에 동일하게 적용했다. Separator는 normal budget에서
제외해 physical tail에 stable 배치하고 `sep_kv.bin` sidecar로 항상 읽는다. 최종
경로는 fresh raster KV를 SSD에 쓴 뒤 다시 읽어 바꾸는 방식이 아니라, prefix KV가
메모리에 있을 때 permutation한 뒤 최종 layout으로 한 번만 쓰는 direct-write다.
온라인 요청은 score/static/diversity 계산 없이 각 layer의 physical first-k chunk만
읽는다.

동일 GQA 40 images / `questions[4:10]` 240 questions, schema-v2 true TTFT, cold page
cache, 64-token chunk 조건 결과다. 서로 다른 run 사이 절대 latency 변동이 있으므로
각 arm의 raw 값과 함께 같은 VisionZip run의 FullLoad 대비 감소율을 사용한다.

| Layout / retrieval | Calib Q | Accuracy | True TTFT | SSD read | preads | Selector |
|---|---:|---:|---:|---:|---:|---:|
| Raster / FullLoad | 0 | 62.08% | 782.11 ms | 1165.073 MB | 64 | - |
| Pixel recomputation / **ReComp** (historical schema-v2) | - | **62.50%** | **503.81 ms** | **0.000 MB** | **0** | - |
| Raster / Prefix25 | 0 | 2.92% | 289.06 ms | 306.079 MB | 65 | 0.210 ms |
| Morton / Prefix25 | 0 | 2.92% | 289.24 ms | 306.079 MB | 65 | 0.226 ms |
| Calib4 importance legacy / Prefix25 | 4 | 60.42% | **262.16 ms** | 306.079 MB | 65 | 0.194 ms |
| **VisionZip image-only / Prefix25** | **0** | **57.92%** | **284.74 ms** | **306.079 MB** | **65** | **0.212 ms** |
| VisionZip image-only / Prefix50 | 0 | 60.00% | 460.86 ms | 603.875 MB | 65 | 0.331 ms |
| Calib4 separator-tail matched / Prefix25 | 4 | 60.83% | 290.44 ms | 306.079 MB | 65 | 0.208 ms |

ReComp는 보존된 calib4 run의 원시 240행을 재사용했다. 이미지·질문·gold·순서와
index/workload hash, 모델·decoding·schema-v2 TTFT 조건이 모두 일치한다. ReComp는
pixel부터 매 요청 재계산하므로 KV-store read와 pread가 모두 0이고, SSD cold-cache와
separator policy는 적용 대상이 아니다. ImageOnly Prefix25는 ReComp보다 accuracy가
4.58 pp 낮지만 TTFT는 43.48% 짧고, Prefix50은 accuracy가 2.50 pp 낮지만 TTFT는
8.53% 짧다. 다만 ReComp는 과거 run이고 `first_token_id`가 저장되기 전 artifact라서
quality는 정확히 paired 비교할 수 있지만 새 image-only run과의 절대 latency 비교는
cross-run 비교로 해석해야 한다.

ImageOnly Prefix25는 같은 physical-layout run의 FullLoad 61.25%보다 3.33 pp,
canonical Raster FullLoad보다 4.17 pp 낮고, legacy calib4 Prefix25보다 2.50 pp,
separator-tail matched calib4보다 2.92 pp 낮다. 반면 Raster/Morton Prefix25보다는
55.00 pp 회복했다. 같은 VisionZip run에서 SSD byte는 73.73%, true TTFT는 60.30%
감소했다. Prefix retrieval은 request당 65 preads이고 selector는 0.212 ms다.

First 25%의 importance-mass coverage는 다음과 같다. 괄호는 image-layer macro / 전체
mass-weighted aggregation이다.

| Layout | VisionZip saliency | 분석 전용 calib4 SparseVLM importance |
|---|---:|---:|
| Raster | 29.56% / 29.48% | 25.27% / 24.20% |
| Morton | 28.54% / 28.43% | 24.99% / 24.02% |
| **VisionZip image-only** | **73.15% / 73.27%** | **41.60% / 36.74%** |
| Calib4 legacy | 50.53% / 50.68% | 73.50% / 61.73% |
| Calib4 separator-tail matched | 51.87% / 52.01% | 74.53% / 63.10% |

즉 image-only repack은 자신이 정의한 visual saliency는 앞쪽에 잘 모으지만,
질문 기반 importance와의 정렬은 calib4보다 약하다. 이것이 25%에서 남은 quality
gap과 일치한다. SparseVLM score는 이 사후 coverage 분석에만 사용했고 layout 생성,
저장, selection, serving에는 전달하지 않았다.

질문 독립성 검증은 3 images × 3 questions에서 pixel/image size/saliency/permutation이
모두 100% 동일했고, layout scoring 중 decoder-layer forward와 `calibrate_image`,
`Server.raters`, SparseVLM Q/K 호출은 모두 0이었다. Direct-write와 fresh-raster 후
post-hoc prototype도 3 images의 Prefix25 payload 및 9개 prediction/first token이
전부 동일했다. 기존 canonical store와 과거 result tree의 hash도 모두 보존됐다.

엄격한 FullLoad output identity는 **238/240(99.17%)로 FAIL**이다. 두 Yes/No first-token
경계 사례가 바뀌어 Raster 62.08%와 repacked 61.25% 사이에 -0.83 pp가 생겼다. 이를
통과로 숨기지 않았다. 다만 해당 두 이미지의 stored-to-original mapping 후 FP16 K/V
1,175,453,696개 원소는 차이 0이고 `sys_kv`/`v_hidden` hash와 240개 I/O가 모두
동일했다. 따라서 structural integrity는 통과했고, 원인은 eager BF16 attention의
물리 순서별 reduction-order 민감성으로 분류했다.

Direct build의 image당 평균은 saliency 45.34 ms, permutation 0.66 ms, KV repack
43.78 ms, SSD write 468.88 ms, total ingestion 1305.09 ms였다. Raster build
1166.23 ms 대비 일회성 증가는 138.87 ms/image이며, 이미지당 1/5/10/20 requests에서
각각 138.87/27.77/13.89/6.94 ms/request로 상각된다. Mapping metadata는 평균
68.2 KB/image다.

사전 판정 규칙에 따른 결론은 **PARTIAL GO**다. Calibration 없이 Raster/Morton보다
압도적으로 높은 quality와 sequential SSD locality를 얻었지만, 25%에서 calib4 및
FullLoad와의 gap이 아직 의미 있어 그대로 새 main contribution으로 전환할 수준의
STRONG GO는 아니다. 50%는 quality를 회복하지만 SSD/TTFT 이점이 감소한다. ReComp
통합본은 전체 2,160-row paired 결과, 64,000-row cumulative coverage,
bootstrap/McNemar, strict validation과 재현 정보를
[`results/image_only_repack_recomp/`](results/image_only_repack_recomp/)에 저장했다.
기존 1,920-row snapshot인
[`results/image_only_repack/`](results/image_only_repack/)은 수정하지 않고 그대로
보존했다.


## 2. ImageOnly VisionZip sequential-Prefix budget sweep: 20% / 25% / 30% (2026-09-14)

ImageOnly VisionZip permutation을 고정한 뒤 budget만 바꾼 순차 Prefix sweep이다.
GQA 40 images / `questions[4:10]` 240 questions, schema-v2 true TTFT, cold page
cache 조건에서 측정했다. Calibration question, 온라인 score, diversity, budget별
layout은 사용하지 않았고, `k = round(n_chunks * budget)`만 바뀐다. 각 layer의
연속 first-k chunk는 하나의 span으로 합쳐 읽으므로 budget과 무관하게 request당
64 normal pread + 1 separator-sidecar pread를 사용한다.

| Budget | Accuracy | Δ vs same-run FullLoad | True TTFT | TTFT 감소 | SSD read | SSD 감소 |
|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 61.25% | -- | 727.37 ms | -- | 1165.07 MB | -- |
| 20% | 54.58% | −6.67 pp | **233.78 ms** | **67.86%** | **252.39 MB** | **78.34%** |
| **25%** | **57.92%** | **−3.33 pp** | **265.17 ms** | **63.54%** | **306.08 MB** | **73.73%** |
| 30% | 58.75% | −2.50 pp | 300.06 ms | 58.75% | 371.51 MB | 68.11% |
| 45% | 60.42% | −0.83 pp | 404.78 ms | 44.35% | 551.03 MB | 52.70% |
| 50% | 60.00% | −1.25 pp | 435.31 ms | 40.15% | 603.87 MB | 48.17% |

20→25%는 53.69 MB와 31.39 ms를 더 써서 accuracy를 +3.33 pp 회복한다.
25→30%의 추가 회복은 +0.83 pp이고, 추가 비용은 65.43 MB / 34.89 ms다. 따라서
aggressive operating point는 25%(FullLoad 대비 loss ≤4 pp), 관측상 balanced
operating point는 45%(loss −0.83 pp)다. 표본은 40개 image cluster이므로 1 pp
미만의 차이를 확정적으로 해석하지 않는다.

ReComp도 같은 조건에서 포함한 비교, paired CI, Pareto frontier, coverage와 raw
artifacts는 [`results/image_only_repack_budget_sweep/`](results/image_only_repack_budget_sweep/)와
[`results/image_only_repack_budget_sweep_with_recomp/`](results/image_only_repack_budget_sweep_with_recomp/)
에 있다.

## 3. VisDial cache-hit serving with Turn-1 piggyback persistence (2026-09-15)

### Experimental contract

VisDial v1.0 validation 100 dialogs × 10 turns에서 method마다 1,000 requests를
실행했다. Paper-facing main population은 Turn 1을 제외한 **Turns 2–10**, 즉
100 dialogs × 9 cache-hit turns = 900 requests/method다. History는 gold
teacher-forced이고 main metric은 `end_to_end_ttft_ms`다.

Turn 1에는 모든 arm이 동일하게 정상 pixel-based multimodal inference를 한 번 수행한다.
Prefix25/45는 이 forward에서 Visual KV와 image-only saliency를 함께 capture하며,
별도의 vision forward나 미리 준비된 SSD cache를 사용하지 않는다. Image당 한 번의
persistence 비용은 cache-hit request TTFT와 분리해서 보고한다.

### Main cache-hit table — Turns 2–10

| Method | Aux quality† | TTFT mean | p50 | p95 | Δ TTFT vs ReComp | TTFT 감소 | E2E mean | SSD MB/request | SSD ratio vs FullLoad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.439 | 539.30 ms | 548.22 | 595.30 | 0.00 | 0.00% | 571.73 ms | 0.00 | — |
| FullLoad | 0.440 | 686.29 ms | 682.80 | 833.99 | +146.99 | −27.26% | 723.88 ms | 1169.43 | 100.00% |
| **Prefix25** | 0.418 | **278.34 ms** | 277.85 | 334.32 | **−260.96** | **48.39%** | **325.26 ms** | **306.83** | **26.24%** |
| **Prefix45** | **0.440** | **417.59 ms** | 419.84 | 494.78 | **−121.71** | **22.57%** | **458.43 ms** | **554.80** | **47.44%** |

† 기존 normalized generative-match auxiliary score이며 공식 VisDial
MRR/R@K/Mean Rank/NDCG 점수가 아니다.

Prefix25는 ReComp 대비 TTFT 48.39%, request E2E 43.11%를 줄이는 대신 auxiliary
quality가 0.021 낮다. Prefix45는 ReComp와 동등한 auxiliary quality를 유지하면서
TTFT 22.57%, request E2E 19.82%를 줄였다. FullLoad가 ReComp보다 느리므로, 이
환경에서는 KV를 SSD에 저장하는 것만으로는 부족하고 read bytes를 줄여야 한다.

### Cache-hit I/O breakdown

| Method | SSD MB/request | SSD read | OS preads/request | Scatter | Raw prefill | First-k planning |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.00 | 0.00 ms | 0.00 | 0.00 ms | 523.91 ms | 0.00 ms |
| FullLoad | 1169.43 | 418.19 ms | 64.00 | N/A | 680.96 ms | 0.00 ms |
| Prefix25 | 306.83 | 137.34 ms | 65.00 | 51.51 ms | 56.05 ms | 0.21 ms |
| Prefix45 | 554.80 | 218.17 ms | 65.00 | 82.38 ms | 56.68 ms | 0.33 ms |

### Turn-1 sanity

| Method | TTFT mean / p50 / p95 | Prediction agreement | First-token agreement | Vision forwards | SSD read |
|---|---:|---:|---:|---:|---:|
| ReComp | 519.42 / 525.40 / 567.87 ms | 100.00% | 100.00% | 1.00 | 0 B |
| FullLoad | 519.56 / 527.03 / 563.79 ms | 100.00% | 100.00% | 1.00 | 0 B |
| Prefix25 | 519.54 / 527.15 / 565.72 ms | 100.00% | 100.00% | 1.00 | 0 B |
| Prefix45 | 520.15 / 527.62 / 563.50 ms | 100.00% | 100.00% | 1.00 | 0 B |

네 arm의 Turn-1 입력·prediction·first token이 모두 일치하며, 각각 vision forward를
정확히 한 번 실행하고 SSD는 읽지 않는다. Saliency와 KV capture는 같은 forward에
piggyback한다.

### One-time persistence overhead

| Component | Mean | p50 | p95 |
|---|---:|---:|---:|
| Saliency extra | 0.01 ms | 0.01 | 0.01 |
| Saliency postprocess | 0.08 ms | 0.08 | 0.09 |
| Permutation | 0.64 ms | 0.65 | 0.70 |
| KV repack | 148.02 ms | 141.93 | 207.72 |
| Buffered SSD write | 445.06 ms | 434.92 | 547.89 |
| fsync | 294.27 ms | 259.93 | 443.09 |
| **Total persistence** | **979.75 ms** | **961.25** | **1239.38** |

평균 one-time write는 image당 1191.74 MB다. 실제 구현에서는 source Answer 1 직후
동기적으로 실행됐으며 background overlap으로 간주하지 않는다. 이 비용을 한 번
부과하는 conservative back-to-back 누적 E2E에서 Prefix25는 Turn 5, Prefix45는
Turn 10에 ReComp보다 빨라진다.

모든 52개 fail-closed 검증이 통과했다. 상세 timing boundary, turn별 추이, raw-derived
CSV와 검증 근거는
[`results/visdial_cache_hit_analysis/`](results/visdial_cache_hit_analysis/)에 있다.
