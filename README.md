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

## SparseVLM-based QA-Select25 SSD baseline (2026-09-20)

Ours의 image-only repacked Prefix25와 같은 nominal 25%에서 비교하는 main baseline을
추가했다. QA-Select25는 Turn 1의 normal pixel inference에서 original/raster Visual-KV,
visual hidden state, first 3 probe-head K를 piggyback 저장한다. 이후 각 질문마다 현재
text raters와 layer별 rater-to-visual Q/K attention으로 spatial token의 정확히 25%를
고르고, 해당 token이 속한 64-token SSD chunk만 읽어 scatter/mask한다. SparseVLM의
`select_raters`, `rater_visual_scores_from_qk(head_reduce="mean")`, `select_topk`를
재사용하지만 adaptive budget, recycling/merging, diversity, calibration, IMPRESS
Jaccard voting/full-layer fallback은 사용하지 않는다.

격리된 GQA 40 images / 240 questions 결과다. Accuracy는 전체 질문, 나머지는
cache-hit Turns 2–6 평균이다.

| Method | Accuracy | TTFT | Nominal KV | Logical kept | Actual SSD | SSD ratio | Selector | Touched chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | 516.50 ms | — | — | 0.00 MB | 0.00% | 0.00 ms | — |
| FullLoad | 62.50% | 650.73 ms | 100% | 100% | 1165.07 MB | 100.00% | 0.00 ms | 100.00% |
| **QA-Select25** | **60.00%** | 903.46 ms | 25% | 25% | 1196.50 MB | 102.72% | 112.52 ms | 96.36% |
| **Ours25** | **57.92%** | **258.97 ms** | 25% | 26.28% | **306.08 MB** | **26.28%** | **0.20 ms** | **24.28%** |

QA는 Ours보다 accuracy가 +2.08 pp 높지만 TTFT는 +644.49 ms, 3.49×이고, probe와
분산 chunk 때문에 logical 25%임에도 FullLoad보다 2.72% 더 많은 SSD bytes를 읽었다.
200개 cache-hit request의 400개 within-image 질문쌍은 모두 다른 selected-token set을
보였고 layer-macro mean Jaccard는 0.7334였다. Ours는 질문마다 같은 first-k prefix,
query-scoring 0, layer당 contiguous run 1을 유지했다.

재현 runner는 `scripts/49_eval_query_aware_baseline.py`, artifact 보호는
`scripts/50_protect_query_aware_artifacts.py`, fail-closed 최종 보고서는
`scripts/51_report_query_aware_baseline.py`다. 27개 run-level gate, 전체 196개 CPU
test, 기존 13,120개 run/result artifact의 before/after hash가 통과했다. 정확한 1–20
분석, per-layer selection/Jaccard, raw timing과 hashes는
[`results/query_aware_baseline/gqa40_240_final/`](results/query_aware_baseline/gqa40_240_final/)에
있다. 이 단계에서는 MT-GQA-reconstructed, ConvBench, VisDial full rerun을 수행하지
않았다.

### Chunk-aligned follow-up: QA-Chunk25

같은 query-aware score를 token 단위가 아니라 SSD chunk 단위로 집계해 Ours25와
동일한 chunk 수만 읽는 QA-Chunk25도 평가했다. 동일한 GQA 40 images / 240
questions에서 QA-Chunk25는 accuracy 58.33%, cache-hit TTFT 439.31 ms,
348.06 MB/request를 기록했다. QA-Token25보다 I/O locality와 TTFT가 크게 개선됐지만,
Ours25(57.92%, 253.72 ms, 306.08 MB)보다 accuracy는 +0.41 pp, TTFT는
+185.59 ms였다. 상세 결과와 검증은
[`results/query_aware_chunk_baseline/gqa40_240_20260919T155349Z/`](results/query_aware_chunk_baseline/gqa40_240_20260919T155349Z/)에 있다.

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


## 4. Cross-dataset ImageOnly generalization (2026-09-17)

Calibration question 없이 동일한 image-only VisionZip repack + sequential Prefix를
GQA-large(395 images / 1,185 questions), VQAv2(250 / 1,000), TextVQA(500 / 500)에
적용했다. 서버 측 end-to-end TTFT를 측정하고, quality는 각 dataset의 기존
repository scorer로 평가했다.

| Dataset | FullLoad | Prefix25 | Δ P25 | Prefix45 | Δ P45 | P25 TTFT 감소 vs ReComp | P45 TTFT 감소 vs ReComp |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA-large | 63.46% | 58.73% | −4.73 pp | 62.87% | −0.59 pp | 44.46% | 16.33% |
| VQAv2 | 81.57% | 76.33% | −5.23 pp | 80.50% | −1.07 pp | 44.50% | 15.84% |
| TextVQA | 61.93% | 57.87% | −4.07 pp | 62.20% | +0.27 pp | 46.27% | 18.77% |

Prefix25는 세 dataset에서 일관된 TTFT 이득을 보였으나 quality 손실은 4–5 pp다.
Prefix45도 세 dataset 모두 ReComp보다 빠르지만, quality-oriented 조건을 모두
충족한 것은 아니다. 종합 판정은 **PARTIALLY SUPPORTED**다. VQAv2/TextVQA
점수는 기존 구현과의 비교를 위한 repository scorer 값이며 공식 evaluator와
byte-for-byte 동일하다고 주장하지 않는다. Dataset별 CI, SSD I/O, category 분석,
검증과 재현 정보는
[`results/image_only_generalization/`](results/image_only_generalization/)에 있다.

## 5. Full MT-GQA three-turn evaluation (2026-09-17)

4,061 dialogues × 3 turns × 4 methods, 총 48,732 requests를 평가했다. 원 논문의
정확한 dialogue artifact를 확보하지 못해 GQA testdev-balanced에서 결정적으로
재구성한 workload이며, 원 benchmark와 동일하다고 주장하지 않는다. Turn 2/3에는
이전 gold answer를 history에 넣는 teacher-forced 조건을 사용한다. 아래 quality는
첫 gold answer와의 **strict normalized exact match**다.

| Method | Turn 1 | Turn 2 | Turn 3 | 평균 | Δ 평균 vs FullLoad | Cache-hit TTFT (Turns 2–3) | SSD MB/request |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 63.19% | 73.65% | 75.18% | 70.67% | −0.02 pp | 528.47 ms | 0.00 |
| FullLoad | 63.19% | 73.73% | 75.15% | 70.69% | -- | 737.07 ms | 1173.92 |
| Prefix25 | 63.19% | 70.65% | 73.23% | 69.02% | −1.67 pp | **275.61 ms** | **307.82** |
| Prefix45 | 63.19% | 72.94% | 74.66% | 70.26% | −0.43 pp | 428.16 ms | 555.57 |

Cache-hit TTFT는 ReComp 대비 Prefix25 **47.85%**, Prefix45 **18.98%** 낮았다.
FullLoad는 ReComp보다 느렸다. Turn 1에는 네 방법의 prediction과 first token이
100% 일치했고, cache persistence는 Turn 1 뒤 한 번 수행해 cache-hit TTFT에서
분리했다. Quality·efficiency 판정은 모두 **SUPPORTED**다. 다만 이 판정은 위의
재구성 workload와 사전 정의된 기준에 한정된다.

실험 설정, 통계, 검증, per-turn 결과는
[`results/mt_gqa_full/`](results/mt_gqa_full/)에 있다. 337 MB raw JSONL은
GitHub 단일 파일 제한 때문에 무손실 gzip(`raw.jsonl.gz`)으로 게시하며,
로컬 원본 `raw.jsonl`은 그대로 보존한다.

## 6. ConvBench full multi-turn evaluation (2026-09-18)

577 conversations, 6,924 method-turn requests를 완주했다. Cache-hit pooled TTFT는
ReComp 597.46 ms, FullLoad 759.04 ms, Prefix25 344.39 ms, Prefix45 489.62 ms였고,
SSD read는 각각 0, 1152.65, 305.72, 545.20 MB/request였다. 따라서 Prefix25와
Prefix45는 ReComp 대비 TTFT를 각각 42.36%, 18.05% 줄였다. Judge unresolved
543건(7.84%)은 고정 denominator에서 0 win으로 처리했으며, 상세 quality와
진단 결과는 [`results/convbench_full/full_577/`](results/convbench_full/full_577/)에
있다.

## 7. MT-GQA gold/generated-history four-arm comparison (2026-09-22)

QA-Chunk25를 포함해 ReComp, FullLoad, QA-Chunk25, Ours25 네 arm을 4,061개
3-turn dialogue에서 gold-history와 method별 generated-history 조건으로 각각
평가했다. Gold-history 평균 정확도는 70.67%, 70.72%, 68.93%, 69.02%였고,
generated-history에서는 66.72%, 66.65%, 64.93%, 65.13%였다. Cache-hit TTFT는
gold-history 기준 QA-Chunk25 450.04 ms, Ours25 278.13 ms였으며, 두 방법의 평균
정확도 차이는 −0.09 pp(QA−Ours)였다. Generated-history에서도 차이는 −0.20 pp로
작았지만 QA-Chunk25의 selector 비용은 약 96.7 ms였다. 전체 per-turn 결과,
paired test, error propagation, I/O 분석은
[`results/mt_gqa_4arm_history_comparison_20260921T081420Z/`](results/mt_gqa_4arm_history_comparison_20260921T081420Z/)에 있다.

## 8. MT-VQA-v2 generated-history evaluation (2026-09-22)

ReComp, FullLoad, QA-Chunk25, Ours25를 250개 이미지의 3-turn dialogue에서
평가했다(총 3,000 requests). 평균 soft VQA score는 각각 80.58%, 80.49%,
79.20%, 78.27%였고, cache-hit T2–T3 TTFT는 529.83, 707.68, 462.61,
286.41 ms였다. QA-Chunk25와 Ours25의 평균 점수 차이는 +0.93 pp이나
이미지 단위 bootstrap 95% CI [-0.84, +2.76] pp에는 0이 포함된다.
이 데이터는 공개 MT-VQA-v2 대화 원본이 없어 구성한 250-image subset이며
공식 benchmark 재현으로 해석하지 않는다. 자세한 결과는
results/mt_vqa_v2_generated_4arm_20260922T064727Z/README.md에 있다.

## 9. MPIC-32 SSD adaptation (2026-09-22)

동일한 GQA 40 images / 240 questions의 다섯 방법 비교에서 MPIC-32의
전체 accuracy는 62.08%, cache-hit TTFT는 646.72 ms, SSD read는
1165.335 MB/request였다. Ours25는 57.92%, 255.56 ms,
306.079 MB/request였다. MPIC는 전체 image context를 유지하면서
작은 prefix를 다시 계산하므로 25% 저장 budget 방법과 조건이 다르다.
이 실험은 단일 이미지 SSD adaptation이며 논문의 multi-image overlap은
측정하지 않았다. 계약과 분석은 docs/mpic_baseline_contract.md 및
results/mpic_baseline/gqa40_240_pilot_20260922T102113Z/ANALYSIS.md에 있다.

## 10. ReKV-Chunk25 SSD adaptation (2026-09-23)

동일한 GQA slice의 여섯 방법 비교에서 ReKV-Chunk25는 전체 accuracy
57.92%, cache-hit TTFT 568.51 ms, SSD read 305.236 MB/request였다.
같은 run의 Ours25는 57.92%, 257.32 ms, 306.079 MB/request였다.
ReKV의 video-frame retrieval을 64-token SSD chunk에 적용한 것으로,
원 논문의 streaming-video 시스템 수치를 재현한 결과는 아니다.
구현 계약과 검증은 docs/rekv_baseline_contract.md 및
results/rekv_baseline/gqa40_240_pilot_20260923T035750Z/ANALYSIS.md에 있다.

## 11. MT-GQA generated-history five-arm main (2026-09-23)

ReComp, FullLoad, MPIC-32, ReKV-Chunk25, Ours25를 같은 MT-GQA-reconstructed
4,061개 3-turn dialogue에서 평가했다(398 images, 총 60,915 requests).
각 방법이 생성한 이전 답변을 다음 turn의 history에 넣었다. 정확도는
strict normalized exact match이며, TTFT와 SSD I/O는 각 방법의 cache-hit
Turns 2–3 요청 8,122개의 평균이다.

| Method | T1 | T2 | T3 | Avg | Cache-hit TTFT | SSD MB/hit |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 63.19% | 68.11% | 68.85% | 66.72% | 531.93 ms | 0.00 |
| FullLoad | 63.19% | 68.06% | 68.70% | 66.65% | 744.66 ms | 1173.92 |
| MPIC-32 (adapted) | 63.19% | 68.01% | 69.10% | 66.77% | 718.02 ms | 1174.18 |
| ReKV-Chunk25 (adapted) | 63.19% | 66.68% | 67.10% | 65.66% | 604.72 ms | 307.45 |
| Ours25 | 63.19% | 65.48% | 66.73% | 65.13% | **301.54 ms** | 307.82 |

Ours25의 cache-hit TTFT는 ReKV-Chunk25보다 50.14% 낮았고, 평균 정확도는
0.53 pp 낮았다(이미지 단위 bootstrap 95% CI: −1.14~+0.10 pp).
Persistence를 매 3-turn session에 부과한 standalone-equivalent 평균은
Ours25 2259.67 ms, ReComp 1668.10 ms였다. 결과와 검증 자료는
[five-arm generated-history results](results/mt_gqa_5arm_generated_20260923T064333Z/)에 있다.
이 실험은 재구성한 MT-GQA workload와 SSD에 맞춘 MPIC·ReKV 구현을 사용한다.

## 12. Qwen2.5-VL image-only 포팅·정합성 확인 (2026-09-28)

Qwen/Qwen2.5-VL-7B-Instruct의 native BF16 Visual KV 저장·재사용 경로를
추가했다. 최초 GPU 검증은 15개 gate 중 13개만 통과하여 초기 pilot은
진단 결과로만 남겼다. 원인 분석 후 고정된 v2 계약의 재검증에서 10쌍의
matched in-memory/SSD correctness gate가 PASS했고, GQA 40-image 및
MT 40-dialogue Chunk25 pilot을 별도로 재실행했다. 기존 ReComp와의 BF16/NF4
수치 차이는 사라졌다고 주장하지 않는다. 초기 실패와 후속 PASS를 구분한
[포팅 보고서](results/qwen25_port_20260928T054537Z/PORT_REPORT.md),
[수치 분기 조사](results/qwen25_correctness_debug_20260928T070425Z/REPORT.md),
[v2 재검증 결과](results/qwen25_correctness_v2_20260928T081111Z/REPORT.md)를 참조한다.

## 13. LLaVA Ours-Chunk25 → Ours-Visual-KV25 (2026-09-29)

기존 Chunk25는 64-token chunk 개수의 25%를 선택했고, 새 Visual-KV25는
실제 visual-content KV 행 N개 중 ceil(0.25×N)개만 attention에 남긴다.
경계 chunk의 추가 행은 읽을 수 있지만 mask로 제외한다. GQA 고정 40 images ×
6 questions의 같은 실행에서 네 arm을 비교했고 GPU correctness와 독립 감사가
통과했다. 아래 수치는 T2–T6 cache hit 평균이다.

| Method | Hit 정답률 | Hit TTFT | Normal / total read MB/hit | 실제 content retention |
|---|---:|---:|---:|---:|
| ReComp | 63.0% | 512.19 ms | 0 / 0 | — |
| FullLoad | 63.0% | 691.73 ms | 1165.073 / 1165.073 | 100% |
| Ours-Chunk25-Legacy | 57.5% | 256.59 ms | 286.052 / 306.079 | 24.99% |
| Ours-KV25-New | 56.5% | 257.51 ms | 299.473 / 319.501 | 25.00% |

KV25−legacy의 paired hit 정답률 차이는 −1.0 pp (95% CI −3.5~+1.0),
TTFT 차이는 +0.92 ms (95% CI −3.60~+5.54)였다. 별도의 MT-GQA
4-image smoke는 full 4,061-dialogue 실험이 아니다.
[예산 계약](docs/llava_kv25_budget_contract.md),
[파일럿 보고서](results/llava_kv25_migration_20260929T032642Z/REPORT.md)와
[재현·감사 자료](results/llava_kv25_migration_20260929T032642Z/REPRODUCE.md)에 상세 결과가 있다.

## 14. Qwen Ours-Chunk25 → Ours-Visual-KV25 (2026-09-29)

동일한 Qwen checkpoint·NF4/BF16/SDPA에서 64-row Chunk25와 정확한
visual-content KV 25% 선택을 별개 arm으로 실행했다. v2 기반 GPU correctness
gate와 독립 감사가 PASS했다. GQA는 40 images × 6 questions, MT는
40 dialogues × 3 generated-history turns이며 모두 4-arm 파일럿이다.

| Dataset / method | Hit 정답률 | Hit TTFT | Normal / total read MB/hit | Content retention |
|---|---:|---:|---:|---:|
| GQA / Chunk25 legacy | 60.00% | 55.92 ms | 6.423 / 7.627 | 31.73% |
| GQA / Visual-KV25 | 57.00% | 55.65 ms | 7.340 / 8.544 | 25.12% |
| MT / Chunk25 legacy | 66.25% | 59.73 ms | 7.065 / 8.269 | 33.40% |
| MT / Visual-KV25 | 67.50% | 59.02 ms | 7.524 / 8.728 | 25.14% |

GQA에서 KV25−legacy hit 정답률 차이는 −3.0 pp (95% CI −6.5~0.0),
MT에서는 +1.25 pp (−2.5~+5.0)다. 두 TTFT 차이의 CI에도 0이 포함되어
동등성이나 속도 개선을 확정할 수 없다. Qwen의 persistence·cold-start
session 비용은 이번 전환에서 재측정하지 않았고, 더 큰 main run에는
저장 공간 계획이 필요하다. [예산 계약](docs/qwen25_kv25_budget_contract.md),
[파일럿 결과](results/qwen25_kv25_migration_20260929T042949Z/REPORT.md),
[타이밍 정정](docs/qwen25_kv25_pilot_timing_erratum.md)을 참조한다.
대용량 run-local store와 원시 요청 로그는 로컬에 보존하고, 여기에는
계약·요약표·검증·감사 자료를 게시한다.

## 15. KV25 selector 보조 실험 (2026-09-29)

LLaVA의 기존 40-image GQA 개발 workload에서 25% visual KV 예산을 고정하고
Contextual Representative 및 SpatialUniform 선택을 시험했다. Contextual
D20+C5의 hit 정답률은 D25보다 +1.0 pp (image-paired 95% CI −3.5~+5.5),
SpatialUniform5는 IndexUniform5보다 −4.0 pp (−9.5~+1.0)였다.
두 차이 모두 개선 또는 동등성을 확정하지 못하며 독립 holdout도 아니다.
[Contextual 보고서](results/llava_contextual_kv25_20260929T074144Z/REPORT.md)와
[Spatial 보고서](results/llava_spatial_kv25_20260929T091132Z/REPORT.md)에
검증·선택 차이·SSD I/O와 paired 비교가 있다.

## 16. SparseVLM-SSD-KV25 AllHead baseline (2026-09-30)

원 SparseVLM의 attention 기반 선택을 SSD Visual-KV25에 맞춘 baseline이다.
Probe3와 32-head AllHead 모두 구현·검증했으며, progressive pruning
전체 시스템의 재현은 아니다. 40-image GQA 파일럿에서 AllHead의 hit
정답률/TTFT/SSD read는 62.0% / 901.50 ms / 1174.14 MB였고,
같은 실행의 Ours-KV25는 56.5% / 309.93 ms / 319.50 MB였다.
40-dialogue MT 파일럿의 AllHead는 78.75% / 868.66 ms / 1204.07 MB,
Ours-KV25는 76.25% / 301.02 ms / 327.58 MB였다.
AllHead는 선택 KV가 25%여도 score 계산에 full K를 읽으므로 SSD read가
25%가 되지 않는다. [검토 보고서](results/sparsevlm_ssd_kv25_20260930T041926Z/REPORT_REVIEWED.md)와
[계약](docs/sparsevlm_ssd_kv25_contract.md)을 참조한다.

## 17. LLaVA MT-GQA generated-history AllHead/KV25 main (2026-09-30)

재구성한 MT-GQA 4,061개 3-turn dialogue(398 images)의 5-arm 본실험을
60,915/60,915 final requests로 완료하고 독립 감사를 통과했다.
각 방법은 자신의 생성 답변을 다음 turn의 history로 사용한다.
ReKV는 이번 신규 비교에서 제외했다. 아래 hit 값은 T2–T3 평균이다.

| Method | 전체 정답률 | Hit 정답률 | Hit TTFT | SSD MB/hit |
|---|---:|---:|---:|---:|
| ReComp | 66.72% | 68.48% | 533.56 ms | 0.00 |
| FullLoad | 66.65% | 68.38% | 749.21 ms | 1173.92 |
| MPIC-32 | 66.77% | 68.55% | 766.71 ms | 1174.18 |
| SparseVLM-SSD-KV25-AllHead | 66.58% | 68.28% | 852.15 ms | 1184.12 |
| Ours-KV25 | 65.18% | 66.18% | 269.52 ms | 320.45 |

Ours−AllHead의 hit 정답률 차이는 −2.105 pp (95% CI −2.868~−1.352),
TTFT 차이는 −582.63 ms (−592.71~−572.45)였다. AllHead는 full K read를
scoring과 answer attention에 재사용해 중복 K read를 세지 않았다.
결과는 이 SSD adaptation과 재구성 workload에 한정된다.
[검토된 본실험 보고서](results/llava_mt_gqa_allhead_kv25_main_20260930T093251Z/analysis_v2/REPORT.md),
[요약표](results/llava_mt_gqa_allhead_kv25_main_20260930T093251Z/analysis_v2/summary.csv),
[실험 계약](docs/llava_mt_gqa_allhead_kv25_main_contract.md)에 상세 자료가 있다.

## 18. Qwen MT-GQA main 준비 상태 (2026-10-01)

Qwen2.5-VL generated-history main은 필수 FP32 통합 gate G7 실패로 중단됐다.
Smoke 0/36, main 0/36,549이며 Qwen 본실험 정확도·TTFT 표는 없다.
기존 KV25 core GPU 검증 PASS를 이 main의 PASS로 확대하지 않는다.
[중단 보고서](results/qwen_mt_gqa_kv25_main_20261001T031459Z/validation_blocked/REPORT.md)와
[계약](docs/qwen_mt_gqa_kv25_main_contract.md)에 재현 근거가 있다.

## 19. Prefix25 persistence ablation (2026-10-01)

FullStore-KV25 [A] 대신 실제 선택 content KV만 기록하는 PrefixStore-KV25
[B]를 LLaVA와 Qwen에 적용했다. 두 모델에서 A/B 선택 KV와 생성열이 일치했고,
각각 GQA 960요청·MT 480요청을 새로 측정했다.

| 모델 | MT write 감소 | MT persistence 감소 | MT 3-turn 세션 B−A | MT 세션 B−ReComp |
|---|---:|---:|---:|---:|
| LLaVA | 74.02% | 65.83% | −1906.8 ms | +317.9 ms |
| Qwen | 73.73% | 32.94% | −53.9 ms | −41.4 ms |

짧은 LLaVA 세션에서는 B도 ReComp보다 느렸고, cache hit-rate·eviction 이득은
측정하지 않았다. [전체 결과](results/prefix25_persistence_20261001T050113Z/REPORT.md)와
[재현 정보](results/prefix25_persistence_20261001T050113Z/REPRODUCE.md)를 참조한다.
수 GB 규모 run-local store와 원시 요청 로그는 로컬에 보존하고,
레포에는 계약·요약·검증·감사 자료만 게시한다.

## 20. LLaVA MT-GQA PrefixStore-KV25 B main (2026-10-02)

기존 FullStore Ours 대신 선택한 visual-content KV만 직렬화하는
PrefixStore-KV25 [B]를 새 5-arm 본실험에 넣었다. 재구성 MT-GQA 398 images,
4,061개 3-turn dialogue에서 60,915/60,915요청을 실행하고 독립 감사를 통과했다.
각 방법은 자신이 생성한 답변을 다음 turn의 history로 사용한다. ReKV·Probe3는
이 비교에 포함하지 않았다. Followup은 T2–T3 요청의 평균이다.

| Method | 전체 정답률 | Followup 정답률 | Followup TTFT | SSD MB/followup |
|---|---:|---:|---:|---:|
| ReComp | 66.716% | 68.481% | 533.24 ms | 0.00 |
| FullLoad | 66.650% | 68.382% | 781.71 ms | 1173.92 |
| MPIC-32 | 66.765% | 68.555% | 793.12 ms | 1174.18 |
| SparseVLM-SSD-KV25-AllHead | 66.585% | 68.284% | 877.51 ms | 1184.12 |
| PrefixStore-KV25 [B] | 65.181% | 66.178% | 262.58 ms | 308.05 |

B의 followup 정답률은 ReComp보다 2.302 pp 낮았다(이미지 단위 paired
95% CI −3.052~−1.540). B의 평균 store 크기는 이미지당 309.676 MB로
FullLoad의 1264.667 MB보다 작다. 실제 공유-store stream에 귀속한
3-turn session은 B가 ReComp보다 458.8 ms 짧았지만, 이미지별 setup을
매 dialogue에 다시 부과한 standalone-equivalent 환산에서는 300.7 ms
길었다. 후자는 독립 cold-new-image session 실측이 아니다.

같은 실험의 Qwen 3-arm은 필수 FP32 통합 gate 실패로 smoke와 main을
실행하지 않았다(0/36,549). Qwen 본실험 성능 수치는 없다.
[최종 보고서](results/mt_gqa_B_llava5_qwen3_20261001T093921Z/REPORT.md),
[품질표](results/mt_gqa_B_llava5_qwen3_20261001T093921Z/final/llava/quality.csv),
[지연·I/O 표](results/mt_gqa_B_llava5_qwen3_20261001T093921Z/final/llava/latency.csv),
[계약](docs/mt_gqa_B_main_contract.md)을 참조한다.
대용량 실행 store와 요청별 진단 CSV는 로컬에 보존한다.

## 21. Qwen MT-GQA PrefixStore-KV25 B main (2026-10-02)

앞선 Qwen 중단 실행과 별개로, Qwen2.5-VL의 새 3-arm 본실험을 완료했다.
재구성 MT-GQA 398 images, 4,061개 3-turn dialogue에서
36,549/36,549 final requests를 실행하고 독립 감사를 통과했다.
ReComp, FullLoad, PrefixStore-KV25 [B] 각각 자신의 생성 답변을 다음
turn의 history로 사용한다. Followup은 T2–T3이며 정답률은 strict
normalized exact match다. 기존 LLaVA 실험은 다시 실행하지 않았다.

| Method | T1 정답률 | T2 정답률 | T3 정답률 | 전체 정답률 | Followup 정답률 | Followup TTFT 평균/p95 | SSD MB/followup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 60.749% | 61.660% | 61.069% | 61.159% | 61.364% | 113.279 / 126.619 ms | 0.000 |
| FullLoad | 60.749% | 61.709% | 61.093% | 61.184% | 61.401% | 71.777 / 79.072 ms | 24.407 |
| PrefixStore-KV25 [B] | 60.749% | 57.892% | 57.843% | 58.828% | 57.868% | 54.010 / 56.384 ms | 6.371 |

B의 followup 평균 TTFT는 ReComp 대비 52.32%, FullLoad 대비 24.75%
짧았지만, followup 정답률은 각각 3.497 pp와 3.534 pp 낮았다.
이미지 단위 paired bootstrap 95% CI는 각각 −4.260~−2.743 pp,
−4.302~−2.784 pp다. 생성 history 차이까지 포함한 방법 수준 비교이며,
정답률이 동등하다는 결과가 아니다. ReComp의 KV read 0은 이미지 파일
I/O까지 0이라는 뜻이 아니고, DONTNEED는 cold NAND를 보장하지 않는다.

| Method | 이미지당 store MB | Persistence 평균 | Activation 평균 | 공유-stream 3-turn session | Standalone-equivalent session |
|---|---:|---:|---:|---:|---:|
| ReComp | — | — | — | 399.590 ms | 399.590 ms |
| FullLoad | 24.342 | 64.294 ms | 14.151 ms | 324.537 ms | 395.419 ms |
| PrefixStore-KV25 [B] | 6.400 | 45.969 ms | 5.115 ms | 285.942 ms | 332.083 ms |

B는 content KV를 이미지별 ceil(N/4)개 보존하며, pooled 보존율은
25.127%다. 구조 토큰은 유지하고 content padding은 기록하지 않는다.
Persistence와 session은 기록된 outer writer-return endpoint 기준이다.
기존 final/의 inner durable-publication 분석은 그대로 보존했다.
공유-stream은 실제 이미지별 setup을 한 번만 귀속한다. Standalone-equivalent는
같은 setup을 매 dialogue에 재부과한 환산값으로, 독립 cold-session 실측이
아니며 비-source capture 추가 비용은 NOT_MEASURED다.

기존 G7 FP32 dense/compact 1e-5 비교의 FAIL은 유지했다. 고정 입력에 대한
연산 shape별 FP32 반올림 차이를 진단하고, 새로운 사전 고정 검증에서
동일-shape 독립 참조와 실제 BF16 KV·위치·mask·logits·생성 ID 일치를 확인했다.
별도의 all-layer FP64 attention 참조는 1e-12 기준을 통과했다.
본실험 추론은 기존 BF16/SDPA 그대로이며 FP64는 검증에만 사용했다.
통합·추가 8-dialogue·동일 이미지·negative fixture·smoke 검증도 통과했다.

[최종 보고서](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/REPORT.md),
[품질표](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/final/quality.csv),
[지연·I/O 표](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/final/latency.csv),
[outer-endpoint session 표](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/reporting_details/session_summary.csv),
[G7 진단](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/G7_REVIEW.md),
[실험 계약](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/CONTRACT.md),
[재현 정보](results/qwen_mt_gqa_prefixstore3_20261002T074856Z/REPRODUCE.md)를 참조한다.
대용량 raw·tensor·store와 요청별 상세 CSV는 로컬에 보존한다.

## 22. GLM-4.6V-Flash MT-GQA PrefixStore-KV25 B main (2026-10-07)

최종 v6 실행은 재구성 MT-GQA 398 images, 4,061개 3-turn dialogue에서
3-arm 36,549/36,549요청을 완료했다. GQA 720요청·MT 360요청 파일럿도 완료했다.
모든 dialogue T1은 full pixels이며, 다음 turn에는 각 방법의 실제 생성 body를
native chat template로 전달한다. Prediction은 native answer-box span을 추출한
strict normalized EM이다. 기존 LLaVA/Qwen main은 재실행하지 않았다.

| Method | 전체 EM | Followup EM | Followup TTFT 평균/p95 | KV MB/followup | Store MB/image | 공유-stream session | Derived standalone session |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 61.216% | 61.635% | 126.523 / 149.011 ms | 0.000 | — | 641.398 ms | 641.398 ms |
| FullLoad | 61.159% | 61.549% | 73.477 / 76.761 ms | 15.180 | 15.170 | 543.477 ms | 614.079 ms |
| PrefixStore-KV25 [B] | 58.852% | 58.089% | 60.578 / 63.024 ms | 4.397 | 4.420 | 513.202 ms | 569.204 ms |

Followup은 T2–T3 요청 가중 평균, store는 이미지 평균, MB=10^6 bytes다.
B의 평균 TTFT는 ReComp 대비 52.12%, FullLoad 대비 17.56% 감소했지만,
followup EM은 각각 3.546 pp와 3.460 pp 낮았다. 이미지 단위 paired bootstrap
95% CI는 −4.416~−2.692 pp와 −4.328~−2.597 pp로 정확도 동등성을 뜻하지 않는다.
Native KV는 40 layers × 2 KV heads × 128 dim의 BF16, 40 KiB/token이며
B는 ceil(original N/4) content와 구조 19행을 보존한다.

세 arm은 RTX4090, decoder NF4/BF16, SDPA, batch1, seed1234,
enable_thinking=False, cap32로 고정했다. GLM cap32는 이전 LLaVA/Qwen cap16과
다르므로 모델 간 latency를 같은 조건의 paired 비교로 해석하지 않는다.
TTFT는 첫 native token까지이며 first lexical answer latency는 NOT_MEASURED다.
Cap/malformed R3/F3/B2건도 품질 분모에 포함했고 empty/thinking은 0건이었다.

Session은 identity 준비부터 wrapper 반환까지 caller outer setup을 포함한다.
공유-stream은 실제 이미지별 setup을 한 번 귀속하고, standalone은 같은 setup을
매 dialogue에 재부과한 DERIVED 값이다. 독립 cold-session 실측이 아니며
비-source capture 증분은 NOT_MEASURED다. OS pread와 DONTNEED는 cold NAND를
보장하지 않는다. 이전 v3/v4 실패·v5 qualified setup 결과는 별도 보존했다.
Native GPU 6사례·FP64 attention 480비교와 독립 796-store 감사를 통과했다.

[최종 보고서](results/glm46v_prefixstore3_v6_20261007T104505Z/REPORT.md),
[품질표](results/glm46v_prefixstore3_v6_20261007T104505Z/final/quality.csv),
[지연표](results/glm46v_prefixstore3_v6_20261007T104505Z/final/latency.csv),
[독립 검토](results/glm46v_prefixstore3_v6_20261007T104505Z/FINAL_ANALYSIS_REVIEW.md),
[재현 정보](results/glm46v_prefixstore3_v6_20261007T104505Z/REPRODUCE.md)에 상세 자료가 있다.
[세 모델 요약표](results/mt_gqa_three_models_summary_20261012/summary.csv)와
[저장 prediction의 raw-EM 사후 재채점](results/mt_gqa_raw_em_20261013/final/RAW_EM_REPORT.md)도
별도로 보존했다. Raw EM은 정규화하지 않은 저장 문자열 비교이며 위 normalized EM을
대체하지 않는다. 대용량 raw·store·weights와 요청별 상세 CSV는 로컬에 보존한다.

## 23. MPIC 다중 이미지 검증·MMDU GPU 파일럿 (2026-10-13)

새 LLaVA 다중 이미지 경로에서 모든 text와 선두 image token만 연산하고,
나머지 full image KV를 유지하는 selective prefill을 구현·검증했다.
원래 interleaving, logical position/1-D RoPE, causal mask, attention 전 K/V 교체를
검사했다. Global k32와 per-image k32를 구분하며 두 번째 이미지는 각각 0개와
32개가 active다. 공식 multi-image k 규칙은 NOT VERIFIED이므로 두 방법은
각각 MPIC-32 (adapted, global k), MPIC-32 (adapted, per-image k)로 표기한다.

고정 MMDU 5대화의 첫 3턴을 4방법·SSD/GPU 상주 조건에서 각각 60요청,
총 120요청 실행했다. 방법마다 자기 generated history만 사용했다.
생성 실패·OOM·retry·fallback은 0건이며 두 조건의 출력·입력·history는
60/60 일치했다. 아래는 방법당 15요청 평균이고 MB=10^6 bytes다.

| Method | SSD TTFT | GPU 상주 TTFT | SSD KV MB/request | 공통 14턴 judge overall /10 |
|---|---:|---:|---:|---:|
| Prefix caching | 233.14 ms | 233.19 ms | 0.00 | 6.86 |
| Full reuse | 738.77 ms | 107.21 ms | 1027.60 | 5.93 |
| MPIC-32 (adapted, global k) | 818.75 ms | 159.94 ms | 1027.87 | 6.07 |
| MPIC-32 (adapted, per-image k) | 831.40 ms | 165.26 ms | 1028.04 | 6.43 |

Full reuse는 two-pass image-KV linking adaptation으로 기존 fixed-prefix FullLoad와
다른 계약이다. Prefix caching은 exact prompt-prefix KV를 GPU에 유지하므로
캐시 residency가 같지 않다. MPIC은 chunk64 정렬 때문에 k32도 full range를 읽는다.
모델은 LLaVA-v1.6-Vicuna-7B NF4/BF16/eager, image KV는 FP16, greedy cap256이다.
25/60응답이 cap에 도달했고 진단 assertion overhead가 TTFT에 포함됐다.
고정 method 순서의 작은 파일럿이므로 전체 MMDU 성능·통계적 우열을 확정하지 않는다.

품질은 실제 로컬 Qwen2.5-VL visual judge 평가 60건 중 59건이 유효했다.
Full reuse의 1건 N/A를 FAIL로 유지하고, 위 점수는 네 방법 공통 14턴만 비교했다.
GPU 상주 출력이 같아 같은 평가를 연결했으며 judge를 두 번 실행한 결과가 아니다.
원 논문의 judge·공식 시스템 재현은 NOT VERIFIED다. Host mixed hit-read/miss
준비는 검증했지만 hit H2D/miss compute overlap은 미구현이며 OS DONTNEED도
cold NAND를 보장하지 않는다. 전체 MMDU나 MT-GQA main을 새로 실행하지 않았다.

기존 FullLoad의 suffix1 초기 prefill을 decode로 오인하던 guard를 request-local
완료 상태로 수정하고 prefix/cache ID/suffix 일치를 I/O 전에 검사하도록 했다.
GPU suffix1/2·A/B/A/B cache isolation을 통과했고, 과거 미검증 36요청을
원본 prompt/suffix hash로 복원해 해당 suffix1 trigger가 없음을 확인했다.
이는 과거 모든 품질·position 계약을 재인증하는 결과가 아니다.

[GPU 파일럿 보고서](results/mpic_gpu_mmdu_20261013_01/REPORT.md),
[최종 상태](results/mpic_gpu_mmdu_20261013_01/FINAL_STATUS.json),
[SSD 요약표](results/mpic_gpu_mmdu_20261013_01/ssd_final_verified/summary.csv),
[GPU 상주 요약표](results/mpic_gpu_mmdu_20261013_01/gpu_resident_verified/summary.csv),
[CPU 구현·수정 기록](results/mpic_multi_fix_20261013_01/REPORT.md),
[초기 독립 감사](audits/mpic_20261013_cpu_01/AUDIT_REPORT.md),
[재현 정보](results/mpic_gpu_mmdu_20261013_01/REPRODUCE.md)에 근거가 있다.
Bounded 요청·judge 결과와 감사 자료는 게시하고, KV payload·tensor·대용량 profiler는
로컬에 보존한다. 초기 CPU-only NOT VERIFIED 기록을 GPU PASS로 덮어쓰지 않았다.

## 24. MT-VQA-v2-reconstructed-25k 본실험 (2026-10-30)

VQAv2 validation 독립 질문을 이미지별 3턴으로 묶은 새 재구성 workload에서
25,000 images/dialogues, 방법당 75,000요청을 실행했다. 최종 v3는 LLaVA 5-arm
375,000, Qwen 3-arm 225,000, GLM 3-arm 225,000으로 총 825,000/825,000요청을
완료하고 모델별 독립 분석을 통과했다. Smoke·기존250 파일럿·새40 파일럿·gate는
main 분모에 합치지 않았다. 저자 subset/순서 또는 MetaCompress 공식 재현은 아니다.

Primary는 고정 GT-Vision-Lab/VQA evaluator의 ten-answer leave-one-out
VQA score (%)이며 이전 MT-GQA normalized EM과 다르다. 해당 revision의
conditional normalization도 유지했다. 각 방법은 자기 native 생성 history를 사용하고
GLM은 scoring answer-box span과 literal history body를 구분한다.

| 모델 | Method | 전체 VQA score | Followup VQA score | Followup TTFT | KV MB/followup | Setup 포함 3-turn session |
|---|---|---:|---:|---:|---:|---:|
| LLaVA | ReComp | 51.7799% | 51.7728% | 530.147 ms | 0.000 | 1679.793 ms |
| LLaVA | FullLoad | 51.7851% | 51.7806% | 755.791 ms | 1172.376 | 4541.380 ms |
| LLaVA | MPIC-32 SSD | 51.8072% | 51.8138% | 774.574 ms | 1172.638 | 4418.585 ms |
| LLaVA | SparseVLM-SSD-KV25-AllHead | 51.6455% | 51.5712% | 844.591 ms | 1183.515 | 4770.998 ms (DERIVED) |
| LLaVA | PrefixStore-KV25 [B] | 50.6715% | 50.1102% | 262.460 ms | 307.755 | 1953.998 ms |
| Qwen | ReComp | 75.1252% | 75.1720% | 112.812 ms | 0.000 | 404.507 ms |
| Qwen | FullLoad | 75.1005% | 75.1350% | 70.760 ms | 24.330 | 398.835 ms |
| Qwen | PrefixStore-KV25 [B] | 72.0063% | 70.4936% | 53.995 ms | 6.366 | 338.656 ms |
| GLM | ReComp | 49.6445% | 48.8068% | 126.383 ms | 0.000 | 655.577 ms |
| GLM | FullLoad | 49.6307% | 48.7860% | 75.665 ms | 15.160 | 636.403 ms |
| GLM | PrefixStore-KV25 [B] | 48.1372% | 46.5458% | 60.796 ms | 4.392 | 588.169 ms |

Followup은 T2–T3의 50,000요청/방법 평균이며 MB=10^6 bytes다.
이미지당 dialogue가 하나이므로 세 요청 E2E와 첫 persistence/activation을 모두 부과한다.
AllHead는 FullLoad와 공유한 build를 독립 배포에 재부과한 DERIVED session을 표시했다.
실제 공유-stream 귀속은 3008.427 ms이며 독립 cold-session 실측으로 해석하지 않는다.

B−ReComp의 followup VQA score 차이는 LLaVA −1.6626 pp
(paired image-cluster 95% CI −1.8456~−1.4816), Qwen −4.6784 pp
(−4.9140~−4.4328), GLM −2.2610 pp (−2.4992~−2.0226)였다.
LLaVA는 TTFT가 줄어도 setup 포함 session은 ReComp보다 274.205 ms 길었다.
Qwen·GLM은 각각 65.851 ms·67.408 ms 짧았지만 점수 손실을 동반한다.
효과는 각 모델 안에서 비교하며 모델 간 결과를 합치거나 정확도 동등성을 주장하지 않는다.

Seed1234/batch1/greedy, 기존 모델별 NF4·native KV/backend/해상도를 유지했다.
LLaVA/Qwen cap16, GLM cap32이며 empty·EOS-only·cap 출력도 분모에 포함했다.
기존250 개발 데이터와 이미지 166개·질문 199개가 겹쳐 unseen holdout은 아니다.
ReComp KV read=0은 이미지 I/O=0이 아니고 OS 반환 bytes/DONTNEED는 cold NAND를
보장하지 않는다. 초기 v1/v2 실패와 legacy FP32 FAIL은 별도 보존했다.

[완료 상태](results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/MAIN_REPORT.md),
[LLaVA main](results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/llava/main/REPORT.md),
[Qwen main](results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/qwen/main/REPORT.md),
[GLM main](results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/glm/main/REPORT.md),
[scorer 검증](results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/SCORER_VALIDATION.md),
[실험 계약](docs/mt_vqa_v2_reconstructed_25k_contract.md),
[재현 정보](results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/REPRODUCE.md)를 참조한다.
코드·요약표·검증/완료 receipt·해시와 provenance를 게시하며, 원본 이미지·전체 manifest·
raw journal·KV payload와 요청별 대용량 CSV는 로컬에 보존한다.

## 25. Full-context KV25 vs reduced-token Recompute KV25 ablation (2026-11-02)

LLaVA GQA 40 images × 6턴(hit 200/arm)에서 Ours-KV25 [B]와, 같은 S·k·저장 순서·bytes로
system+S+separator만 별도 prefill해 KV를 다시 계산한 Recomp-KV25를 비교했다.
Orig는 원래 position ID, Compact는 연속 position을 쓰고, PosShift는 Ours store의
text 시작만 `v0+k+n_sep`로 옮긴 진단 arm이다. 모든 gate(G1–G7)를 통과했다.
G2는 rev. 3이며, rev.-2 run이 n51002에서 shape-rounding 때문에 중단된 뒤 품질 결과를
열람하기 전에 개정했다. 구조 행은 same-length truncation 대조와 bit-identity로 검사한다.

| Arm | Hit acc. | Ours−arm (95% CI) | Hit TTFT mean | Prefill-only Δ/image |
|---|---:|---:|---:|---:|
| FullLoad | 63.00% | — | 691.16 ms | — |
| Ours-KV25 | 56.50% | — | 265.71 ms | 0 |
| Recomp-KV25-Orig | 56.00% | +0.50 pp (−2.50~+3.50) | 262.17 ms | 87.81 ms |
| Recomp-KV25-Compact | 60.50% | −4.00 pp (−8.00~+0.00) | 266.27 ms | 87.79 ms |
| Ours-KV25-PosShift | 0.00% | +56.50 pp (+48.50~+64.00) | 265.85 ms | 0 |

사전 규칙상 Ours−Orig는 차이가 없음을 보여 주지 못했다(동등성 아님).
Ours−Compact도 CI 상한이 정확히 0이라 "차이 없음 표시"로 분류하지만 경계 사례다.
Orig−Compact는 −4.50 pp(−8.00~−0.50)로 Compact가 높았다.
PosShift 붕괴는 Ours store에 원래 text position이 필요함을 보여 준다.
BF16 shape rounding이 context가 같은 행에도 deep-layer drift를 만든다(single row 최대 0.032 rel L2, context 효과 0.21/0.40).
그래서 1 pp 미만 차이는 해석하지 않는다. 40 cluster·개발 세트·LLaVA 한정이며 Stage 2와 Qwen은 실행하지 않았다.
[전체 결과](results/recompute_kv25_ablation_20261102T061243Z/REPORT.md),
[실험 계약](docs/recompute_kv25_ablation_contract.md),
[재현 정보](results/recompute_kv25_ablation_20261102T061243Z/REPRODUCE.md)를 참조한다.

## 26. Ours-ReRoPE와 Pruned-T1 (2026-11-02)

25절의 2×2(문맥 × 위치)를 한 run에서 완성했다. Ours-ReRoPE는 전체 문맥 Ours KV의
S·sep 행 Key만 persistence 시점에 연속 위치로 한 번 재회전(FP64, FP16 저장)한 store이고,
Pruned-T1은 캐시 없이 sys + (S ∪ sep) + 질문을 연속 위치로 한 번 prefill하는 첫 턴이다.
GQA 40 images × 6턴, 7 arm, 모든 gate(G1–G7, G-a~G-d) 통과. rev.3 arm 재현율 100%.
첫 두 run은 측정 전에 중단됐고(스크립트 결함, 디스크 floor) 기록으로 보존했다.

| Context \ Position | 원래 위치 | 연속 위치 |
|---|---:|---:|
| 전체 T1 문맥 | 56.50% (Ours-KV25) | 57.00% (Ours-ReRoPE) |
| 축소 prefill 문맥 | 56.00% (Recomp-Orig) | 60.50% (Recomp-Compact) |

ReRoPE−Ours +0.50 pp (−4.50~+5.50), ReRoPE−Compact −3.50 pp (−8.00~+1.00)로 차이 미입증.
공동 bootstrap 요인 효과: 문맥 −1.50 pp (−4.25~+1.50), 위치 +2.50 pp (−1.00~+6.00),
교호작용 +4.00 pp (−1.00~+9.00). 모두 CI가 0을 포함한다.
Pruned-T1 vs full-token 단독 추론: T1 −5.00 pp (−15.00~+5.00), T2–T6 −1.50 pp (−6.50~+3.00),
전체 −2.08 pp (−6.25~+2.08). 대화 단위: [T1 full + Ours] 57.08%, [T1 full + ReRoPE] 57.50%,
[전부 Pruned] 60.42%, 차이는 모두 미입증. ReRoPE 재회전 비용은 이미지당 7.31 ms
(Recomp 축소 prefill 87.7 ms). Pruned-T1 TTFT는 compose 측정이라 효율 비교에 쓰지 않는다.
개발 세트 반복 사용으로 모든 결론은 탐색적이며 1 pp 미만 차이는 해석하지 않는다.
[전체 결과](results/rerope_kv25_20261102T093759Z/REPORT.md),
[실험 계약](docs/rerope_kv25_contract.md),
[재현 정보](results/rerope_kv25_20261102T093759Z/REPRODUCE.md)를 참조한다.

## 27. VZ-Pruned-E2E vs Ours-KV25 vs ReComp, 전체 MT-GQA (2026-11-03)

26절의 Pruned-T1을 실제 serving 경로로 올려(vision tower 1회 → image-only saliency →
[sys | keep | question] 연속 위치 1회 prefill, 그 prefix KV를 store에 저장, T2–T3는 Compact
규칙으로 hit) B main과 같은 재구성 MT-GQA 398 images / 4,061 dialogues × 3턴에서
ReComp, Ours-KV25(B 경로)와 비교했다. 36,549 요청 모두 완료, 모든 gate(G1–G8) 통과.
G2: 구조 행은 전 layer bit-identical, layer-0 K 재정렬 rel L2 최대 4.6e-3(상한 1e-2),
oracle 비 최대 1.07(상한 1.5), 음성 대조 최소 1.10(> 0.1). G6: ReComp·Ours 각각 12,183 요청이
B main과 토큰 단위로 100% 일치. G8: 20 images × T2,T3 hit과 cache-less 단일 prefill이
40/40 토큰 동일. 앞선 두 freeze run은 측정 전 중단(audit 동점 규칙, 용량 guard)이라
기록으로 보존했다.

| Arm | T1 EM | Hit EM (T2–T3) | All EM | T1 TTFT (ms) | Hit TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| ReComp | 63.19 | 68.48 | 66.72 | 522.6 | 529.3 |
| Ours-KV25 | 63.19 | 66.18 | 65.18 | 522.7 | 246.4 |
| VZ-Pruned-E2E | 60.99 | 66.58 | 64.72 | 159.6 | 243.7 |

사전 지정 primary endpoint(Ours − VZ): T1 +2.19 pp [+1.10, +3.26] (차이 있음),
all +0.46 pp [−0.23, +1.15] (미입증). Secondary(탐색적): Ours − VZ hit −0.41 pp
[−1.19, +0.37] 미입증, Ours·VZ 모두 ReComp 대비 hit/all에서 −1.5~−2.3 pp 열위(CI 0 제외).
VZ T1 TTFT는 Ours 대비 −363 ms [−369, −357] (LM prefill 627 vs 2,242 토큰), hit TTFT 차이
+2.7 ms(Ours − VZ). 대화 단위 shared-stream session: ReComp 1,666 / Ours 1,172 / VZ 804 ms.
Store 용량은 두 arm 동일(309.7 MB/image), VZ persistence +10.9 ms/image. 같은 재구성
MT-GQA를 재사용했으므로 VZ 비교는 사전 등록된 탐색적 비교이며 1 pp 미만 차이는
해석하지 않는다.
[전체 결과](results/vz_pruned_e2e_mtgqa_20261102T120205Z/REPORT.md),
[실험 계약](docs/vz_pruned_e2e_mtgqa_contract.md),
[재현 정보](results/vz_pruned_e2e_mtgqa_20261102T120205Z/REPRODUCE.md)를 참조한다.

## 28. PosShift 용량-반응 진단 (2026-11-03)

25절 PosShift의 hit EM 0%(0/200)가 구현 결함인지 확인하기 위해 같은 40 GQA images × T2–T6에서
Ours store(전체 T1 문맥, 원래 위치 Key)를 다시 만들고 text 시작 위치만 `(v0+V) − δ`로 옮겼다.
δ=0은 기준 경로와 모든 step logits가 bit-identical(200/200), 기준 경로는 25절 Ours 행을 토큰 단위로
재현(200/200), δmax는 25절 PosShift 행을 재현(200/200)했다. 모든 gate 통과.

| δ | 0 | 1 | 4 | 16 | 64 | 256 | 1024 | max (1080–2160) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hit EM % | 56.5 | 58.0 | 57.5 | 58.5 | 57.0 | 15.5 | 0.0 | 0.0 |
| 첫 토큰 = 기준 % | 100 | 95.0 | 89.0 | 89.5 | 88.5 | 14.0 | 0.0 | 0.0 |
| 16-token cap % | 0 | 0 | 0 | 0 | 0 | 32.5 | 100 | 100 |

사전 고정 판독: δ\*(EM ≤ 5%) = 1024 → **점진**. 64 position까지는 EM 차이가 CI 안에 있고(+0.5~+2.0 pp,
모두 0 포함), 256에서 −41.0 pp [−50.5, −32.0], 1024 이상에서 0%와 형식 붕괴(개행·기호·기능어 반복)가
나타난다. 25절의 0%는 구현 결함이 아니라 위치 불일치 크기의 결과로 읽으며, 25절 사전 판정은 그대로다.
진단 목적의 탐색적 곡선이고 attention 분포는 측정하지 않았다.
[전체 결과](results/posshift_dose_20261103T060151Z/REPORT.md),
[실험 계약](docs/posshift_dose_contract.md),
[재현 정보](results/posshift_dose_20261103T060151Z/REPRODUCE.md)를 참조한다.

## 29. Layer-wise KV25 budget allocation (2026-11-06)

전체 content KV 예산은 그대로 두고 layer별 보존 행 수만 다르게 배분했다.
32-layer LLaVA의 고정 ratio vector와 Hamilton 배분으로 이미지마다
Σ k_l = 32 × ceil(original N/4)를 만족한다. 기존 image-only saliency 순서,
구조 행·원래 position·full T1 capture는 유지하고 layer별 파일은 정확히 k_l행에서
끝난다. Uniform vector는 기존 PrefixStore-KV25와 payload·mask·생성 token이 같다.

CAL은 VQAv2 250 images/1,250질문, DEV는 GQA 40 images/240질문이다.
CAL sensitivity로 7개 vector를 만들고 DEV에서 A1-2를 선택한 뒤,
MT-GQA-reconstructed 398 images/4,061 dialogues의 4-arm HOLD를
48,732/48,732요청으로 완료했다. 독립 감사 PASS이며 실패·skip·resume는 0이다.
DEV의 40이미지는 모두 HOLD에 포함되므로 독립 unseen holdout이라고 주장하지 않는다.

| Method | 전체 EM | Followup EM | Followup TTFT | SSD MB/followup | Shared-stream session | Standalone-equivalent session |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 66.716% | 68.481% | 529.27 ms | 0.000 | 1665.8 ms | 1665.8 ms |
| FullLoad | 66.650% | 68.382% | 736.50 ms | 1173.919 | 2362.0 ms | 4956.6 ms |
| B-uniform | 65.181% | 66.178% | 248.94 ms | 308.050 | 1183.3 ms | 1886.8 ms |
| B-A1-2 | 66.051% | 67.483% | 279.80 ms | 308.050 | 1243.2 ms | 1933.8 ms |

Followup은 T2–T3, strict normalized EM이며 MB=10^6 bytes다. B-A1-2는
B-uniform보다 +1.305 pp(이미지 단위 paired bootstrap 95% CI +0.560~+2.046)
높았지만 ReComp보다 −0.997 pp(−1.663~−0.341) 낮았다.
SSD bytes는 동일해 ≤1.02× 조건을 통과했지만 TTFT는 +30.855 ms로
≤1.05× 조건을 충족하지 못했다. 따라서 실행은 완료됐어도 사전 판정은 **PARTIAL**이다.
파일 크기별 pread 진단도 보존했으며, 성능을 맞추려고 serving I/O 경로를 바꾸지 않았다.

Shared-stream은 실제 이미지 setup을 한 번 귀속하고 standalone은 이를 매 dialogue에
다시 부과한 DERIVED 값이다. 독립 cold-session 실측이 아니며 standalone에서
B-A1-2는 ReComp보다 268.003 ms 길었다. Dense GPU buffer는 유지하므로
GPU 메모리 75% 절감이나 cache saturation/hit-rate 이득을 주장하지 않는다.
Qwen·GLM은 실행하지 않았다. 초기 gate 실패·소스 freeze amendment·공용 코드 diff는
별도 보존하며, 이전 실험 수치를 이번 allocation의 결과로 재사용하지 않았다.
보고기의 gate/protection 표기 변경 전 frozen 사본도 보존했다.

[전체 결과](results/layer_budget_kv25_20261105T125754Z/REPORT.md),
[HOLD 품질표](results/layer_budget_kv25_20261105T125754Z/hold/quality.csv),
[HOLD 지연·I/O](results/layer_budget_kv25_20261105T125754Z/hold/latency_io.csv),
[최종 판정](results/layer_budget_kv25_20261105T125754Z/decision.json),
[실행 기록](results/layer_budget_kv25_20261105T125754Z/execution_record.json),
[실험 계약](docs/layer_budget_kv25_contract.md),
[재현 정보](results/layer_budget_kv25_20261105T125754Z/REPRODUCE.md)를 참조한다.
소스·요약표·그림·검증·재현 자료를 게시하고 대용량 raw·tensor·KV payload는 로컬에 보존한다.

## 30. Layer-wise KV25 v2: Packed PrefixStore + allocation baselines (2026-11-07)

v1의 layer별 파일 대신 K/V·separator를 이미지당 한 packed 파일에 직접 기록하고,
hit마다 pinned host buffer로 os.preadv 1회 읽도록 했다. Uniform과 layerwise 모두
같은 packed 경로를 쓰며 payload·mask·생성 결과의 unpacked 동등성을 검증했다.
기존 dense GPU buffer·scatter·원래 position·image-only saliency 순서·총 KV 예산은 유지했다.
Pinned buffer 준비는 activation에 포함하고 요청 TTFT 바깥에서 수행한다.

LLaVA MT-GQA-reconstructed 398 images/4,061 dialogues × 3턴의 7-arm HOLD를
85,281/85,281요청으로 완료했다. 독립 감사 PASS, 실패·skip·resume는 0이다.
A1-2는 v1에서 선택한 vector를 사전 고정했고, v2 결과를 보고 다시 선택하지 않았다.

| Method | 전체 EM | Followup EM | Followup TTFT | SSD MB/followup | Hit read syscalls |
|---|---:|---:|---:|---:|---|
| ReComp | 66.716% | 68.481% | 531.17 ms | 0.00 | 0 |
| FullLoad | 66.650% | 68.382% | 762.42 ms | 1173.92 | 64 pread |
| U_packed | 65.181% | 66.178% | 212.06 ms | 308.05 | 1 preadv |
| A1-2_packed | 66.051% | 67.483% | 215.62 ms | 308.05 | 1 preadv |
| VLC_offline_packed | 65.764% | 67.052% | 214.62 ms | 308.05 | 1 preadv |
| SA_offline_packed | 64.344% | 64.922% | 216.75 ms | 308.05 | 1 preadv |
| PYR_packed | 65.805% | 67.114% | 215.44 ms | 308.05 | 1 preadv |

Followup은 T2–T3 strict normalized EM이며 MB=10^6 bytes다. A1-2−U는
+1.305 pp(이미지 단위 paired bootstrap 95% CI +0.560~+2.046), bytes는 동일하고
TTFT 비율은 1.017×로 ≤1.05×를 통과했다. 따라서 v2 사전 판정은 **GO**다.
29절 v1의 PARTIAL을 바꾸는 것이 아니라 새 packed 실행의 판정이다.
v1과 공유한 48,732요청의 저장 prediction은 모두 같았으며, v1→v2 TTFT 비교는
cross-run이므로 같은 실행 안의 paired 효과나 CI로 해석하지 않는다.

VLC/SA는 VL-Cache·SqueezeAttention의 online 기준을 CAL 평균의 고정 vector로
옮긴 offline adaptation이다. PYR은 고정 PyramidKV형 schedule이며 원 시스템의
완전한 재현을 주장하지 않는다. A1-2−VLC +0.431 pp(CI −0.094~+0.965),
A1-2−PYR +0.369 pp(−0.203~+0.933)는 우월성·동등성 모두 미입증이다.
A1-2−SA는 +2.561 pp(+1.748~+3.382)지만 이 offline 조건에 한정한다.
부차 지표인 packed vector 간 TTFT ±2% 범위는 SA +2.2%로 충족하지 못했다.

Shared-stream session은 U/A1-2가 1118.4/1125.8 ms이지만, setup을 매 dialogue에
재부과한 standalone-equivalent는 1906.1/1919.3 ms로 ReComp 1670.3 ms보다 길다.
후자는 DERIVED이며 독립 cold-session 실측이 아니다. DEV 40이미지가 HOLD에
포함되므로 unseen holdout은 아니고, dense GPU buffer를 유지해 메모리 절감을
주장하지 않는다. OS read와 DONTNEED는 cold NAND를 보장하지 않는다.
SA-2g/PYR-beta/directcopy는 DEV 진단이며 HOLD arm에 포함하지 않았다.
Qwen·GLM 추가 실험은 없고, 실패 시도와 freeze amendment·보고기 frozen 사본을 보존했다.

[전체 결과](results/layer_budget_kv25_v2_20261106T073502Z/REPORT.md),
[HOLD 품질표](results/layer_budget_kv25_v2_20261106T073502Z/hold/quality.csv),
[HOLD 지연·I/O](results/layer_budget_kv25_v2_20261106T073502Z/hold/latency_io.csv),
[paired 비교](results/layer_budget_kv25_v2_20261106T073502Z/hold/paired_analysis.csv),
[최종 판정](results/layer_budget_kv25_v2_20261106T073502Z/decision.json),
[v1/v2 prediction 대조](results/layer_budget_kv25_v2_20261106T073502Z/v1_v2_prediction_equivalence.json),
[실험 계약](docs/layer_budget_kv25_v2_contract.md),
[재현 정보](results/layer_budget_kv25_v2_20261106T073502Z/REPRODUCE.md)를 참조한다.
코드·요약표·그림·검증·재현 자료를 게시하며 대용량 raw·KV payload·외부 논문 사본은 로컬에 보존한다.

## 31. LLaVA PyramidKV β=20: floor vs raw (2026-11-12 정리)

30절의 선형 PYR과 구분해, PyramidKV β=20 원식의 layer 비율을 사용한 raw와
하한 0.10을 적용한 PYR-beta를 같은 실행에서 비교했다. Image-only VisionZip 순서,
총 KV bytes와 packed 경로는 같고 layer별 자르는 지점만 다르다. 원 PyramidKV의
attention 기반 token 선택이나 전체 시스템을 재현한 것은 아니다.
MT-GQA 398 images/4,061 dialogues × 3턴, 3-arm 36,549요청을 완료하고 독립 감사를 통과했다.

| Method | 전체 EM | Followup EM | Followup TTFT | SSD MB/followup |
|---|---:|---:|---:|---:|
| U_packed | 65.181% | 66.178% | 230.34 ms | 308.05 |
| PYR-beta_packed | 66.010% | 67.422% | 226.64 ms | 308.05 |
| PYR-beta-raw_packed | 66.010% | 67.422% | 226.84 ms | 308.05 |

두 벡터 모두 uniform 대비 +1.244 pp이며 CI는 각각 +0.600~+1.896,
+0.624~+1.864 pp다. Bytes·TTFT 조건도 통과해 사전 판정은 둘 다 GO다.
Raw−floor는 0.000 pp(CI −0.314~+0.307)로 사전 분류값은 FLOOR_UNNECESSARY이고
기본안은 raw다. 이는 이 표본에서 하한 효과 미입증이라는 뜻이지 동등성 증명은 아니다.
U의 12,183요청이 부모 v2와 토큰 단위로 같음을 확인했지만, 실행 간 절대 TTFT는
비교하지 않는다. ReComp는 이번 실행에 포함하지 않았다.

[전체 결과](results/layer_budget_kv25_pyrbeta_20261110T073604Z/REPORT.md),
[최종 판정](results/layer_budget_kv25_pyrbeta_20261110T073604Z/decision.json),
[실험 계약](docs/layer_budget_kv25_pyrbeta_contract.md),
[재현 정보](results/layer_budget_kv25_pyrbeta_20261110T073604Z/REPRODUCE.md)를 참조한다.

## 32. Qwen layer-wise PyramidKV + PackedPrefixStore (2026-11-12)

Qwen2.5-VL의 28 layers/4 KV heads/BF16 native KV에 packed store와 layer별 compact
ragged cache를 적용했다. 원래 MRoPE/logical position을 유지하고 각 layer의 mask를
실제 cache 길이에 맞춘다. MT-GQA 398 images/4,061 dialogues의 6-arm 본실험을
73,098요청으로 완료했다. Gate·독립 감사·U_packed와 기존 B의 12,183요청 토큰 일치는
통과했지만 품질 개선 판정은 NO-GO다.

| Method | 전체 EM | Followup EM | Followup TTFT | SSD MB/followup |
|---|---:|---:|---:|---:|
| ReComp | 61.159% | 61.364% | 113.226 ms | 0.000 |
| FullLoad | 61.184% | 61.401% | 70.789 ms | 24.407 |
| Ours B (unpacked) | 58.828% | 57.868% | 53.875 ms | 6.371 |
| U_packed | 58.828% | 57.868% | 46.614 ms | 6.371 |
| PYR-b20_packed | 58.229% | 56.969% | 46.607 ms | 6.371 |
| PYR-b2p5_packed | 58.639% | 57.584% | 46.679 ms | 6.371 |

β=20−U는 −0.899 pp(CI −1.490~−0.302)로 같은 bytes의 uniform보다 낮았다.
β=2.5−U의 −0.283 pp(−0.738~+0.172)는 민감도 보고일 뿐 주 판정을 바꾸지 않는다.
Qwen의 N은 260–529이고 β=20의 마지막 layer는 3–7행만 남는 조건이다.
Packed hit는 preadv 1회이며 U−B TTFT −7.261 ms는 읽기 호출 감소뿐 아니라
CPU→GPU 조립 위치 변경 등을 포함한 새 hit 경로 전체의 효과다.
모든 followup은 방법별 자기 생성 history를 사용한 strict normalized EM이다.

[전체 결과](results/qwen_layer_budget_pyr_20261111T075124Z/REPORT.md),
[품질표](results/qwen_layer_budget_pyr_20261111T075124Z/quality.csv),
[최종 판정](results/qwen_layer_budget_pyr_20261111T075124Z/decision.json),
[ragged cache 설계](results/qwen_layer_budget_pyr_20261111T075124Z/design_ragged_cache.md),
[실험 계약](docs/qwen_layer_budget_pyr_contract.md),
[재현 정보](results/qwen_layer_budget_pyr_20261111T075124Z/REPRODUCE.md)를 참조한다.

## 33. Qwen 보존 비율 sweep: 25/30/40/50/100% (2026-11-12)

Layer 배분 대신 모든 layer에 같은 k=ceil(ratio×original N)를 적용하고, 같은
중요도 순서의 앞 k개와 구조 21행을 보존했다. MT-GQA 398 images/4,061 dialogues의
7-arm 본실험 85,281요청을 완료했다. 실패·skip 0, 독립 감사 PASS다.
GO/NO-GO 판정 없이 정확도·속도 곡선을 보고한다. 아래 행 수는 중간값 이미지 N=345 기준이다.

| Arm | Content rows/layer | Followup EM | Followup TTFT | SSD MB/followup |
|---|---:|---:|---:|---:|
| ReComp | — | 61.364% | 113.222 ms | 0.000 |
| FullLoad | 345 | 61.401% | 70.770 ms | 24.407 |
| U25_packed | 87 | 57.868% | 46.614 ms | 6.371 |
| U30_packed | 104 | 58.963% | 46.956 ms | 7.403 |
| U40_packed | 138 | 60.133% | 47.679 ms | 9.448 |
| U50_packed | 173 | 60.859% | 48.234 ms | 11.507 |
| U100_packed | 345 | 61.401% | 52.270 ms | 21.765 |

사전 고정한 기술 통계 기준선 (a)는 U50이다. ReComp 대비 −0.505 pp의 CI가
−1.050~+0.026으로 0을 겨우 포함한 경계선상의 차이 미입증이며, 동등성은 아니다.
같은 U50을 FullLoad와 비교하면 −0.542 pp(CI −1.079~−0.012)로 0을 포함하지 않는다.
기준선 (b), 기존 LLaVA 균등 25%의 손실 점추정 2.30 pp 이하는 U40에서 처음 만족했다
(Qwen−ReComp −1.231 pp). 두 기준선은 이 HOLD에서 관찰한 값이며 다른 데이터에서
확인되지 않았다. LLaVA 기준값도 이 sweep에서 다시 측정하지 않았다.

U100과 FullLoad는 12,183요청의 생성 token이 전부 같다. U100의 TTFT −18.500 ms는
파일 배치·57 pread→1 preadv·padding read 제거·GPU 조립을 합친 packed 경로 효과이며
토큰 선택 효과가 아니다. Shared-stream session은 U25/U100이 268.763/283.038 ms,
standalone-equivalent는 290.944/328.455 ms(ReComp 399.398 ms)다.
Standalone은 setup 재귀속 DERIVED 값이며 독립 cold-session 실측이 아니다.
실행 간 절대 TTFT나 모델 간 효과는 합치지 않는다. DONTNEED는 cold NAND를 보장하지 않는다.

[전체 결과](results/qwen_budget_sweep_20261111T152143Z/REPORT.md),
[품질표](results/qwen_budget_sweep_20261111T152143Z/quality.csv),
[지연·I/O](results/qwen_budget_sweep_20261111T152143Z/latency_io.csv),
[결과 기록](results/qwen_budget_sweep_20261111T152143Z/RESULT_RECORD.md),
[실험 계약](docs/qwen_budget_sweep_contract.md),
[재현 정보](results/qwen_budget_sweep_20261111T152143Z/REPRODUCE.md)를 참조한다.
세 실험의 그림·완료/검증 receipt·실패/개발 리허설 기록·freeze/provenance를 함께 보존한다.
대용량 raw·KV payload·전체 manifest·프롬프트 파일은 커밋에서 제외한다.

## 34. LLaVA-1.5-7B 포팅 + MT-GQA packed KV 실험 (2026-11-15 정리)

LLaVA-1.5-7B의 고정 576 visual tokens, separator 0 조건에 별도 runner/store/serving
경로를 추가했다. 모델은 HF revision b234b804, NF4/BF16/eager, FP16 SSD KV,
greedy cap16이다. HF 기본 336×336 중앙 자르기 전처리를 유지했으며 공개 GQA 수치와는
양자화·전처리·프롬프트·재구성 표본이 달라 직접 비교하지 않는다.
선택 순서는 image-only CLIP CLS→patch attention이고 질문은 사용하지 않는다.

MT-GQA 398 images/4,061 dialogues × 3턴의 5-arm main을 60,915/60,915요청으로
완료했다. 독립 감사 PASS, 실패·skip 0이며 모든 arm의 T1 생성 token이
4,061/4,061대화에서 일치했다. Followup은 T2–T3의 method-local generated history
아래 strict normalized EM이다. Packed arm은 hit당 preadv 1회, vision/score 호출 0이다.

| Arm | Content rows/layer | 전체 EM | Followup EM | Followup TTFT | SSD MB/followup |
|---|---|---:|---:|---:|---:|
| ReComp | — | 63.318% | 65.279% | 126.86 ms | 0.00 |
| U100_packed | 576 | 63.334% | 65.304% | 209.32 ms | 301.99 |
| U50_packed | 288 | 63.104% | 64.959% | 126.77 ms | 150.99 |
| U25_packed | 144 | 62.013% | 63.322% | 86.38 ms | 75.50 |
| PYR-b20_packed | 281…7 (합 4,608) | 62.045% | 63.371% | 86.36 ms | 75.50 |

사전 고정 주 운영점 U25는 ReComp 대비 −1.958 pp(이미지 단위 paired bootstrap
95% CI −2.693~−1.239)로 정확도가 낮지만, 평균 TTFT는 0.681×로 31.9% 짧았다.
U50−ReComp는 −0.320 pp(CI −0.817~+0.171)로 차이 미입증이지 동등성은 아니다.
PYR-b20−U25는 +0.049 pp(CI −0.472~+0.570)로 bytes·TTFT 조건을 만족해도
개선 기준에 못 미쳐 layer 배분 사전 판정은 **NO-GO**다.
PYR은 기존 중요도 순서의 layer별 절단만 사용하며 PyramidKV 전체 시스템 재현이 아니다.

100% visual KV는 301,989,888 bytes다. U100/ReComp TTFT 비율 R100=1.650으로
사전 예상 구간을 벗어났고, U25/U100 비율 S25=0.413이었다.
LLaVA-NeXT의 100% packed 지점은 미측정이므로 직접적인 packed 모델 간 비교가 아니다.
U100과 ReComp는 prefill shape·FP16 왕복 등의 수치 차이가 있어 token 완전 동치를
주장하지 않는다. 별도 native generate·capture·same-shape reference·I/O gate는 통과했다.

U25의 shared-stream session은 386.5 ms(ReComp 445.1 ms)이지만, setup을 매
dialogue에 다시 부과한 standalone-equivalent는 575.4 ms로 ReComp보다 130.4 ms 길다.
Standalone은 DERIVED이며 독립 cold-session 실측이 아니다. Dense GPU buffer는 유지하므로
GPU 메모리 절감을 주장하지 않는다. H2D/scatter 분해는 gate fixture 진단이며 HOLD 계측과
구분한다. DONTNEED는 cold NAND 보장이 아니고 실행 간 절대 TTFT도 비교하지 않는다.
이번에는 MT-VQA·Qwen·GLM·LLaVA-NeXT를 재실행하지 않았다.

[전체 결과](results/llava15_mt_gqa_20261112T062708Z/REPORT.md),
[품질표](results/llava15_mt_gqa_20261112T062708Z/quality.csv),
[지연·I/O](results/llava15_mt_gqa_20261112T062708Z/latency_io.csv),
[최종 판정](results/llava15_mt_gqa_20261112T062708Z/decision.json),
[결과 기록](results/llava15_mt_gqa_20261112T062708Z/RESULT_RECORD.md),
[실험 계약](docs/llava15_mt_gqa_contract.md),
[재현 정보](results/llava15_mt_gqa_20261112T062708Z/REPRODUCE.md)를 참조한다.
본실험과 개발 리허설의 검증·freeze·원래 수정 이력을 구분해 보존하며,
대용량 raw·KV·모델 가중치·전체 데이터 manifest·프롬프트 파일은 제외한다.

## 35. LLaVA-NeXT PYR-beta-raw packed: MT-VQA 25k (2026-11-16)

MT-GQA에서 사용한 image-only VisionZip 순서와 PyramidKV β=20 원식 layer 배분
(PYR-beta-raw, 하한 0)을 MT-VQA-v2-reconstructed-25k에 적용했다.
LLaVA-NeXT-7B(v1.6-vicuna, revision c916e6cd), NF4/BF16/eager, FP16 SSD KV,
greedy cap16 조건이다. 새 본실험은 PYR packed **단일 arm**이며,
25,000 images × 3턴 = 75,000/75,000요청을 23.55시간에 완료했다.
실패·skip·재시도 0, 독립 감사 PASS다. Smoke·pilot40·gate 요청은 main 분모에서 제외한다.

아래 baseline은 2026-10의 v3 실행 기록이다. 정확도는 같은 image·turn을 짝지은
**cross-run, 결정성 확인됨** 비교다. T1 생성 token은 25,000/25,000장 일치했고,
균등 U_packed의 pilot40 gate도 기존 [B]와 120/120요청의 입력·생성 token·점수·읽기
바이트가 일치했다. 채점은 ten-answer leave-one-out 공식 VQA score(%)이며,
followup(T2–T3)은 method-local generated history를 사용한다.

| Arm | 실행 | T1 | T2 | T3 | 전체 | Followup |
|---|---|---:|---:|---:|---:|---:|
| PYR-beta-raw packed | 이번 main | 51.7940 | 50.7968 | 50.8964 | 51.1624 | 50.8466 |
| ReComp | v3 기록 | 51.7940 | 51.6864 | 51.8592 | 51.7799 | 51.7728 |
| FullLoad | v3 기록 | 51.7940 | 51.6852 | 51.8760 | 51.7851 | 51.7806 |
| SparseVLM-SSD-KV25-AllHead | v3 기록 | 51.7940 | 51.4476 | 51.6948 | 51.6455 | 51.5712 |
| PrefixStore-KV25 [B] (균등 배분) | v3 기록 | 51.7940 | 49.8396 | 50.3808 | 50.6715 | 50.1102 |

사전 고정 주 판정은 **IMPROVED**다. PYR−[B] followup은 +0.7364 pp
(이미지 단위 paired bootstrap 95% CI +0.5916~+0.8824, 10,000회, seed1234)로,
ReComp 대비 [B]의 손실 중 44.3%를 회복했다. MT-GQA의 +1.0 pp 기준은 참고값이며
이번 판정 기준이 아니다. PYR−ReComp는 −0.9262 pp(CI −1.0846~−0.7638)로
원래 정확도에는 못 미친다. MT-GQA의 +1.244 pp(EM)와 개선 방향은 같지만 지표가 달라
효과 크기를 직접 비교하지 않는다. MPIC는 이번 비교표에서 제외했다.

이번 실행의 followup TTFT는 평균 228.34 ms(p50 228.67, p95 269.88),
T1 TTFT는 527.00 ms이며 같은 실행 안 비율은 0.4333이다.
SSD 읽기는 [B]와 같은 평균 307.75 MB/followup, hit당 preadv 1회·pread 0회다.
PYR은 같은 중요도 순서를 layer별로 자르며 PyramidKV의 attention 기반 선택은 쓰지 않는다.
정확도 비교와 달리 지연 차이에는 packed I/O 경로 효과도 섞이므로,
v3의 절대 TTFT·session은 cross-run 참고치로만 다룬다.

이미지당 store는 평균 310.48 MB, persistence 711.49 ms, activation 189.53 ms다.
이 비용을 포함한 3-turn session은 1,981.98 ms이며 이미지당 dialogue 1개라 분할 상각은 없다.
기존 개발 표본과 이미지 166개가 겹쳐 unseen holdout이 아니다. DONTNEED는 cold NAND를
보장하지 않으며 dense GPU buffer를 유지하므로 GPU 메모리 절감을 주장하지 않는다.
Qwen·GLM·LLaVA-1.5·MT-GQA와 baseline arm들은 이번에 재실행하지 않았다.

[전체 결과](results/llava_next_mtvqa_pyr_20261114T182331Z/REPORT.md),
[품질표](results/llava_next_mtvqa_pyr_20261114T182331Z/quality.csv),
[짝지은 비교](results/llava_next_mtvqa_pyr_20261114T182331Z/paired_analysis.csv),
[지연·I/O](results/llava_next_mtvqa_pyr_20261114T182331Z/latency.csv),
[최종 판정](results/llava_next_mtvqa_pyr_20261114T182331Z/decision.json),
[결과 기록](results/llava_next_mtvqa_pyr_20261114T182331Z/RESULT_RECORD.md),
[실험 계약](docs/llava_next_mtvqa_pyr_contract.md),
[재현 절차](results/llava_next_mtvqa_pyr_20261114T182331Z/REPRODUCE.md)를 참조한다.
실험 코드·결과표·검증 receipt·freeze와 소표본 분석 수정 이력을 함께 보존하며,
대용량 raw·KV·모델·전체 데이터 manifest·프롬프트 파일은 제외한다.

## 36. IMPRESS 중요도 지표 × fallback V 읽기 파일럿 (2026-11-16)

IMPRESS 이식 경로에 `importance_metric=h2o`(전체 suffix query 행의 attention 열 합)와
`fallback_v=selected`(head별 선택의 합집합을 덮는 V chunk만 읽기)를 추가했다.
기본값은 기존 `sparsevlm`·`full`이다. H2O는 rater 선택과 `v_hidden.pt`를 사용하지 않으며,
store 재배치와 온라인 선택의 중요도 지표 일치도 검사한다.

GQA 40장/240문항, ratio=0.25, α=0.6, probe heads=3에서 지표별 store를 사용했다.
각 지표 실행에 ReComp·FullLoad·full·selected를 함께 측정했고,
별도 SparseVLM gate는 기존 `eval_b25.json`의 240문항 답·읽기량·fallback을 재현했다.
아래 TTFT는 각 지표의 별도 실행에서 얻은 값이다.

| 중요도 | Fallback V | 정확도 | TTFT 평균 | SSD MB/요청 | 같은 실행 FullLoad 대비 읽기량 | Fallback 비율 |
|---|---|---:|---:|---:|---:|---:|
| SparseVLM | full | 60.00% | 841.5 ms | 928.1 | 79.66% | 15.8% |
| SparseVLM | selected | 60.00% | 851.5 ms | 927.5 | 79.61% | 15.8% |
| H2O | full | 61.25% | 773.3 ms | 1010.4 | 86.72% | 33.3% |
| H2O | selected | 61.25% | 784.2 ms | 1009.8 | 86.68% | 33.3% |

같은 지표의 full/selected는 각각 240/240문항의 생성 token이 일치했다.
V 읽기 절감은 요청당 SparseVLM 0.63 MB, H2O 0.54 MB에 그쳤다.
별도 30요청 진단에서 fallback의 head별 토큰 합집합이 V chunk의 약 99%를 덮었기 때문이다.
head·token 단위로 정확히 읽는 가상 하한은 실측 SSD 바이트에 포함하지 않는다.
Selected는 full 다음에 고정 측정돼 지연 차이와 순서 효과가 분리되지 않는다.
H2O와 SparseVLM의 정답 차이는 3문항이며, 이 파일럿으로 지표 간 정확도 우위를 주장하지 않는다.
실행 간 절대 TTFT도 직접 비교하지 않는다.

[전체 결과·한계](results/impress_fidelity_20261116T052410Z/report.md),
[요약·재현 gate](results/impress_fidelity_20261116T052410Z/summary.json),
[SparseVLM 결과표](results/impress_fidelity_20261116T052410Z/sparsevlm/summary.csv),
[H2O 결과표](results/impress_fidelity_20261116T052410Z/h2o/summary.csv)를 참조한다.
변경 전 기본 옵션과의 GPU 동등성 검사 로그·store 재배치 절차·요청별 결과도 함께 보존한다.

## 37. IMPRESS 다음 layer probe key 선행 로딩 (2026-11-16)

`prefetch=next_layer`는 layer i의 hook이 끝나면 단일 CPU worker로 layer i+1의
probe key를 읽어 현재 layer 연산과 겹친다. 선택된 K/V chunk와 fallback K 읽기는 동기다.
기본값 `none`과 FullLoad는 기존 동기 경로를 사용한다. Worker 시작은 요청 타이머 전이며
실제 읽기는 타이머 안에서 수행한다. 실패·조기 종료 시 진행 중인 읽기를 정리한다.

H2O 재배치 store, `importance_metric=h2o`, `fallback_v=selected`로
GQA 40장/240문항을 측정했다. 두 IMPRESS arm의 측정 순서는 요청마다 번갈아 바꿔
동기 먼저/선행 로딩 먼저가 각 120건이다. 별도 2장/12문항 smoke는 아래 분모에서 제외한다.

| Arm | 정확도 | TTFT 평균 | p50 | p95 | SSD MB/요청 |
|---|---:|---:|---:|---:|---:|
| ReComp | 62.50% | 508.3 ms | 510.2 ms | 595.5 ms | 0.0 |
| FullLoad | 62.08% | 680.9 ms | 679.5 ms | 807.1 ms | 1165.1 |
| IMPRESS 동기 | 61.25% | 781.9 ms | 796.8 ms | 895.0 ms | 1009.8 |
| IMPRESS 선행 로딩 | 61.25% | 731.4 ms | 740.0 ms | 850.0 ms | 1009.8 |

두 IMPRESS arm은 240/240문항의 생성 token·답·읽기 바이트·pread 횟수·fallback 비율이
같고, 동기 arm도 이전 H2O selected 결과를 요청 단위로 재현했다.
선행 로딩−동기의 paired TTFT는 평균 −50.45 ms(요청 단위 Student t 95% CI
−62.65~−38.26)로 약 6.5% 감소했다. 이는 40개 이미지에 속한 요청 단위 구간이며
이미지 cluster bootstrap이나 반복 실행의 변동을 추정한 구간은 아니다.
선행 읽기 31개 중 평균 30.78개가 사용 시점에 완료됐고, 대기는 요청당 0.28 ms였다.
그래도 같은 실행의 FullLoad보다 1.074×, ReComp보다 1.439× 느리다.
읽기와 연산이 겹치므로 hook 시간과 디스크 시간을 더해 TTFT로 해석하지 않는다.

[전체 결과·실행 명령](results/impress_prefetch_20261116T064612Z/report.md),
[요약·동등성·paired 분석](results/impress_prefetch_20261116T064612Z/summary.json),
[본실험 표](results/impress_prefetch_20261116T064612Z/main/summary.csv),
[요청별 기록](results/impress_prefetch_20261116T064612Z/main/results.jsonl)을 참조한다.
두 파일럿의 실행 코드·기존 구현 백업·검증 로그·결과를 포함하며 약 49.7 GB의 H2O KV store는
로컬에 보존한다. MT-GQA 재실행·다른 비율·warm cache·반복 실행은 이번에 측정하지 않았다.
Cold 설정의 page-cache drop 요청은 cold NAND를 보장하지 않는다.

## 38. MT-GQA 다중 턴: FullLoad·IMPRESS·Ours 비동기 KV 읽기 (2026-11-17)

LLaVA-NeXT-7B, NF4/BF16/eager, FP16 SSD KV에서 네 방법을 같은 실행 안에서 비교했다.
환경은 기존 HOLD와 같은 transformers 4.57.6이며, 앞선 IMPRESS 단일 턴 파일럿의
5.13.1 환경과 구분한다. MT-GQA 398 images/4,061 dialogues × 3턴 × 4 arm의
48,732/48,732요청을 완료했고, 398장 모두 독립 감사 PASS다.
모든 방법은 매 대화 T1을 픽셀부터 계산하며 T2–T3는 method-local generated history를
사용한다. 아래 정답률은 followup 8,122요청/arm의 strict normalized EM이다.

FullLoad는 layer별 K·V 전체, IMPRESS는 다음 layer의 probe key를 CPU worker로 미리 읽는다.
IMPRESS는 H2O, probe heads=3, α=0.6, 25% 유지, chunk64, selected V 설정이며,
이미지 첫 대화 T1 질문의 중요도로 한 번 재배치한다. Ours는 기존 PYR-beta-raw packed
벡터와 저장 형식을 유지하고 separator → layer별 구간을 미리 읽는다.
읽기는 요청 타이머 시작 뒤에 수행하며 GPU 적재는 요청 스레드가 맡는다.
동기 경로의 기본값은 유지한다. 이번 Ours 비동기 경로는 hit당 33회 preadv이며,
기존 동기 packed의 hit당 1회 읽기와 구분한다.

| 방법 | Followup EM | TTFT 평균 | p50 | p95 | SSD MB/followup |
|---|---:|---:|---:|---:|---:|
| ReComp | 68.48% | 530.0 ms | 532.7 ms | 596.9 ms | 0.0 |
| FullLoad | 68.38% | 488.0 ms | 504.2 ms | 581.4 ms | 1173.9 |
| IMPRESS | 68.20% | 721.3 ms | 727.3 ms | 818.6 ms | 1020.3 |
| Ours | 67.42% | 148.7 ms | 149.4 ms | 170.3 ms | 308.0 |

Ours의 TTFT는 같은 실행의 FullLoad 대비 0.305×, ReComp 대비 0.281×다.
다만 Ours−IMPRESS 정답률은 −0.78 pp(95% CI −1.43~−0.14),
Ours−FullLoad는 −0.96 pp(CI −1.60~−0.34), Ours−ReComp는 −1.06 pp
(CI −1.69~−0.44)로 손실이 있다. IMPRESS−FullLoad는 −0.18 pp
(CI −0.60~+0.24)로 차이 미입증이며 동등성의 증거는 아니다.
CI는 이미지 단위 paired bootstrap 10,000회, seed1234다.
측정 보고이며 GO/NO-GO 판정은 두지 않았다.

본실험 전 gate 4장/49대화와 DEV 40장/378대화에서 동기·비동기 생성 token과 읽기
바이트가 모두 일치했고, 기존 HOLD 요청과의 token 대조도 통과했다.
DEV의 같은 요청에 대한 비동기−동기 followup TTFT 차이는 아래와 같다.

| 방법 | 동기 평균 | 비동기 평균 | Paired 차이 [95% CI] |
|---|---:|---:|---:|
| FullLoad | 772.4 ms | 479.9 ms | −292.5 [−300.5, −284.0] ms |
| IMPRESS | 751.9 ms | 727.7 ms | −24.2 [−25.8, −22.8] ms |
| Ours | 218.9 ms | 147.8 ms | −71.2 [−74.3, −67.8] ms |

Gate 1,176요청과 DEV 7,938요청은 main 분모에서 제외한다. DEV 이미지는 main에도
포함되며, 398장 전체를 기존 HOLD와 token 단위로 다시 대조한 것은 아니다.
IMPRESS의 main fallback은 29.1%, K·V chunk 비율은 81.7%였다.
재배치하지 않았을 경우의 99.8%는 같은 선택을 raster 위치로 되돌린 계산값이며 실측이 아니다.

Ours의 3턴 shared session은 평균 996.1 ms, 매 대화에 setup을 재부과한
standalone-equivalent는 1,806.8 ms(ReComp 1,668.0 ms)다. Standalone은 DERIVED이며
독립 cold-session 실측이 아니다. IMPRESS의 별도 중요도 집계 508 ms와 재배치 796 ms는
followup TTFT 밖의 이미지당 비용이다. 이를 T1 추론에 합치는 구현은 측정하지 않았다.
읽기·연산이 겹치므로 hook과 디스크 시간을 더해 TTFT로 해석하지 않으며,
DONTNEED는 cold NAND를 보장하지 않는다. 반복 실행·다른 비율·Qwen·GLM 실험은 미실행이다.

[전체 결과](results/mt_gqa_async_20261116T081635Z/report.md),
[본실험 표](results/mt_gqa_async_20261116T081635Z/main/arms.csv),
[짝지은 비교](results/mt_gqa_async_20261116T081635Z/main/paired.csv),
[DEV 동기·비동기](results/mt_gqa_async_20261116T081635Z/dev/sync_async.csv),
[저장·T1 비용](results/mt_gqa_async_20261116T081635Z/main/storage.csv),
[세션 비용](results/mt_gqa_async_20261116T081635Z/main/sessions.csv),
[요약·검증 상태](results/mt_gqa_async_20261116T081635Z/summary.json),
[실험 계약](docs/mt_gqa_async_contract.md)을 참조한다.
실행 코드·결과표·동결 설정·검증 receipt를 보존하며 대용량 raw·KV·전체 manifest와
파일별 보호 inventory는 로컬에 둔다.

## 39. MT-GQA: 공개 IMPRESS 구현 로직 이식 (2026-11-17, 본실험 미실행)

IMPRESS(FAST'25)·HyperInfer(IEEE TC'26) 논문은 중요도를 "attention 열 합"이라고만 적으므로,
baseline 정의를 공개 구현([HyperInfer](https://github.com/ISCS-ZJU/HyperInfer) 커밋 `91df7a2`)으로 바꿔
`mmimpress/impress_official.py`로 이식했다. probe head 0·1·2, 전체 행(prefix + suffix)의 attention 열 합,
head별 상위 25% → 평균 pairwise Jaccard ≥ 0.3이면 득표 합의, 아니면 그 layer의 K·V 전체를 mask 없이 사용,
chunk 64, 첫 대화 T1 통계로 layer별 재배치, 다음 layer probe 선행 읽기다. 공개 함수와의 동등성은
같은 텐서를 넣는 CPU 테스트로 확인했다(공개 코드는 복사하지 않고 레포 밖 체크아웃을 참조).

사전 고정 검증 3번("유지 100% IMPRESS = raster store의 FullLoad, 모든 요청")이 gate 147건 중 1건의
BF16 logit 동점으로 FAIL이어서 398장 본실험은 돌리지 않았다. DEV 40장(arm당 followup 756건)은 다음과 같다.

| 방법 | Followup EM | TTFT 평균 | SSD MB/followup |
|---|---:|---:|---:|
| ReComp | 68.92% | 528.9 ms | 0.0 |
| FullLoad | 69.44% | 481.7 ms | 1166.9 |
| IMPRESS (공개 구현) | 61.38% | 738.4 ms | 1042.4 |
| IMPRESS (38절의 이전 구현) | 68.52% | 709.6 ms | 1013.9 |
| Ours | 68.52% | 149.2 ms | 306.8 |

공개 구현 − 이전 구현: 정답률 −7.14 pp(이미지 단위 95% CI −10.49~−3.97), fallback 52.4% vs 27.5%.

[보고서](results/mt_gqa_impress_official_20261117T052201Z/report.md),
[실험 계약](docs/mt_gqa_impress_official_contract.md).

## 40. LLaVA-1.5 × MT-VQA 25k: 38절 IMPRESS 구현 (2026-11-17, 중단)

LLaVA-1.5-7B(576 visual token, separator 없음)로 ReComp / FullLoad / IMPRESS(38절 구현) / Ours(34절 PYR-b20 packed)를
MT-VQA-v2-reconstructed-25k에서 재려 했다. 첫 계약은 검증 4("유지 100% IMPRESS = raster FullLoad")가
132건 중 1건의 BF16 근소차(19.375 vs 19.25)로 FAIL이라 본실험 없이 끝났다.
사용자 결정으로 검증 4를 "같은 재배치 store를 FullLoad loader로 읽은 결과와 비교"로 바꾼 새 계약으로 다시 돌려 통과했고
본실험을 시작했지만, IMPRESS baseline을 공개 구현으로 다시 정하기로 하면서 5,956 / 25,000장(71,472 / 300,000요청)에서
이미지 경계로 중단했다. 부분 결과는 표로 내지 않는다(42절이 대체한다).

두 계약의 pilot 40장(본조건 4 arm, followup 80건/arm, 공식 VQA score)은 같은 정답률을 냈다:
ReComp 47.50, FullLoad 45.75, IMPRESS 47.75, Ours 44.13; TTFT 122 / 117 / 237 / 74 ms. 표본이 작아 모든 정답률 차이의 CI가 0을 포함한다.

[첫 계약 보고서](results/llava15_mtvqa_impress_20261117T074736Z/report.md),
[새 계약 pilot](results/llava15_mtvqa_impress_20261117T085444Z/pilot/report.md),
[중단 기록](results/llava15_mtvqa_impress_20261117T085444Z/STATUS_STOPPED.json),
[계약 1](docs/llava15_mtvqa_impress_contract.md), [계약 2](docs/llava15_mtvqa_impress_v2_contract.md).

## 41. 실험 A — LLaVA-NeXT × MT-GQA, IMPRESS = 공개 구현 (2026-11-18)

39절의 공개 구현 IMPRESS를 유일한 IMPRESS arm으로 두고 38절과 같은 조건(398장 / 4,061대화 × 3턴 × 4 arm,
48,732요청, transformers 4.57.6, cold SSD, 비동기 읽기)에서 다시 쟀다. 검증 3번은 사용자 지시에 따라
"유지 100% IMPRESS = 같은 재배치 store의 FullLoad loader"로 정했고(147/147 일치), 검증 1–7이 모두 PASS였다.
ReComp·FullLoad·Ours의 생성 토큰은 38절 본실험과 12,183건씩 모두 같았다.

| 방법 | Followup EM | TTFT 평균 | p50 | p95 | SSD MB/followup |
|---|---:|---:|---:|---:|---:|
| ReComp | 68.48% | 530.1 ms | 533.0 ms | 595.5 ms | 0.0 |
| FullLoad | 68.38% | 489.5 ms | 487.6 ms | 653.9 ms | 1173.9 |
| IMPRESS | 60.63% | 727.8 ms | 730.7 ms | 868.0 ms | 1052.2 |
| Ours | 67.42% | 149.8 ms | 150.1 ms | 173.7 ms | 308.0 |

Ours − IMPRESS +6.80 pp(CI +5.79~+7.80), Ours − FullLoad −0.96 pp(CI −1.60~−0.34),
IMPRESS − FullLoad −7.76 pp(CI −8.73~−6.81). Ours TTFT는 FullLoad 대비 0.306×, ReComp 대비 0.283×다.
IMPRESS는 fallback이 layer·요청 평균 53.8%(평균 Jaccard 0.338, 임계값 0.3 근처)라 K·V chunk의 84.5%를 읽고,
읽기 호출이 124회로 FullLoad(64회)보다 많아 GPU 읽기 대기가 381.9 ms였다. 중요도 합의 98.7%가 prefix 행이다.
공개 구현과 다른 점 15개와 그 방향은 보고서에 표로 적었다. 측정 보고이며 GO/NO-GO 판정은 없다.

[보고서](results/mt_gqa_impress_code_20261117T185832Z/report.md),
[본실험 표](results/mt_gqa_impress_code_20261117T185832Z/main/arms.csv),
[짝지은 비교](results/mt_gqa_impress_code_20261117T185832Z/main/paired.csv),
[IMPRESS layer별 통계](results/mt_gqa_impress_code_20261117T185832Z/main/impress_layers.csv),
[실험 계약](docs/mt_gqa_impress_code_contract.md).

## 42. 실험 B — LLaVA-1.5 × MT-VQA 25k, IMPRESS = 공개 구현 (2026-11-19)

41절과 같은 IMPRESS 코드로 LLaVA-1.5-7B × MT-VQA-v2-reconstructed-25k를 쟀다.
25,000장 × 3턴 × 4 arm = 300,000요청, 35.3시간, 검증 1–8 모두 PASS. 공식 ten-answer VQA score다.

| 방법 | Followup score | TTFT 평균 | p50 | p95 | SSD MB/followup |
|---|---:|---:|---:|---:|---:|
| ReComp | 48.61 | 122.6 ms | 133.0 ms | 135.9 ms | 0.00 |
| FullLoad | 48.59 | 118.4 ms | 117.8 ms | 129.0 ms | 301.99 |
| IMPRESS | 48.11 | 240.6 ms | 237.8 ms | 268.4 ms | 263.35 |
| Ours | 47.21 | 74.7 ms | 73.9 ms | 81.2 ms | 75.50 |

Ours − IMPRESS −0.91 pp(CI −1.09~−0.72), Ours − FullLoad −1.38 pp(CI −1.55~−1.21),
IMPRESS − FullLoad −0.47 pp(CI −0.62~−0.33). Ours TTFT는 FullLoad 대비 0.631×, ReComp 대비 0.610×다.
IMPRESS fallback 47.96%, 평균 Jaccard 0.342.

[보고서](results/llava15_mtvqa_impress_code_20261118T053221Z/report.md),
[실험 계약](docs/llava15_mtvqa_impress_code_contract.md).

## 43. Ours: 중요도 순 재배열을 빼도 되는가 (2026-11-20)

Ours는 이미지 토큰을 VisionZip 점수 내림차순으로 재배열해 저장하고 GPU cache에도 그 순서로 올린다
(layer l은 저장 행 [0, k_l)). 지금은 25%만 저장하고 전부 읽으므로 저장 순서가 읽기량을 바꾸지 않는다.
packed store에 `row_order="original"` 옵션을 추가했다(기본값은 기존 재배열 그대로).
layer별 토큰 집합과 k_l은 같고, 그 집합을 원래 이미지 토큰 순서로 저장해 GPU cache의 원래 위치에 올리며
나머지 위치는 mask로 가린다. 파일 구성·비동기 읽기 순서·읽기 호출 수는 같고, 메타데이터에는 중요도 순열 대신
layer별 원래 인덱스를 복원하는 표만 둔다.

41절과 같은 조건(398장, ReComp / FullLoad / Ours(재배열) / Ours(원래 순서), 48,732요청)에서 쟀다.
gate(4장 → DEV 40장): 두 store의 같은 토큰 K·V 행이 비트 단위로 같음(DEV 1,397,760행 중 차이 0),
원래 순서의 동기 = 비동기, 두 번 실행 시 토큰 동일, ReComp·FullLoad·Ours(재배열)의 토큰이 41절 실행과 같음,
기존 테스트 통과가 모두 PASS였다. "질문 무관" gate는 layer별 인덱스 표·행 수·배치가 질문과 무관한지로 판정했다
(사용자 결정). T1에서 캡처되는 Visual KV 값 자체는 프롬프트의 토큰 길이에 따라 달라지며, 이는 두 방식 모두 같다.

| 방법 | Followup EM | TTFT 평균 | p50 | p95 | SSD MB/followup | 읽기 호출 수 |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 68.48% | 529.4 ms | 532.6 ms | 592.6 ms | 0.0 | 0 |
| FullLoad | 68.38% | 487.4 ms | 492.0 ms | 578.6 ms | 1173.9 | 64 |
| Ours(재배열) | 67.42% | 148.3 ms | 149.0 ms | 169.6 ms | 308.0 | 33 |
| Ours(원래 순서) | 67.54% | 148.2 ms | 148.7 ms | 170.7 ms | 308.0 | 33 |

실행 전에 고정한 판정은 **"빼도 됨"**이다. (a) 정답률 차이(원래 순서 − 재배열) +0.12 pp, 95% CI −0.07~+0.33으로
하한이 −0.5 pp보다 크다(차이 자체는 미입증). (b) TTFT 평균 비율 0.999 ≤ 1.02. (c) followup 8,122쌍 모두
읽기 바이트와 읽기 호출 수가 같다. 두 방식의 생성 토큰이 다른 요청은 T2 41건, T3 40건이다.
T1 저장에서는 행 골라내기가 60.4 → 13.4 ms, 메타데이터가 102 KB → 46 KB로 줄었다.
두 Ours의 arm 순서는 블록마다 자리를 맞바꿔 직전 arm 구성을 같게 했다.
25%보다 많이 저장하고 일부만 읽는 구성(재배열은 구간 앞부분만 읽을 수 있다)은 재지 않았다.

[보고서](results/mt_gqa_ours_order_20261120T052725Z/report.md),
[판정](results/mt_gqa_ours_order_20261120T052725Z/verdict.json),
[실험 계약](docs/mt_gqa_ours_order_contract.md).

## 44. 실험 A 표의 Ours 행 교체 (2026-11-21)

41절 표에서 Ours 행만 43절의 Ours(원래 순서)로 바꾼 표다. ReComp·FullLoad·IMPRESS는 41절 값 그대로다.
43절 실행의 ReComp·FullLoad·Ours(재배열) 토큰이 41절과 모두 같았으므로 정답률과 정답률 차이는 정확하다.
TTFT는 Ours만 11/20, 나머지는 11/17에 잰 값이다(같은 실행 비율은 FullLoad 대비 0.304×, ReComp 대비 0.280×).

| 방법 | Followup EM | TTFT 평균 | 비교 (정답률, 95% CI) |
|---|---:|---:|---|
| ReComp | 68.48% | 530.1 ms | |
| FullLoad | 68.38% | 489.5 ms | |
| IMPRESS | 60.63% | 727.8 ms | |
| Ours | 67.54% | 148.2 ms | − IMPRESS +6.92 pp [+5.95, +7.90]; − FullLoad −0.84 [−1.45, −0.23]; − ReComp −0.94 [−1.55, −0.33] |

[보고서](results/mt_gqa_impress_code_ours_original_update/report.md) (`scripts/205_report_mt_gqa_ours_swap.py`).

## 45. LLaVA-1.5 × MT-GQA 본실험: ReComp / FullLoad / IMPRESS / Ours (2026-11-21)

43절의 원래 순서 저장(`row_order="original"`)을 LLaVA-1.5 경로(`mmimpress/llava15/store.py`, `serve.py`)에 옮겼다.
기본값은 기존 재배열 그대로이고, 원래 순서는 옵션이다. 새 store 형식은 layer별 토큰 집합과 k_l(PYR-b20, 합계 4,608행)이
재배열 store와 같고, 원래 이미지 토큰 순서로 저장해 GPU cache의 원래 위치에 올린다. IMPRESS는 11/18 실행(42절)의 구현과
설정을 그대로 썼다. 398장, followup T2–T3, 48,732요청을 한 실행에서 쟀다. 측정 보고이며 판정은 두지 않는다.

gate(4장 → DEV 40장): Ours store의 K·V 행이 재배열 store의 같은 토큰 행과 비트 단위로 같음, 질문 무관,
Ours 동기 = 비동기, 반복 실행 동일, IMPRESS r100 = FullLoad, ReComp 토큰이 11/12 실행과 같음, 기존 테스트 새 실패 없음이
모두 PASS였다. 본실험은 중단 없이 2.6시간 걸렸다. T1 생성 토큰은 4,061개 대화 모두 네 방법에서 같고,
ReComp 토큰은 11/12 실행과 12,183/12,183 일치했다.

| 방법 | Followup EM | TTFT 평균 | p50 | p95 | SSD MB/followup | 읽기 호출 수 |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 65.28% | 126.1 ms | 134.5 ms | 136.7 ms | 0.0 | 0 |
| FullLoad | 65.37% | 116.1 ms | 115.4 ms | 123.2 ms | 302.0 | 64 |
| IMPRESS | 64.52% | 244.8 ms | 243.8 ms | 260.9 ms | 267.6 | 99 |
| Ours | 63.46% | 74.0 ms | 73.5 ms | 79.7 ms | 75.5 | 32 |

짝지은 정답률 차이(이미지 단위 bootstrap 95% CI): Ours − IMPRESS −1.06 pp [−1.80, −0.34],
Ours − FullLoad −1.91 [−2.65, −1.18], Ours − ReComp −1.82 [−2.55, −1.12], IMPRESS − FullLoad −0.85 [−1.45, −0.26],
FullLoad − ReComp +0.09 [−0.18, +0.36]. TTFT 비율은 Ours/FullLoad 0.637×, Ours/ReComp 0.587×, Ours/IMPRESS 0.302×,
IMPRESS/FullLoad 2.109×다. IMPRESS는 fallback 51.5%, 평균 Jaccard 0.337, K·V chunk 읽기 비율 83.9%였다.
T1 저장은 이미지당 FullLoad 318.8 MB / 290 ms, IMPRESS 319.2 MB / 533 ms(재배치 포함), Ours 78.1 MB / 117 ms다.
DEV 40장 참고값: Ours − Ours(재배열) +0.66 pp [0.00, +1.34], TTFT 비율 1.005×, 읽기 바이트·호출 수 차이 없음.

[보고서](results/llava15_mt_gqa_main_20261120T165001Z/report.md),
[실험 계약](docs/llava15_mt_gqa_main_contract.md),
코드 `scripts/206_eval_llava15_mt_gqa_main.py`(실행), `207`(gate), `208`(보고서), `scripts/run_llava15_mt_gqa_main.sh`(드라이버),
테스트 `tests/test_llava15_original_order.py`.

## 46. ConvBench 본실험: ReComp / FullLoad / IMPRESS / Ours (2026-11-24)

ConvBench 577대화 × 3턴에서 LLaVA-NeXT-7b와 LLaVA-1.5-7b의 네 방법을 쟀다(2모델 × 4방법 × 577 × 3 = 13,848요청).
설정은 11/22 점검 실행(`docs/convbench_baseline_alignment.md`: 원본 LLaVA 토큰열, 생성 상한 1024, 판정기 Llama-3.1-8B-Instruct,
판정 출력 상한 2048, 공식 2차 추출)에서 동결한 그대로다. 공통: NF4/BF16/eager, FP16 SSD KV, greedy, 모든 방법이 T1을 픽셀부터 계산,
T2–T3는 자기 방법의 생성 이력, 요청 전 page cache drop, 저장된 KV를 읽는 세 방법은 비동기 읽기.
FullLoad·IMPRESS는 LLaVA-NeXT 41절, LLaVA-1.5 45절 실행과 같은 구현이고, Ours는 재배열 없는 원래 순서 store(총 KV 25 %;
LLaVA-NeXT PYR-beta-raw(43–44절), LLaVA-1.5 PYR-b20(45절))다.
측정 보고이며 판정은 두지 않는다.

gate(4대화 → 파일럿 40대화, 모델마다): 기존 테스트 새 실패 없음, ReComp = 점검 실행, Ours 동기 = 비동기·반복 동일,
IMPRESS 유지 100% = 같은 store의 FullLoad loader, T1 토큰이 네 방법에서 같음, 판정기가 점검 실행의 판정을 글자까지 재현이
두 모델 모두 PASS였다. 분리 실행한 드라이버가 재시작 없이 38.6시간 돌았고 실패한 요청은 0건이다.
ReComp는 점검 실행과 모델마다 1,731/1,731 요청 같았고, T1 생성 토큰은 577/577 대화에서 네 방법이 같았다.

품질(승률 %, 분모 577, 판정 불가는 0승):

| 모델 | 방법 | S1 | S2 | S3 | 평균 |
|---|---|---:|---:|---:|---:|
| LLaVA-NeXT | ReComp | 9.19 | 8.49 | 7.63 | 8.43 |
| | FullLoad | 8.32 | 9.36 | 6.59 | 8.09 |
| | IMPRESS | 7.45 | 10.40 | 6.24 | 8.03 |
| | Ours | 7.28 | 9.19 | 5.72 | 7.39 |
| LLaVA-1.5 | ReComp | 4.85 | 4.16 | 4.51 | 4.51 |
| | FullLoad | 4.33 | 2.77 | 4.16 | 3.76 |
| | IMPRESS | 3.99 | 4.33 | 4.51 | 4.27 |
| | Ours | 3.99 | 3.99 | 4.85 | 4.27 |

평균 점수의 짝지은 차이(대화 단위 bootstrap 95% CI)는 두 모델 모두 모든 비교에서 0을 포함한다.
LLaVA-NeXT: Ours − ReComp −1.04 pp [−2.37, +0.23], Ours − FullLoad −0.69 [−1.96, +0.52], Ours − IMPRESS −0.64 [−1.91, +0.58],
FullLoad − ReComp −0.35 [−1.50, +0.81]. LLaVA-1.5: Ours − ReComp −0.23 [−1.21, +0.81], Ours − FullLoad +0.52 [−0.46, +1.50],
Ours − IMPRESS −0.00 [−0.98, +0.98], FullLoad − ReComp −0.75 [−1.56, +0.06].

시스템(재사용 턴 T2–T3 평균):

| 모델 | 방법 | TTFT 평균 | p50 | p95 | SSD MB/followup | 읽기 호출 수 |
|---|---|---:|---:|---:|---:|---:|
| LLaVA-NeXT | ReComp | 608.8 ms | 590.0 | 869.8 | 0.0 | 0 |
| | FullLoad | 547.1 ms | 541.9 | 762.8 | 1,152.7 | 64.0 |
| | IMPRESS | 799.2 ms | 795.7 | 1,047.5 | 1,010.2 | 131.5 |
| | Ours | 219.5 ms | 206.4 | 313.1 | 302.3 | 33.0 |
| LLaVA-1.5 | ReComp | 161.3 ms | 147.9 | 244.0 | 0.0 | 0 |
| | FullLoad | 131.9 ms | 123.9 | 166.9 | 302.0 | 64.0 |
| | IMPRESS | 265.5 ms | 258.8 | 305.0 | 259.3 | 101.0 |
| | Ours | 92.0 ms | 84.0 | 116.6 | 75.5 | 32.0 |

TTFT 비율은 LLaVA-NeXT Ours/FullLoad 0.401×, Ours/ReComp 0.361×, Ours/IMPRESS 0.275×, IMPRESS/FullLoad 1.461×,
LLaVA-1.5 0.698×, 0.570×, 0.346×, 2.014×다(CI 폭 ±0.01 이내). IMPRESS는 fallback 47.5 % / 42.5 %, 평균 Jaccard 0.346 / 0.352,
K·V chunk 읽기 비율 82.3 % / 81.1 %였다. T1 저장은 대화당 LLaVA-NeXT FullLoad 1,262 MB / 1.06 s, IMPRESS 쓰기 2,491 MB / 2.02 s
(재배치 포함), Ours 321 MB / 0.48 s, LLaVA-1.5 335 MB / 0.32 s, 652 MB / 0.57 s, 94 MB / 0.14 s다. 4096 토큰을 넘는 입력은
LLaVA-NeXT T3에서 8건(Ours 7건)이었고 모두 정상 처리됐다. 장치 쓰기량은 3.227 TB(예상 3.21 TB), SMART는 권한이 없어 재지 못했다.

0단계(보고만): 11/20–11/21 LLaVA-NeXT의 MT-GQA·MT-VQA 프롬프트는 원본 LLaVA 토큰화와 비교해 `<image>` 직후의 `▁` 한 토큰이
빠져 있다(표본 120/120). LLaVA-1.5는 불일치가 없다. 고치지 않았고 그 영향은 재지 않았다.

[보고서](results/convbench_main_20261122T103401Z/report.md),
[실험 계약](docs/convbench_main_contract.md), [점검 실행 설정](docs/convbench_baseline_alignment.md),
[점검 실행 보고서](results/convbench_alignment_20261121T083134Z/report.md),
코드 `scripts/216`(0단계), `217`(실행; LLaVA-NeXT는 `209`, LLaVA-1.5는 `206`의 방법 호출을 import), `218`(gate), `219`(판정),
`220`(보고서), `221` + `scripts/run_convbench_main.sh`(드라이버), `scripts/convbench_main_common.py`,
점검 실행 `scripts/212`–`215`, 테스트 `tests/test_convbench_main.py`.
대화별 답(`<model>/<phase>/conversations/`)과 판정문(`judge/<model>/judgments/`)을 함께 올렸고, 판정 cache와 1 MB 넘는 원시 판정 로그는 올리지 않았다.

## 47. LLaVA-NeXT × MT-GQA 본실험 v2: 원본 LLaVA 토큰열 (2026-11-24)

46절 0단계에서 찾은 토큰 차이(`<image>` 직후 `▁` 한 토큰 누락)를 LLaVA-NeXT MT-GQA 경로에서 고치고(`mmimpress/mt_gqa_tokens.py`,
`tokenization="official"`; 기존 경로는 `"legacy"` 옵션으로 남김) 네 방법을 한 실행에서 다시 쟀다. 398장, followup T2–T3,
48,732요청. 이 실행이 LLaVA-NeXT × MT-GQA의 기준 행이 된다. 측정 보고이며 판정은 두지 않는다.

gate: 기존 본실험이 보낸 프롬프트 48,732개가 고친 경로에서 원본 LLaVA 토큰 ID와 모두 같음(기존 경로 0개), LLaVA-1.5 경로 불변,
기존 경로 옵션으로 11/20 ReComp 재현(147/147), Ours 동기 = 비동기·반복 동일, IMPRESS 유지 100% = FullLoad loader,
T1 토큰이 네 방법에서 같음, 기존 테스트 새 실패 없음이 모두 PASS였다. 본실험 요청마다 확인한 토큰 ID도 48,732/48,732 일치했다.

| 방법 | Followup EM | TTFT 평균 | p50 | p95 | SSD MB/followup | 읽기 호출 수 |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 68.73% | 529.2 ms | 532.0 | 595.3 | 0.0 | 0 |
| FullLoad | 68.46% | 478.1 ms | 481.1 | 588.5 | 1,173.9 | 64.0 |
| IMPRESS | 61.09% | 725.3 ms | 729.1 | 854.4 | 1,052.5 | 123.9 |
| Ours | 67.46% | 149.5 ms | 149.5 | 173.5 | 308.0 | 33.0 |

짝지은 정답률 차이(이미지 단위 bootstrap 95% CI): Ours − IMPRESS +6.37 pp [+5.36, +7.40], Ours − FullLoad −1.00 [−1.60, −0.39],
Ours − ReComp −1.27 [−1.87, −0.68], IMPRESS − FullLoad −7.36 [−8.42, −6.32], FullLoad − ReComp −0.27 [−0.49, −0.06].
TTFT 비율은 Ours/FullLoad 0.313×, Ours/ReComp 0.282×, Ours/IMPRESS 0.206×, IMPRESS/FullLoad 1.517×다.

[보고서](results/mt_gqa_main_v2_20261124T051939Z/report.md), [실험 계약](docs/mt_gqa_main_v2_contract.md),
코드 `mmimpress/mt_gqa_tokens.py`, `scripts/222`(실행; `209`의 방법 호출을 import), `223`(gate), `224`(보고서),
`scripts/run_mt_gqa_main_v2.sh`(드라이버), 테스트 `tests/test_mt_gqa_main_v2.py`.

## 48. HyperInfer 이식과 MT-GQA 측정: 공개 코드의 시간 예산 (2026-11-25)

비교 기준을 IMPRESS에서 HyperInfer(IEEE TC 2026, 공개 코드 ISCS-ZJU/HyperInfer `91df7a2`)로 바꾸기 위해,
IMPRESS 이식 위에 HyperInfer의 다음 layer KV prefetch를 옮겼다(`mmimpress/hyperinfer.py`; 출처 대조표
`docs/hyperinfer_port_contract.md`의 v1 사본은 실행 디렉터리의 `PORT_TABLE.md`). layer l 계산 중에 worker가 layer l+1의
probe key와 예측 chunk(K → V, chunk 번호 순)를 읽고, 공개 코드의 시간 예산(chunk당 읽기 시간 상수, 첫 요청의 절반 규칙)으로
멈춘다. 선택은 IMPRESS와 같고 빠진 KV만 동기로 읽는다. HyperInfer만 두 모델에서 쟀고 나머지 네 방법은 47절·45절 기록과 짝지었다.

gate(4장 → DEV 40장): 공개 코드 로직 대조, prefetch 끔 ≡ IMPRESS, 출력 = IMPRESS, 기존 IMPRESS 기록과 일치,
읽지 않은 KV를 쓰지 않음(NaN 주입), 동기 ≡ 비동기, 반복 동일, 기존 테스트 새 실패 없음이 두 모델 모두 PASS였다.
본실험 생성 토큰은 두 모델 모두 기존 IMPRESS 기록과 12,183/12,183 같았다(EM = IMPRESS).

| 모델 | HyperInfer EM | HyperInfer TTFT | IMPRESS TTFT | HyperInfer / IMPRESS (같은 실행 DEV) | Ours / HyperInfer | prefetch 대기 / 요청 | prefetch로 채운 비율 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-NeXT | 61.09% | 777.0 ms | 725.3 ms | 1.071 (1.084) | 0.192 | 107.9 ms | 27.2% |
| LLaVA-1.5 | 64.52% | 254.9 ms | 244.8 ms | 1.041 (1.072) | 0.290 | 41.9 ms | 54.2% |

HyperInfer가 IMPRESS보다 느렸다. layer당 GPU 계산 시간(≈ 1.6–1.9 ms)보다 worker 작업(probe key 읽기 + chunk 읽기)이 길어
예산 안에서도 매 layer가 다음 선택 전에 prefetch를 기다렸다.

[보고서](results/mt_gqa_hyperinfer_20261124T171039Z/report.md), [실험 계약](docs/mt_gqa_hyperinfer_contract.md),
상수 `docs/mt_gqa_hyperinfer_chunk_times.json`, 코드 `mmimpress/hyperinfer.py`, `scripts/225`(실행), `226`(gate), `227`(보고서),
`scripts/run_mt_gqa_hyperinfer.sh`(드라이버), 테스트 `tests/test_hyperinfer.py`.

## 49. HyperInfer MT-GQA 재측정: 계산이 끝나면 prefetch를 멈춤 (2026-11-25)

48절에서 prefetch 멈춤 방식만 논문 IV-C / Fig. 10(c)대로 바꿨다(`mmimpress/hyperinfer_v2.py`). layer l의 GPU 계산이 끝나면
(다음 layer hook 첫머리의 `torch.cuda.synchronize()` 반환) worker는 새 읽기를 시작하지 않고 진행 중이던 읽기 하나만 끝낸다.
시간 예산·chunk 시간 상수·절반 규칙은 쓰지 않는다. 출처 대조표 `docs/hyperinfer_port_contract.md`는 이 구현에 맞게 고쳤다.
HyperInfer만 다시 쟀고 나머지는 48절과 같이 짝지었다.

gate(4장 → DEV 40장): 출력 = IMPRESS, prefetch 끔 ≡ IMPRESS, 읽지 않은 KV를 쓰지 않음, 반복 동일, 멈춤 확인(계산 종료 뒤 시작된
prefetch 읽기 0건), 기존 테스트 새 실패 없음이 두 모델 모두 PASS였다. 본실험 생성 토큰은 두 모델 모두 기존 IMPRESS 기록과 12,183/12,183 같았다.

| 모델 | HyperInfer TTFT (48절) | HyperInfer / IMPRESS [95% CI] (같은 실행 DEV) | Ours / HyperInfer | prefetch 대기 / 요청 | prefetch로 채운 비율 | layer당 계산 시간 | 창 안에 끝난 읽기 / layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-NeXT | 727.6 ms (777.0) | 1.003 [0.998, 1.009] (0.990) | 0.205 | 0.63 ms | 1.6% | 1.90 ms | 0.39 |
| LLaVA-1.5 | 235.4 ms (254.9) | 0.961 (0.974) | 0.314 | 0.21 ms | 19.4% | 1.64 ms | 2.02 |

기다림은 사라졌지만, layer당 계산 시간 안에서 probe key 읽기(NeXT 1.25 ms, 1.5 0.50 ms)가 대부분을 차지해 미리 읽히는 KV가
적다. 그래서 HyperInfer는 IMPRESS와 거의 같다(LLaVA-1.5에서 4% 빠름).

[보고서](results/mt_gqa_hyperinfer_v2_20261125T034057Z/report.md), [실험 계약](docs/mt_gqa_hyperinfer_v2_contract.md),
[출처 대조표](docs/hyperinfer_port_contract.md), 코드 `mmimpress/hyperinfer_v2.py`, `scripts/228`(실행), `229`(gate), `230`(보고서),
`scripts/run_mt_gqa_hyperinfer_v2.sh`(드라이버), 테스트 `tests/test_hyperinfer_v2.py`.

## 50. ConvBench에서 HyperInfer 측정 (2026-11-25)

49절의 HyperInfer 구현과 설정을 그대로 46절 ConvBench 실행기에 연결해 HyperInfer만 두 모델에서 쟀다
(모델당 577대화 × 3턴 = 1,731요청). ReComp·FullLoad·IMPRESS·Ours는 46절 기록과 같은 대화·같은 턴끼리 짝지었다.
ConvBench는 이미지 뒤 텍스트가 길어 layer당 계산 시간이 MT-GQA보다 길다. 측정 보고이며 판정은 두지 않는다.

gate(4대화 → 파일럿 40대화, 모델마다): 출력 = 같은 세션의 IMPRESS(생성 토큰과 layer별 선택 집합), prefetch 끔 ≡ IMPRESS
(토큰·읽기 바이트·읽기 호출 수), 읽지 않은 KV를 쓰지 않음, 반복 동일, 멈춤 확인, 파일럿 토큰 = 기존 IMPRESS 기록, 기존 테스트
새 실패 없음이 두 모델 모두 PASS였다. 드라이버는 재시작 없이 6.0시간 돌았고 실패한 요청은 0건, 장치 쓰기량은 2.037 TB(예상 2.026 TB)다.
본실험 생성 토큰이 두 모델 모두 기존 IMPRESS 기록과 1,731/1,731 같아서, 품질 점수는 IMPRESS의 판정을 그대로 썼다
(LLaVA-NeXT 평균 8.03, LLaVA-1.5 4.27).

| 모델 | HyperInfer TTFT (T2–T3) | HyperInfer / IMPRESS [95% CI] | 같은 실행 파일럿 | Ours / HyperInfer | HyperInfer / FullLoad | layer당 계산 시간 (MT-GQA) | prefetch로 채운 비율 (MT-GQA) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-NeXT | 741.7 ms | 0.928 [0.923, 0.933] | 0.948 [0.932, 0.963] | 0.296 | 1.356 | 3.35 ms (1.90) | 12.6% (1.6%) |
| LLaVA-1.5 | 250.7 ms | 0.944 [0.940, 0.948] | 0.945 [0.927, 0.964] | 0.367 | 1.901 | 1.98 ms (1.64) | 26.4% (19.4%) |

이미지 뒤 텍스트가 길수록 prefetch로 채우는 비율이 늘었다. LLaVA-NeXT에서 텍스트 길이 사분위 Q1(평균 97토큰)은
HyperInfer/IMPRESS 0.955, Q4(평균 654토큰)는 채운 비율 31.2%, 0.886이었다. 총 읽기 바이트는 IMPRESS 대비 ×1.0006 / ×1.0018이고
읽기 호출 수는 2.5배 / 2.1배다. 4096 토큰을 넘는 입력은 LLaVA-NeXT T3에서 8건이었고 모두 정상 처리됐다.
HyperInfer는 11/25, 나머지 네 방법은 11/22–24 실행이라 날짜 간 변동이 섞인다. 파일럿에서 다시 잰 IMPRESS는 기존 대비 1.004 / 0.979였다.

[보고서](results/convbench_hyperinfer_20261125T131424Z/report.md), [실험 계약](docs/convbench_hyperinfer_contract.md),
코드 `scripts/convbench_hyperinfer_common.py`, `scripts/231`(실행; `217` 실행기에 `228`·`225`의 HyperInfer 요청 함수를 연결),
`232`(gate), `233`(보고서), `234` + `scripts/run_convbench_hyperinfer.sh`(드라이버), `235`(판정; 이번에는 쓰지 않음),
테스트 `tests/test_convbench_hyperinfer.py`. 대화별 답(`<model>/<phase>/conversations/`)을 함께 올렸다.
