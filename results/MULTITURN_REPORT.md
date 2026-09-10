# Multi-turn visual-KV SSD reuse 실험 보고서

실험일: 2026-09-09  
모델: `llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4, eager attention, greedy decoding  
시스템 조건: visual chunk 64 tokens, `max_new_tokens=16`, cold page cache, seed 1234

이 보고서는 기존 GQA/VQAv2/TextVQA 산출물을 수정하지 않고 별도
`runs/{visdial,mmdu}_multiturn` 및 `results/{visdial,mmdu}_multiturn` 경로에서 수행한
결과만 다룬다. VisDial 시스템 측정은 완료됐지만, MMDU는 사전 correctness gate를
통과하지 못했으므로 method별 SSD/TTFT/quality 본 실험을 실행하지 않았다.

## A. 기존 구조 분석

기존 구현은 이미지별로 `[system tokens | expanded single-image block]`의 prefix KV를
한 번 계산해 SSD에 token-major K/V 파일로 저장한다. 요청 시에는 이미지 뒤의 text
suffix만 토큰화한다. `ReComp`는 pixels부터 재계산하고, `FullLoad`는 visual KV 전체를
읽으며, `SparseVLM`은 layer hook에서 query-dependent 선택·읽기를 수행한다.
`static_diverse_chunk`는 미리 만든 image-static chunk score와 diversity metadata로
forward 전에 chunk를 고르고 필요한 token-major spans 및 구조적 row-separator sidecar만
`pread`한 뒤 full-length GPU `PrefixCache`의 해당 위치에 scatter한다.

True TTFT 경계는 기존 schema-v2 정의를 유지했다.

```text
request start
→ selector
→ SSD pread
→ cache reconstruction/scatter
→ prompt prefill
→ first output token argmax/materialization
→ CUDA synchronize
```

입력 processor/tokenization, H2D 준비와 cold page-cache eviction은 시작점 밖이다. 첫
token 이후 token 2…N은 `decode_ms`, 최종 동기화까지는 `e2e_ms`이며, 각 raw row에서
`e2e_ms ≈ ttft_ms + decode_ms`를 검증했다.

VisDial은 동일 이미지가 항상 고정된 선두 prefix이므로 이 abstraction에 정확히 맞는다.
MMDU는 새 이미지가 이전 image/text 뒤에 나타난다. 상위 transformer layer의 그 이미지
K/V는 이전 causal context와 절대 position에 의존하므로, 이미지 A와 B의 독립 prefix
KV를 단순 tensor-concat하는 것은 올바르지 않다.

## B. Multi-turn을 위해 변경한 부분

| 파일 | 변경 내용 |
|---|---|
| `mmimpress/model.py` | 완성 prompt용 `encode_prompt()`와 순서를 보존하는 복수 visual-token span 검출을 추가했다. 기존 single-image 경로는 strict single-span check를 유지한다. |
| `mmimpress/serve.py` | 외부 multi-turn suffix를 기존 store/server에 연결하고 true-TTFT 경계를 보존했다. Static+Diverse 경로의 불필요한 history embedding을 제거했다. |
| `mmimpress/multiturn.py` | VisDial/MMDU canonical schema, deterministic selection, gold-history prompt, `<ImageHere>`/active-image 매핑 및 quality helpers를 추가했다. |
| `scripts/06_build_static.py` | 격리된 run별 static build summary 출력을 추가했다. |
| `scripts/13_build_multiturn_index.py` | seed가 기록되는 deterministic canonical index builder와 overwrite 보호를 추가했다. |
| `scripts/14_eval_multiturn.py` | VisDial 전용 5-arm 시스템 runner를 추가했다. correctness가 확인되지 않은 MMDU SSD 실행은 명시적으로 거부한다. |
| `scripts/15_run_multiturn_eval.py` | build→importance reorder→static metadata→eval→analysis 단계, store fingerprint와 safe resume을 구현했다. |
| `mmimpress/multiturn_results.py` | overall/by-turn/by-active-image/by-dialog 집계, 결과 matrix·timing·budget 검증을 추가했다. |
| `scripts/16_analyze_multiturn.py` | 표, hypothesis checks 및 turn/active-image 그래프를 생성한다. |
| `scripts/17_validate_mmdu_cache.py` | 공식 image order/resize를 보존한 full recompute 대 native append-only `DynamicCache` progressive correctness gate를 추가했다. |
| `scripts/18_visdial_official_quality.py` | 100 candidate 조건부 likelihood와 MRR/R@K/Mean Rank/NDCG를 계산하는 quality-only runner를 추가했다. |
| `scripts/19_analyze_mmdu_feasibility.py` | 실제 cache tensor geometry에 기반한 MMDU context/capacity 분석을 추가했다. |
| `tests/test_multiturn.py`, `tests/test_visdial_official_quality.py` | history/leakage/budget/timing/completeness와 candidate branch/rank/NDCG/resume 회귀 검사를 추가했다. |

CPU 회귀 테스트는 최종적으로 20/20 PASS다.

## C. VisDial workload

- VisDial v1.0 validation, 100 dialogs / 100 images / 1,000 evaluated turns
- dialog마다 caption + 10 rounds; 매 turn active image는 동일한 1장
- seed 1234, dense round가 각 turn에 10개씩 오도록 deterministic stratification
- canonical index SHA-256:
  `8c3dd7e983cb39e61d26362a0353b86ac84845bd7537a6331078ab7707777383`
- 모든 arm이 동일한 caption, gold teacher-forced history, current question을 사용
- image KV build 100회, static metadata build/load 100회: 각각 dialog당 정확히 1회
- caption-only pre-dialog calibration을 사용했고 evaluation/future turn 질문·답변 사용은 0회
- history tokens 평균은 turn 1의 18.33에서 turn 10의 172.60으로 단조 증가
- 1,000 requests × 5 arms = 5,000 raw rows; completeness 및 cross-method identity PASS

100-image visual-KV payload는 116.943 GB다. static builder의 component timer는
이미지당 평균 CLIP 42.21 ms + metadata 556.86 ms = 599.07 ms였고, 10 turns에
단순 상각하면 59.91 ms/evaluated turn이다. 이는 전체 process wall-clock이 아니며
importance reorder/calibration과 `static.pt` 최종 serialization 등 timer 밖 작업이
있다. 이 timed component는 online TTFT에 포함하지 않는다. 참고로 59.91 ms만을
분석적으로 더해도 SD25/50의
amortized 값은 397.15/618.40 ms로, FullLoad 대비 각각 40.30%/7.04% 짧다. 이것은
실측 TTFT가 아니라 비용 상각 계산이다.

## D. MMDU workload

공식 benchmark 전체 canonical index는 110 dialogs / 1,645 turns / 421 image records
(resolved path 기준 419 unique)다. 한 dialog에는 2–20 images가 있고, turn별 active
images는 평균 3.2498, 최대 20이다. `<ImageHere>` 등장 순서대로 새 이미지를 추가하고
gold history를 누적하는 local correctness representation을 만들었다.

실제 GPU correctness run의 cache shape/dtype에서 얻은 geometry는 다음과 같다.

- 1 image: 1,176 visual tokens
- 1 visual token: 524,288 B (`32 layers × K/V × 32 heads × 128 × bf16`)
- 1 image: 616,562,688 B (0.616563 GB)
- 최대 20-image dialog: 12,331,253,760 B (12.331254 GB)
- 전체 421 image IDs: 259.572892 GB; 419 unique paths: 258.339766 GB

LLaVA/Vicuna 4,096-token 한계에서 `prompt + 16`이 들어오는 turn은 340/1,645이고,
1,305/1,645는 초과한다. 전체 turn이 들어오는 dialog는 1/110 (`mmdu:69`)뿐이다.
전체 길이는 official 336×336 resize를 반영한 tokenizer projection이며 실제 processor
sequence와의 exact cross-check는 아래 correctness subset 6 turns로 한정된다.

## E. Correctness validation

### VisDial

`validation.json`의 모든 검사가 PASS했다: 이미지 KV/static metadata 1회 build,
10-turn store reuse, history 단조 증가, method 간 동일 history/suffix/active image,
future-turn leakage 없음, FullLoad exact bytes, Static budget 준수, 5,000-key
completeness, `TTFT < E2E`, `E2E ≈ TTFT + decode`.

### MMDU

사전 고정한 `mmdu:70`, `mmdu:35`의 첫 3 turns씩을 full recomputation과 native
append-only `DynamicCache`로 비교했다. 독립 image-prefix SSD store나 KV concat은
사용하지 않았다.

| Dialog | Turn | New / active images | Prompt / reserved tokens | Max / mean absolute first-logit diff |
|---|---:|---:|---:|---:|
| `mmdu:70` | 1 | 1 / 1 | 1,210 / 1,226 | 0 / 0 |
|  | 2 | 1 / 2 | 2,683 / 2,699 | 0.125 / 0.014577 |
|  | 3 | 0 / 2 | 3,020 / 3,036 | 0.1328125 / 0.021940 |
| `mmdu:35` | 1 | 1 / 1 | 1,214 / 1,230 | 0 / 0 |
|  | 2 | 0 / 1 | 1,838 / 1,854 | 0.15625 / 0.019001 |
|  | 3 | 1 / 2 | 3,744 / 3,760 | 0.15625 / 0.017113 |

Token sequence, visual spans/image order, cache lengths, greedy first token, generated token
IDs와 response는 모두 6/6 일치했다. 그러나 실행 전에 고정한 tolerance
`max_abs ≤ 0.125 AND mean_abs ≤ 0.01`은 2/6만 통과했고, 관측 최댓값은 각각
0.15625와 0.021940이었다. 따라서
`append_only_dynamic_cache_gate_passed=false` 및
`static_diverse_mmdu_gate_passed=false`다. 공식 `[INST]` + generated-history protocol
equivalence도 주장하지 않는다. 지시대로 이 지점에서 MMDU full method run을 중단했다.

## F. 결과표

### VisDial 100-dialog system run

아래 quality는 reference answer와의 normalized generative match 보조지표이며 공식
VisDial 점수가 아니다.

| Method | Budget | Aux quality | TTFT mean / p50 / p95 (ms) | Decode / E2E mean (ms) | SSD MB/request | SSD/Full | Selector (ms) | Logical visual ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | — | 0.452 | 526.4 / 535.5 / 582.9 | 32.4 / 558.8 | 0.0 | 0% | — | — |
| FullLoad | 100% | 0.452 | 665.2 / 662.5 / 785.3 | 36.9 / 702.1 | 1,169.4 | 100% | — | 100% |
| SparseVLM | 25% | 0.447 | 718.4 / 730.3 / 845.1 | 42.9 / 761.3 | 912.0 | 78.0% | hook-interleaved | 24.6% |
| Static+Diverse | 25% | 0.436 | 337.2 / 337.1 / 384.7 | 32.3 / 369.5 | 301.8 | 25.8% | 15.83 | 24.3% |
| Static+Diverse | 50% | 0.442 | 558.5 / 562.3 / 623.3 | 32.6 / 591.1 | 594.7 | 50.9% | 22.44 | 49.0% |

Static+Diverse 25%는 FullLoad 대비 TTFT 49.30%, SSD bytes 74.20%를 줄였고,
SparseVLM 대비 각각 53.06%, 66.91%를 줄였다. aux quality delta는 FullLoad 대비
-0.016이다. 50%는 FullLoad 대비 TTFT 16.04%, SSD 49.15%를 줄였고 aux quality
delta는 -0.010이다. SD25는 ReComp보다도 TTFT가 35.93% 짧았지만, SD50은
ReComp보다 6.10% 길었다.

FullLoad/SparseVLM의 hook interval에는 prefill·I/O·scatter가 중첩되므로 배타적
component 합으로 해석하지 않는다. Static+Diverse의 hook-free 분해에서 25%의 평균
scatter/pure-prefill은 52.27/55.42 ms, 50%는 89.42/56.81 ms였다.

### VisDial candidate-ranking quality-only smoke

2 dialogs / 20 rounds / 100 candidates per round의 작은 validation smoke다. Sparse
retrieval은 20 rounds, dense NDCG는 두 dialog의 dense-annotated 2 rounds만 사용했다.

| Method | MRR | R@1 | R@5 | R@10 | Mean Rank | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.6164 | 0.550 | 0.700 | 0.800 | 9.45 | 0.4020 |
| FullLoad | 0.6151 | 0.550 | 0.700 | 0.800 | 9.55 | 0.4020 |
| SparseVLM 25% | 0.6124 | 0.550 | 0.650 | 0.800 | 9.85 | 0.4007 |
| Static+Diverse 25% | 0.5819 | 0.500 | 0.650 | 0.800 | 9.55 | 0.3793 |
| Static+Diverse 50% | 0.6150 | 0.550 | 0.700 | 0.800 | 9.70 | 0.4003 |

이 runner는 source-order 100 candidates에 대해 EOS를 포함한 unnormalized conditional
log-likelihood를 계산한다. 공식 starter의 metric 구현과 대조했고 rank permutation,
score/rank/GT/NDCG alignment, cross-method prompt identity, leakage 검사가 PASS했다.
그러나 sample이 너무 작아 통계적 결론이나 full-val EvalAI 공식 점수로 사용할 수
없다. 이 quality-only run에는 TTFT/SSD 측정이 없다. 100-dialog 전체 candidate run은
`100 × 10 × 100 × 5 = 500,000` method-candidate branches가 필요해 이번 단계에서는
실행하지 않았다.

### MMDU

Correctness gate가 닫혔으므로 ReComp/FullLoad/SparseVLM/Static+Diverse의 TTFT,
SSD read 및 official quality 결과는 없다. 위 feasibility 수치를 method 측정값으로
간주해서는 안 된다.

## G. Turn별 변화

| Turn | History tokens | FullLoad TTFT | SD25 TTFT / reduction | SD50 TTFT / reduction | SD25 selector |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.33 | 673.84 | 329.14 / 51.16% | 552.53 / 18.00% | 16.38 |
| 2 | 34.62 | 663.58 | 331.93 / 49.98% | 551.13 / 16.95% | 16.60 |
| 3 | 52.10 | 660.49 | 329.28 / 50.15% | 551.83 / 16.45% | 15.28 |
| 4 | 69.27 | 661.86 | 328.58 / 50.35% | 550.91 / 16.76% | 15.77 |
| 5 | 86.87 | 665.77 | 335.67 / 49.58% | 554.76 / 16.67% | 15.79 |
| 6 | 103.66 | 661.52 | 338.69 / 48.80% | 557.30 / 15.75% | 15.76 |
| 7 | 121.53 | 661.15 | 342.95 / 48.13% | 564.44 / 14.63% | 15.70 |
| 8 | 138.35 | 671.18 | 342.69 / 48.94% | 564.10 / 15.95% | 15.61 |
| 9 | 155.41 | 666.70 | 345.85 / 48.12% | 568.88 / 14.67% | 15.69 |
| 10 | 172.60 | 666.00 | 347.63 / 47.80% | 569.03 / 14.56% | 15.76 |

모든 값은 ms다. visual image와 store는 고정이고 gold text history만 길어진다.
SD25 selector는 15.28–16.60 ms로 거의 일정했고, SSD bytes도 method별로 모든 turn에서
정확히 일정했다. SD25의 이점은 10개 turn 모두 유지됐으나, pure text prefill이
커지면서 FullLoad 대비 감소율은 51.16%에서 47.80%로 조금 좁아졌다. SD50도 모든
turn에서 이점이 있었지만 18.00%에서 14.56%로 좁아졌다.

## H. Memory growth

VisDial에서는 active image가 항상 1이므로 visual working set이 turn에 따라 늘지
않는다. 이미지별 full visual KV는 0.768–1.535 GB, 평균 1.169 GB였고 같은 SSD store를
10 turns에서 재사용했다. 측정 장비는 RTX 4090 25.391 GB, system RAM 134.818 GB였고
VisDial run 시작 시 available RAM은 127.584 GB였다. Turn 1→10의 mean CUDA peak는
FullLoad 6.529→6.675 GB, SD25 6.529→6.677 GB, SD50 6.529→6.676 GB로 증가했다.
ReComp는 7.049→8.544 GB였으며, 전체 request 중 method별 관측 peak의 최댓값은
ReComp 10.61 GB였다. 이는 allocator 관측치이며 동시 live tensor working set과
동일한 지표는 아니다.

MMDU의 full visual working set projection은 active image 수에 선형으로 증가한다:
1 image 0.617 GB, 2 images 1.233 GB, 20 images 12.331 GB. 전체 per-ID cache
259.573 GB는 이 기기의 GPU 25.391 GB의 10.22배, RAM 134.818 GB의 1.93배다
(path-dedup 시 각각 10.17배/1.92배; 분석 시작 available RAM 126.605 GB).
이는 dataset 전체 image cache footprint이지 한 요청의 동시 working set은 아니다.
최대 dialog의 visual KV 12.331 GB 자체는 nominal GPU capacity보다 작지만 모델,
text KV 및 임시 tensor가 별도로 필요하다. 따라서 이 수치는 dataset-scale capacity
동기이며 실제 OOM이나 MMDU online serving 성능을 측정한 결과는 아니다.

중요하게 현재 `PrefixCache`는 선택하지 않은 row를 포함한 full-length GPU tensor를
할당한다. 따라서 Static+Diverse의 `selected_visual_kv_bytes` 감소는 실제 SSD/H2D
payload 감소이지만 GPU allocation 또는 attention sequence length 감소는 아니다.

## I. 실패 및 제약 사항

1. VisDial calibration은 caption-only, pre-dialog 1회다. 미래 turn leakage는 없지만
   기존 GQA의 question-dependent importance reorder와 calibration distribution이 다르다.
2. MMDU 후속 image KV에는 이전 image/text와 position dependency가 있어 현행 독립
   single-image store를 직접 조립할 수 없다. 가장 작은 올바른 확장은 contextual
   multi-image spans와 absolute/cache position을 보존하는 conversation-prefix store다.
3. MMDU 행동 출력은 6/6 일치했으나 사전 numeric logit tolerance가 실패했다.
   tolerance를 결과에 맞춰 완화하지 않았고 full experiment를 강행하지 않았다.
4. MMDU 1,645 turns 중 1,305가 4,096-token context를 초과한다. 임의 truncation은
   benchmark semantics를 바꾸므로 적용하지 않았다.
5. Local correctness prompt는 Vicuna/gold-history이고 공식 quality protocol의
   `[INST]`/generated-history equivalence는 검증되지 않았다.
6. VisDial system-run normalized match는 공식 metric이 아니다. Candidate-ranking
   결과도 2-dialog smoke이며 공식 server 제출이나 published baseline 재현이 아니다.
7. Candidate likelihood는 EOS 포함 unnormalized sum이라 짧은 답을 선호할 수 있다.
8. FullLoad/SparseVLM hook timing은 component가 중첩된다. Static+Diverse만
   selector/read/scatter/prefill의 배타적 해석이 가능하다.
9. MMDU full system TTFT/SSD/quality 수치는 의도적으로 생성하지 않았다.

## J. 연구적으로 얻은 결론

| 주장 | 판정 | 실제 증거 범위 |
|---|---|---|
| Multi-turn에서 visual context의 반복·누적으로 SSD-backed reuse 필요성이 증가한다. | **부분 지원** | VisDial에서 image KV/static metadata를 한 번 만들고 10 turns 재사용했다. 단 ReComp가 FullLoad보다 빨라 SSD의 runtime 필요성 자체를 입증한 것은 아니다. MMDU는 dataset-store capacity 동기를 보였지만 최대-dialog cache는 nominal GPU보다 작고 runtime 필요성/효과를 측정하지 못했다. |
| Static+Diverse가 FullLoad보다 적게 읽어 SSD I/O와 TTFT를 줄인다. | **VisDial에서 지원** | SD25: SSD -74.20%, TTFT -49.30%; SD50: SSD -49.15%, TTFT -16.04%. aux quality delta는 각각 -0.016/-0.010이며 작은 official-style smoke도 함께 보고했다. |
| 이 이점이 conversation turn 및 active image 수 증가에도 유지된다. | **turn축 지원, active-image축 미검증** | 고정 single-image VisDial에서 T1–T10 내내 TTFT 이점과 일정한 selector/SSD I/O를 확인했다. MMDU method run이 없으므로 multi-image scaling 주장은 할 수 없다. |

따라서 이번 결과가 직접 지지하는 핵심 결론은 다음과 같다.

> 고정된 한 이미지를 10-turn gold-history conversation에서 반복 참조할 때,
> Static+Diverse 25/50%는 image-static metadata를 한 번만 만들고 재사용하면서
> FullLoad보다 실제 SSD payload와 true TTFT를 줄였다. 이 결과를 progressive
> multi-image MMDU나 GPU-memory/attention-length 절감으로 일반화할 근거는 아직 없다.

## 산출물

- VisDial system: `results/visdial_multiturn/main_seed1234_true_ttft/`
- VisDial official-style quality smoke:
  `results/visdial_multiturn/official_quality_smoke_seed1234/`
- MMDU correctness: `results/mmdu_multiturn/correctness_progressive_seed1234_v2/`
- MMDU feasibility: `results/mmdu_multiturn/feasibility_seed1234/`
- Canonical indices:
  `data/visdial_v1.0/subsets/main_seed1234/index.json`,
  `data/mmdu/subsets/full_seed1234/index.json`

외부 protocol 참고:

- 연구 repository: <https://github.com/HJ990412/MLLM_visionzip>
- VisDial v1.0 data: <https://visualdialog.org/data>
- VisDial official starter metrics:
  <https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/metrics.py>
- MMDU repository: <https://github.com/Liuziyu77/MMDU>
- MMDU LLaVA-NeXT generation reference:
  <https://github.com/Liuziyu77/MMDU/blob/main/model_generation/LLaVa_next_gen_ans.py>
