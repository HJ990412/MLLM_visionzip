# LLaVA-1.5-7B 포팅 + MT-GQA HOLD (ReComp / U100 / U50 / U25 / PYR-b20 packed)

Run `runs/llava15_mt_gqa_20261112T062708Z`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `6b727e7749cc9989…`, gate·smoke·main 전에 freeze). 단위: MB = 10^6 B, ms, 정확도 차이는 pp. followup = T2–T3. CI = 이미지 단위 paired bootstrap 95% (10,000회, seed 1234).

상태: **FINAL** — main 60,915 / 60,915 요청(arm당 12,183), 이미지 398 / 398, 실패·스킵 0, 독립 감사 PASS, T1 arm 간 토큰 일치 4,061 / 4,061 (PASS).

## 이 실험에서 반드시 알아야 할 것

- **LLaVA-1.5-7B는 이미지당 시각 토큰이 576개로 고정이고 KV는 약 302 MB(301,989,888 B, FP16, 32 layer × K·V × 576행 × 8,192 B)이며 separator가 없다.** 전처리는 HF 기본 processor 그대로다(CLIPImageProcessor: 짧은 변 336 resize 후 336×336 중앙 자르기). 원 LLaVA-1.5 저장소는 정사각형 패딩이라 공개 GQA 수치와 조건이 다르다.
- **모든 packed arm은 같은 중요도 순서(CLIP 끝에서 두 번째 layer의 CLS→patch attention head 합, 내림차순 stable 정렬)와 같은 I/O 경로(이미지당 파일 1개, hit당 preadv 1회, dense 버퍼 + layer별 mask)를 쓰고, 비율과 layer별 자르는 지점만 다르다.** 질문은 선택에 쓰지 않는다. 데이터로 무엇을 맞추거나 고르지 않았다.
- **PYR-b20은 PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 attention 기반 token 선택은 쓰지 않았다.** 총 바이트는 U25와 같다.
- **LLaVA-NeXT의 100% packed 지점은 측정된 적이 없다.** (B)의 LLaVA-NeXT 참고값은 다른 I/O 경로(파일 분리)의 값이다.
- **MT-VQA·Qwen·GLM·LLaVA-NeXT는 이번에 실행하지 않았다.** 참고값은 선행 run의 기록이다. 절대 TTFT는 실행 간 비교하지 않는다.
- 주 운영점은 25%로 실행 전에 고정했다. 50%·100%는 곡선의 점이다.

## (D) 곡선 — HOLD followup

| Arm | 비율 | layer당 행 | Followup % [95% CI] | vs ReComp Δpp [CI] | TTFT mean / p50 / p95 ms | TTFT / ReComp | SSD MB / followup |
|---|---|---|---|---|---|---|---|
| ReComp | – | – | 65.279 [63.898, 66.675] | – | 126.86 / 135.29 / 137.44 | 1.000× | 0.00 |
| U100_packed | 100% | 576 | 65.304 [63.907, 66.722] | +0.025 [-0.235, +0.295] | 209.32 / 208.95 / 218.73 | 1.650× | 301.99 |
| U50_packed | 50% | 288 | 64.959 [63.582, 66.364] | -0.320 [-0.817, +0.171] | 126.77 / 126.37 / 132.62 | 0.999× | 150.99 |
| U25_packed | 25% | 144 | 63.322 [61.930, 64.748] | -1.958 [-2.693, -1.239] | 86.38 / 86.27 / 89.83 | 0.681× | 75.50 |
| PYR-b20_packed | 25% | 281…7 (합 4,608) | 63.371 [62.005, 64.770] | -1.908 [-2.626, -1.204] | 86.36 / 86.02 / 90.38 | 0.681× | 75.50 |

Δ는 (행의 arm − ReComp)이다. CI가 0을 포함하면 '차이 미입증'이며 동등하다는 뜻이 아니다. T1은 모든 arm이 full pixel이고, 차이는 generated history가 갈라지는 효과를 포함한다.

![accuracy vs TTFT](fig_accuracy_vs_ttft.png)

## (A) 주 운영점 — U25_packed vs ReComp

| 항목 | 값 |
|---|---|
| Followup 정확도 | U25 63.322% vs ReComp 65.279% |
| Δ pp [95% CI] | -1.958 [-2.693, -1.239] — ReComp보다 낮음 (CI 0 제외) |
| 원래 성능 대비 | 97.00% |
| TTFT mean / p50 / p95 ms | U25 86.38 / 86.27 / 89.83  vs  ReComp 126.86 / 135.29 / 137.44 |
| TTFT 비율 (mean) [95% CI] | 0.681× [0.679, 0.683] (단축 31.9%); p50 0.638×, p95 0.654× |
| SSD MB / followup | 75.50 |

## (B) KV 크기 축 지표 (보고, 판정 아님)

| 지표 | 정의 | LLaVA-1.5-7B (이 실행) [95% CI] | Qwen2.5-VL (같은 packed 경로) | LLaVA-NeXT-7B (v1, 파일 분리 경로) |
|---|---|---|---|---|
| R100 | TTFT(U100) / TTFT(ReComp) | **1.650** [1.645, 1.655] | 0.462 | 1.39 |
| S25 | TTFT(U25) / TTFT(U100) | **0.413** [0.411, 0.414] | 0.892 | 0.34 |
| S50 | TTFT(U50) / TTFT(U100) | **0.606** [0.604, 0.608] | – | – |
| 이미지당 KV (100%) |  | 302 MB | 약 22 MB | 약 1,174 MB |

사전 기록 가설: R100은 0.462와 1.39 사이 → **틀림** (관측 1.650). S25는 0.34와 0.892 사이 → **맞음** (관측 0.413). Qwen2.5-VL 값은 같은 packed 경로(22 MB/이미지). LLaVA-NeXT-7B 값은 v1 run의 파일 분리 경로(100% = FullLoad 65 pread, 25% = unpacked)이며 LLaVA-NeXT의 100% packed 지점은 측정된 적이 없어 직접 비교가 아니다. 절대 TTFT는 실행 간 비교하지 않는다.

## (C) layer 배분 판정 — PYR-b20_packed − U25_packed

| 판정 | Δ pp [95% CI] | CI의 0 | SSD MB/followup (PYR / U25, ≤1.02×) | TTFT 평균 ms (PYR / U25, ≤1.05×) | PYR만 / U25만 / 둘 다 / 둘 다 오답 | 예측 일치율 |
|---|---|---|---|---|---|---|
| **NO-GO** | +0.049 [-0.472, +0.570] | 포함 (차이 미입증) | 75.497 / 75.497 (OK) | 86.36 / 86.38 = 1.000× (OK) | 197 / 193 / 4950 / 2782 | 93.07% |

규칙(실행 전 고정, LLaVA-NeXT·Qwen과 동일): GO = Δ ≥ +1.0 pp & CI 0 제외 & SSD MB ≤ U25 × 1.02 & TTFT 평균 ≤ U25 × 1.05; PARTIAL = Δ ≥ +0.5 pp이고 CI 0 포함, 또는 GO에서 MB/TTFT 중 정확히 하나 위반; 그 외 NO-GO. 같은 규칙의 선행 결과: LLaVA-NeXT-7B GO (+1.244 pp, PYR-beta-raw vs U_packed), Qwen2.5-VL NO-GO (-0.899 pp).

## 정확도 (turn별) — (F) T1 정상성 포함

| Arm | T1 % | T2 % | T3 % | All % (정답/N) | Followup % (정답/N) |
|---|---|---|---|---|---|
| ReComp | 59.394 | 64.861 | 65.698 | 63.318 (7714/12183) | 65.279 (5302/8122) |
| U100_packed | 59.394 | 64.664 | 65.944 | 63.334 (7716/12183) | 65.304 (5304/8122) |
| U50_packed | 59.394 | 64.516 | 65.403 | 63.104 (7688/12183) | 64.959 (5276/8122) |
| U25_packed | 59.394 | 63.039 | 63.605 | 62.013 (7555/12183) | 63.322 (5143/8122) |
| PYR-b20_packed | 59.394 | 62.965 | 63.777 | 62.045 (7559/12183) | 63.371 (5147/8122) |

T1은 모든 arm이 full pixel이다. arm 간 T1 생성 토큰 일치: 4,061 / 4,061 요청 → PASS. T1 정확도는 공개 LLaVA-1.5-7B GQA 수치(61.9)와 양자화(NF4)·전처리(중앙 자르기)·멀티턴 프롬프트 구성·398장 재구성 표본이 달라 직접 비교가 아니며 판정에 쓰지 않는다.

| 비교 (A − B) | Followup Δpp [95% CI] | All Δpp [95% CI] | T2 Δpp | T3 Δpp | A만 / B만 정답 (followup) | 예측 일치율 (followup) |
|---|---|---|---|---|---|---|
| U25_packed − ReComp | -1.958 [-2.693, -1.239] | -1.305 [-1.795, -0.826] | -1.822 | -2.093 | 288 / 447 | 87.55% |
| U50_packed − ReComp | -0.320 [-0.817, +0.171] | -0.213 [-0.545, +0.114] | -0.345 | -0.295 | 153 / 179 | 94.45% |
| U100_packed − ReComp | +0.025 [-0.235, +0.295] | +0.016 [-0.157, +0.197] | -0.197 | +0.246 | 49 / 47 | 98.53% |
| PYR-b20_packed − ReComp | -1.908 [-2.626, -1.204] | -1.272 [-1.751, -0.803] | -1.896 | -1.921 | 256 / 411 | 88.04% |
| PYR-b20_packed − U25_packed | +0.049 [-0.472, +0.570] | +0.033 [-0.315, +0.380] | -0.074 | +0.172 | 197 / 193 | 93.07% |
| U50_packed − U25_packed | +1.638 [+0.986, +2.314] | +1.092 [+0.658, +1.542] | +1.477 | +1.798 | 361 / 228 | 90.14% |
| U100_packed − U50_packed | +0.345 [-0.145, +0.836] | +0.230 [-0.097, +0.558] | +0.148 | +0.542 | 185 / 157 | 94.42% |
| U100_packed − U25_packed | +1.982 [+1.249, +2.726] | +1.322 [+0.833, +1.818] | +1.625 | +2.339 | 446 / 285 | 87.66% |

## Followup 성능과 I/O

| Arm | TTFT mean / p50 / p95 ms | E2E mean / p95 ms | TTFT / ReComp | TTFT / U100 | SSD MB | 읽기 호출 (pread / preadv) | peak GPU alloc MB (평균) | T1 TTFT mean ms |
|---|---|---|---|---|---|---|---|---|
| ReComp | 126.86 / 135.29 / 137.44 | 154.50 / 192.30 | 1.000× | 0.606× | 0.00 | 0 / 0 | 4579.7 | 109.25 |
| U100_packed | 209.32 / 208.95 / 218.73 | 237.87 / 268.77 | 1.650× | 1.000× | 301.99 | 0 / 1 | 4808.2 | 109.23 |
| U50_packed | 126.77 / 126.37 / 132.62 | 155.32 / 185.85 | 0.999× | 0.606× | 150.99 | 0 / 1 | 4808.2 | 109.21 |
| U25_packed | 86.38 / 86.27 / 89.83 | 114.92 / 145.52 | 0.681× | 0.413× | 75.50 | 0 / 1 | 4808.2 | 109.25 |
| PYR-b20_packed | 86.36 / 86.02 / 90.38 | 114.95 / 145.09 | 0.681× | 0.413× | 75.50 | 0 / 1 | 4808.2 | 109.26 |

## (E) 구간별 시간 (followup 평균 ms)

| Arm | 입력 준비 (prompt·tokenize / 이미지 전처리) | read (preadv 1회) | H2D·scatter·mask | cache 할당 등 | prefill | decode |
|---|---|---|---|---|---|---|
| ReComp | 4.55 (processor 전체) | – | – | – | 119.83 (vision + 전체 prefill) | 27.50 |
| U100_packed | 0.79 | 90.88 | 79.27 | 3.27 | 35.05 (suffix) | 28.39 |
| U50_packed | 0.72 | 45.84 | 41.94 | 3.24 | 34.98 (suffix) | 28.39 |
| U25_packed | 0.70 | 23.02 | 24.35 | 3.22 | 35.03 (suffix) | 28.38 |
| PYR-b20_packed | 0.70 | 23.44 | 23.92 | 3.23 | 35.02 (suffix) | 28.43 |

구간은 TTFT의 구성 요소지만 계측 경계가 달라 합이 TTFT와 정확히 같지는 않다. H2D와 scatter는 HOLD 경로에서 한 호출(`PrefixCache.write`)이라 분리하지 않았다. 아래는 gate fixture 10장(turn 3)에서 계측용 복제 경로로 잰 분해이며 **HOLD 측정이 아닌 진단값**이다(복제 결과는 production cache와 bitwise 동일).

| Vector | host FP16→BF16 변환 | H2D 복사 | GPU index write | mask | (참고) 같은 fixture의 production scatter | H2D MB |
|---|---|---|---|---|---|---|
| U100 | 53.94 | 17.20 | 2.60 | 1.76 | 77.13 | 301.99 |
| U50 | 27.92 | 9.44 | 2.24 | 1.68 | 41.03 | 150.99 |
| U25 | 13.65 | 5.87 | 2.03 | 1.63 | 24.24 | 75.50 |
| PYR-b20 | 13.95 | 5.55 | 2.00 | 1.65 | 23.58 | 75.50 |

## 저장과 setup (이미지당 1회)

| Arm | 보존 % | 파일 수 | store MB / 이미지 | persistence mean / p50 / p95 ms | writer ms (materialize·repack / write / fsync·rename) | 완료 hash ms | activation mean / p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| U100_packed | 100.0 | 5 | 304.65 | 495.8 / 492.0 / 536.4 | 338.0 (38.9 / 105.6 / 64.5) | 156.6 | 185.2 / 185.3 / 190.9 |
| U50_packed | 50.0 | 5 | 153.65 | 272.9 / 271.4 / 286.2 | 191.6 (39.8 / 54.3 / 31.8) | 80.2 | 96.0 / 95.9 / 100.6 |
| U25_packed | 25.0 | 5 | 78.15 | 157.0 / 155.8 / 165.3 | 112.8 (38.2 / 23.6 / 16.7) | 43.4 | 52.6 / 52.4 / 55.2 |
| PYR-b20_packed | 25.0 | 5 | 78.16 | 160.8 / 160.1 / 168.0 | 116.9 (38.8 / 26.3 / 17.0) | 43.1 | 52.9 / 52.6 / 55.2 |

persistence = Turn-1 capture에서 store-ready(fsync + atomic rename)까지 + 완료 봉인(전체 SHA256 1회). activation = 메타 검증 + payload SHA256 전체 검증 + pinned buffer 할당. 네 store는 각 arm 자신의 T1 capture에서 만들었고 순서표는 이미지마다 네 store가 동일하다(감사).

## 3-turn session

| Arm | setup ms / 이미지 | shared-stream mean / p50 / p95 ms | standalone-equivalent (DERIVED) mean / p50 / p95 ms |
|---|---|---|---|
| ReComp | 0.0 | 445.1 / 438.4 / 498.9 | 445.1 / 438.4 / 498.9 |
| U100_packed | 681.0 | 678.5 / 609.5 / 1285.9 | 1293.0 / 1287.6 / 1361.7 |
| U50_packed | 368.9 | 482.8 / 443.4 / 807.6 | 815.5 / 810.0 / 873.5 |
| U25_packed | 209.6 | 386.5 / 361.5 / 568.7 | 575.4 / 568.6 / 628.2 |
| PYR-b20_packed | 213.7 | 386.9 / 362.0 / 571.8 | 579.5 / 573.2 / 635.1 |

| A − B | accounting | Δ ms [95% CI] |
|---|---|---|
| U25_packed − ReComp | shared | -58.6 [-59.9, -57.2] |
| U25_packed − ReComp | standalone | +130.4 [+129.3, +131.5] |
| U50_packed − ReComp | shared | +37.8 [+35.7, +40.0] |
| U50_packed − ReComp | standalone | +370.4 [+368.8, +372.2] |
| U100_packed − ReComp | shared | +233.5 [+230.0, +237.3] |
| U100_packed − ReComp | standalone | +847.9 [+845.2, +850.7] |
| PYR-b20_packed − U25_packed | shared | +0.5 [-0.0, +0.9] |
| PYR-b20_packed − U25_packed | standalone | +4.0 [+3.1, +5.0] |

shared-stream은 이미지의 첫 dialogue에만 setup(persistence + activation)을 더한 실측 합이다. standalone-equivalent는 모든 dialogue에 setup을 더한 DERIVED 값이다.

## Gate

| Gate | 결과 | 내용 |
|---|---|---|
| LG1 runner 동치 | PASS | 새 runner recompute 경로 == HF generate(greedy, max 16), fixture 10장 × 3턴 토큰 전부 동일 |
| LG2 geometry·순서 | PASS | 시각 span 576개 연속 1구간, separator 0, CLS saliency (1, 576) 유한, 순서 = 0..575 순열, 질문을 바꿔도·다시 실행해도 순열 SHA 동일, LLM 호출 0 |
| LG3 capture 정확성 | PASS | T1 capture의 prefix KV == 질문 자리를 filler로 바꾼 같은 길이의 독립 forward(3종), 32 layer K/V bitwise. 문자 그대로의 짧은 prefix-only forward(진단): layer 0만 bitwise, layer ≥ 1은 최대 rel-L2 0.0147 |
| LG4 저장 정확성 | PASS | packed 파일의 layer slice == capture → FP16 → 순서표 앞 k_l행 (bitwise), 파일 크기 = 기대 바이트(separator 0 B), 변조 meta·SHA·행 수·잘린 파일 거부 |
| LG5 hit 경로 구현 | PASS | SSD 경로 / 같은 shape의 독립 in-memory 참조 / 반복 실행: 첫 logits bitwise 동일, 생성 ID 동일, layer별 가시 행 == 순서표 앞 k_l개 (mask bit 비교) |
| LG6 의미 검증 | PASS | 전체 길이 참조(576행 실제 값 + layer별 가시 집합 mask) vs production: FP32 첫 토큰 logits atol = rtol = 1e-4. U100 vs recompute 첫 토큰 일치 29 / 30 (96.7%, 기준 95%) |
| LG7 I/O | PASS | hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기, vision 0·score 0, 질문만 바꿔도 read range·선택 ID 불변 |
| LG8 CPU | PASS | 벡터 SHA·합, N = 576에서 Σ k_l = 32·ceil(r·576), layer_budget 기본값에 기대는 호출 0, fail-closed 검사 75건(negative fixture 포함) |
| LG9 회귀 | PASS | 기존 CPU 테스트 전체 통과(Ran 568 tests in 11.817s), GLM 기존 실패 3건 분리 보고, 기존 소스 무수정 |
| LG10 통합 | PASS | smoke dialogue 5 arm × 3턴 generated history 60 요청 + 독립 재감사 |
| Artifact protection before/after | PASS | SHA 4841 → 4841 files, kvstore inventory 16160, 다른 run inventory 1858839 |
| Deletion log | PASS | run-local scratch / gate store만 삭제, 기존 runs·results·stores 삭제 0 |

U100 vs recompute 첫 토큰 불일치(원인 기록): n204894 T2 (U100 top-2 logit 차 0.1250). U100은 T1의 prefix KV(FP16 왕복)에 suffix만 prefill하고 recompute는 전체를 prefill하므로 BF16 반올림이 shape에 따라 달라져 동률에 가까운 첫 토큰이 뒤집힐 수 있다.

진단(비 gate): 같은 가시 집합을 원래 token 순서로 배치한 참조와 production의 첫 토큰 logits 최대 차 0.2500, 첫 토큰 일치 119 / 120, 생성 ID 일치 119 / 120 (key 합산 순서에 따른 BF16 반올림).

## 실행 기록

- 실제 요청 수: main 60,915 / 60,915 (5 arm × 12,183), smoke 60 / 60, integration(LG10) 60 / 60. 실패·스킵 요청 0. 실패 attempt 기록 0건, 중단 기록 0건.
- main 소요: 3.20 h (첫 시작 → 완료, wall). freeze 시 예상 2.5–4.5 h.
- 삭제: main scratch store 398개 이미지 · 파일 7,960개 · 244.6 GB(정확한 SHA allowlist 후), gate store 10개 이미지. 기존 runs/results/stores 삭제 0.
- freeze amendment: 없음.
- 환경: 3.10.20, transformers 4.57.6, bitsandbytes 0.49.2, torch 2.5.1+cu121, 모델 revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.

## 프롬프트와 다르게 한 점

1. LG3: 프롬프트의 '독립적인 prefix-only forward'를 문자 그대로(길이 P의 짧은 forward) 구현하면 BF16 GEMM 반올림이 tensor shape에 의존해 layer ≥ 1에서 bitwise가 성립하지 않는다(개발 probe에서 확인, 저장소 선례 scripts/30·docs/recompute_kv25_ablation_contract.md). freeze 전에 계약에서 LG3 참조를 '질문 자리를 filler 토큰으로 바꾼 같은 길이의 독립 forward(3종)'로 정의했다. bitwise·전 layer·K/V 기준은 그대로다. 문자 그대로의 짧은 forward는 LG3-literal 진단으로 실행해 수치만 기록했다(gate 아님).
2. LG6: '전체 길이 참조'는 576행이 모두 실제 값인 cache에 layer별 가시 집합을 mask로만 적용한 독립 구현이며, 행 배치는 production과 같은 저장 순서다(기존 LLaVA G2와 같은 배치·같은 1e-4 기준). 원래 token 순서로 배치한 변형은 key 합산 순서가 달라 BF16 반올림 수준의 차이가 생기므로 진단으로만 기록했다(gate 아님).
3. (E)의 H2D와 scatter는 HOLD 경로에서 분리 측정하지 않았다. hit 경로는 LLaVA-NeXT와 같은 `PrefixCache.write`(FP16→BF16 변환 + H2D + index write가 한 호출)를 그대로 쓰므로 HOLD 수치는 'H2D·scatter·mask' 합산이다. 분해 수치는 gate fixture 10장에서 계측용 복제 경로로 잰 진단값이다(HOLD 측정이 아님).
4. HOLD 프롬프트는 프롬프트 1절의 단일 턴 템플릿이 아니라 '기존 멀티턴 형식'(LLaVA-NeXT HOLD 164와 같은 `Current question Q{t}` 형식)을 모든 턴에 썼다. 단일 턴 템플릿은 runner.prompt()에 기록만 했다.
5. PYR-b20 벡터 문서: PyramidKV run의 PYR-beta-raw json을 그대로 복사해 보관하고 ratio만 가져왔다(새로 계산하지 않음, ratio SHA 동일). 복사본의 clip 범위는 [0.0, 0.6]이지만 프롬프트 지시대로 모든 layer_budget 호출은 r_min=0.0, r_max=1.0을 명시한다. 최대 ratio가 0.4875라 k_l은 두 범위에서 같다(Phase 0에서 확인).
6. 실행 환경: conda env mllm_ft의 Python을 절대 경로로 호출했다(셸 PATH의 python은 다른 venv). transformers 4.57.6 / bitsandbytes 0.49.2 / torch 2.5.1+cu121.
7. 개발 rehearsal: 공식 freeze 전에 gate fixture 이미지로 코드 동작을 확인했다(runs/llava15_mt_gqa_dev_*; 측정이 아니며 보고 수치에 쓰지 않음).

## 한계와 NOT_MEASURED

- HOLD 경로에서 H2D 단독 시간(합산 측정; 분해는 fixture 진단)
- NAND/controller 수준 cold read, O_DIRECT (DONTNEED는 page-cache hint)
- 독립 GPU footprint
- LLaVA-NeXT 100% packed 지점
- MT-VQA / Qwen / GLM / LLaVA-NeXT (이번에 실행하지 않음)
- FullLoad(파일 분리) arm
- PyramidKV의 attention 기반 token 선택
- DONTNEED는 OS page-cache hint이며 NAND cold나 O_DIRECT가 아니다.
- CI가 0을 포함한다는 것은 동등성 증명이 아니다(이미지 398개 cluster bootstrap). 곡선과 (B)·(C)는 HOLD에서 관찰한 값이며 다른 데이터에서 확인되지 않았다.
- packed serving은 dense GPU 버퍼를 유지하므로 GPU 메모리 절감은 주장하지 않는다.
