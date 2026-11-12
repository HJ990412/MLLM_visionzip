# Qwen2.5-VL Layer-wise Budget — PyramidKV(β=20) + PackedPrefixStore, MT-GQA HOLD

Run `runs/qwen_layer_budget_pyr_20261111T075124Z`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `dade82a1e88ab674…`, gate·smoke·main 전에 freeze). 설계 [design_ragged_cache.md](design_ragged_cache.md). 단위: MB = 10^6 B, ms, 정확도 차이는 pp. followup = T2–T3.

## 판정

**NO-GO** — 주 판정(사전 고정): Δ = followup 정확도(PYR-b20_packed − U_packed) = **-0.899 pp** [-1.490, -0.302] (이미지 단위 paired bootstrap 95% CI, 10,000회, seed 1234).

| 조건 | 값 | 기준 | 충족 |
|---|---|---|---|
| Δ followup (pp) | -0.899 | ≥ +1.0 (GO) / ≥ +0.5 (PARTIAL) | 아니오 |
| 95% CI가 0 제외 | [-1.490, -0.302] | 0 제외 (양의 방향) | 아니오 — 0 제외이지만 음의 방향 |
| SSD MB/followup | 6.371 vs U 6.371 | ≤ U × 1.02 | 예 |
| TTFT 평균 (ms) | 46.607 vs U 46.614 (1.000×) | ≤ U × 1.05 | 예 |
| U_packed ≡ B (토큰 단위) | 12,183 / 12,183 | 100% | 예 |
| 요청 완주·실패 0·감사 | 73,098 / 73,098, 실패·스킵 0, 감사 PASS | 전부 | 예 |

CI 전체가 0보다 작다: Δ = -0.899 pp [-1.490, -0.302]. 이 HOLD에서 PYR-b20은 같은 바이트의 균등 배분(U_packed)보다 followup 정확도가 낮다. 개선이 아니라 저하이다.

이 실험에서 반드시 알아야 할 것:

- **PYR-b20 = PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 하한이 없다.** 원 PyramidKV의 attention 기반 token 선택은 쓰지 않고 image-only saliency 순서에서 layer별 자르는 지점만 따랐다.
- **Qwen은 N이 작아(260–529) β=20의 마지막 layer가 3–7행이다.** (LLaVA는 17–36행)
- **모든 packed arm은 같은 바이트·같은 I/O 경로·같은 token 순서를 쓰고 layer별 자르는 지점만 다르다.**
- ragged cache는 설계 A(layer별 compact cache + 각 layer self-attention forward pre-hook으로 mask를 오른쪽 정렬)이다. 검증 범위는 고정 core pair 10개(hit prompt 2종)와 integration dialogue 4개이다.
- MT-VQA·GLM·LLaVA는 실행하지 않았다.

## Gate

| Gate | 결과 |
|---|---|
| QG1 CPU (vectors, row budget, validator, no LLaVA defaults) | PASS |
| 기존 CPU 테스트 (non-GLM 전체) | PASS |
| QG2 U 회귀 (bytes·텐서·SDPA 입력·logits·ID bitwise) | PASS |
| QG3 layerwise 구현 (PYR-b20, PYR-b2p5) | PASS |
| QG4 FP64 의미 검증 1e-12 | PASS |
| QG5 negative fixture | PASS |
| QG6 I/O (preadv 1회, vision 0, score 0) | PASS |
| QG7 통합 (6 arm × 12 요청 × 4 dialogue + 독립 감사) | PASS |
| Smoke 72 요청 | PASS |
| 독립 최종 감사 (73,098 요청) | PASS |
| Artifact protection before/after | PASS |
| Deletion log (run-local scratch·gate store만) | PASS |

기존 CPU 테스트: Ran 542 tests in 8.113s (새 테스트 14개 포함), returncode 0. GLM-4.6V 테스트는 분리 실행했고 3건이 이 변경 전과 똑같이 실패한다(설치된 transformers API 문제, 새 파일을 import하지 않음). QG4의 legacy FP32 1e-5 비교는 진단값으로만 기록했다(gate 아님).

## 정확도

| Arm | T1 % | T2 % | T3 % | All % (정답/N) | Followup % (정답/N) |
|---|---|---|---|---|---|
| ReComp | 60.749 | 61.660 | 61.069 | 61.159 (7451/12183) | 61.364 (4984/8122) |
| FullLoad | 60.749 | 61.709 | 61.093 | 61.184 (7454/12183) | 61.401 (4987/8122) |
| Ours B (unpacked) | 60.749 | 57.892 | 57.843 | 58.828 (7167/12183) | 57.868 (4700/8122) |
| U_packed | 60.749 | 57.892 | 57.843 | 58.828 (7167/12183) | 57.868 (4700/8122) |
| PYR-b20_packed | 60.749 | 57.400 | 56.538 | 58.229 (7094/12183) | 56.969 (4627/8122) |
| PYR-b2p5_packed | 60.749 | 57.917 | 57.252 | 58.639 (7144/12183) | 57.584 (4677/8122) |

| 비교 (A − B) | Followup Δpp [95% CI] | All Δpp [95% CI] | T2 Δpp | T3 Δpp | A만 / B만 정답 (followup) | 예측 일치율 |
|---|---|---|---|---|---|---|
| PYR-b20_packed − U_packed | -0.899 [-1.490, -0.302] | -0.599 [-0.993, -0.201] | -0.492 | -1.305 | 194 / 267 | 88.45% |
| PYR-b2p5_packed − U_packed | -0.283 [-0.738, +0.172] | -0.189 [-0.492, +0.114] | +0.025 | -0.591 | 147 / 170 | 91.90% |
| PYR-b20_packed − PYR-b2p5_packed | -0.616 [-1.065, -0.182] | -0.410 [-0.710, -0.121] | -0.517 | -0.714 | 105 / 155 | 93.36% |
| U_packed − Ours B (unpacked) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 | +0.000 | 0 / 0 | 100.00% |
| Ours B (unpacked) − ReComp | -3.497 [-4.260, -2.743] | -2.331 [-2.840, -1.829] | -3.768 | -3.226 | 279 / 563 | 82.75% |
| U_packed − ReComp | -3.497 [-4.260, -2.743] | -2.331 [-2.840, -1.829] | -3.768 | -3.226 | 279 / 563 | 82.75% |
| PYR-b20_packed − ReComp | -4.395 [-5.286, -3.557] | -2.930 [-3.524, -2.371] | -4.260 | -4.531 | 325 / 682 | 79.07% |
| PYR-b2p5_packed − ReComp | -3.780 [-4.597, -2.973] | -2.520 [-3.065, -1.982] | -3.743 | -3.817 | 313 / 620 | 80.55% |
| Ours B (unpacked) − FullLoad | -3.534 [-4.302, -2.784] | -2.356 [-2.868, -1.856] | -3.817 | -3.250 | 277 / 564 | 82.85% |
| U_packed − FullLoad | -3.534 [-4.302, -2.784] | -2.356 [-2.868, -1.856] | -3.817 | -3.250 | 277 / 564 | 82.85% |
| PYR-b20_packed − FullLoad | -4.432 [-5.313, -3.599] | -2.955 [-3.542, -2.399] | -4.309 | -4.556 | 319 / 679 | 79.22% |
| PYR-b2p5_packed − FullLoad | -3.817 [-4.633, -3.023] | -2.545 [-3.089, -2.015] | -3.792 | -3.841 | 311 / 621 | 80.62% |

T1은 모든 arm이 full-pixel이다. 차이는 generated history가 갈라지는 효과를 포함한 method 수준 비교이다. CI가 0을 포함하는 차이는 '차이 미입증'이며 동등하다는 뜻이 아니다.

β 민감도(보고만, 판정에 쓰지 않음): PYR-b2p5 − U_packed = -0.283 [-0.738, +0.172] pp, PYR-b20 − PYR-b2p5 = -0.616 [-1.065, -0.182] pp.

β=2.5가 β=20보다 좋게 나왔지만 판정은 바꾸지 않는다. 이것은 HOLD에서 본 뒤의 사후 관찰이므로 다른 데이터에서 확인이 필요하다.

## 손실 회복 (ReComp·FullLoad 대비, followup)

| Arm | vs ReComp Δpp [95% CI] | vs FullLoad Δpp [95% CI] | TTFT / ReComp | TTFT / FullLoad |
|---|---|---|---|---|
| Ours B (unpacked) | -3.497 [-4.260, -2.743] | -3.534 [-4.302, -2.784] | 0.476× | 0.761× |
| U_packed | -3.497 [-4.260, -2.743] | -3.534 [-4.302, -2.784] | 0.412× | 0.658× |
| PYR-b20_packed | -4.395 [-5.286, -3.557] | -4.432 [-5.313, -3.599] | 0.412× | 0.658× |
| PYR-b2p5_packed | -3.780 [-4.597, -2.973] | -3.817 [-4.633, -3.023] | 0.412× | 0.659× |

균등 25%(U_packed)의 ReComp 대비 손실은 3.497 pp이다. PYR-b20은 손실을 되찾지 못하고 0.899 pp 더 키웠다(ReComp 대비 손실 4.395 pp, 균등의 125.7%).


## Followup 성능과 I/O

| Arm | TTFT mean / p50 / p95 ms | E2E mean / p95 ms | SSD MB | 읽기 호출 (pread / preadv) | GPU cache KV MB | peak GPU alloc MB (followup 평균 / 전체 최대) |
|---|---|---|---|---|---|---|
| ReComp | 113.226 / 113.146 / 126.492 | 133.386 / 157.610 | 0.000 | 0 / 0 | 0.000 | 6165.4 / 6202.5 |
| FullLoad | 70.789 / 71.034 / 74.735 | 91.106 / 107.416 | 24.407 | 57 / 0 | 21.765 | 6116.9 / 6192.5 |
| Ours B (unpacked) | 53.875 / 53.844 / 56.322 | 73.896 / 87.939 | 6.371 | 57 / 0 | 6.371 | 6101.5 / 6203.5 |
| U_packed | 46.614 / 46.585 / 48.606 | 66.897 / 80.682 | 6.371 | 0 / 1 | 6.371 | 6101.5 / 6203.5 |
| PYR-b20_packed | 46.607 / 46.668 / 48.500 | 66.880 / 80.788 | 6.371 | 0 / 1 | 6.371 | 6101.5 / 6203.5 |
| PYR-b2p5_packed | 46.679 / 46.577 / 48.662 | 66.939 / 80.918 | 6.371 | 0 / 1 | 6.371 | 6101.5 / 6203.5 |

구간별 평균 ms (followup; 구간은 서로 겹칠 수 있어 합산하지 않는다. `읽기 포함`은 계획 + 읽기 + (B/FullLoad는) CPU 조립):

| Arm | prompt 준비 | 읽기 포함 | raw read | 계획 | H2D·조립 | hook 설치 | prefill | decode |
|---|---|---|---|---|---|---|---|---|
| ReComp | 4.490 | – | – | – | – | – | 106.431 | 20.153 |
| FullLoad | 0.904 | 24.161 | 20.644 | 0.050 | 3.081 | – | 42.409 | 20.310 |
| Ours B (unpacked) | 0.908 | 9.561 | 7.777 | 0.036 | 1.399 | – | 41.802 | 20.014 |
| U_packed | 0.910 | 2.834 | 2.357 | 0.442 | 0.682 | 0.080 | 41.799 | 20.277 |
| PYR-b20_packed | 0.890 | 2.890 | 2.382 | 0.475 | 0.679 | 0.080 | 41.760 | 20.267 |
| PYR-b2p5_packed | 0.889 | 2.884 | 2.393 | 0.458 | 0.680 | 0.078 | 41.843 | 20.254 |

**packed 효과** (같은 KV, 저장 형식만 다름): U_packed − B의 TTFT = -7.261 ms [-7.340, -7.184] (0.865×; p50 46.585 vs 53.844, p95 48.606 vs 56.322). 읽기 호출은 57회 pread → 1회 preadv, 읽은 바이트는 같다(6.371 MB). packed 경로는 buffer를 한 번 H2D한 뒤 GPU에서 조립하므로 B(CPU 조립 후 layer·종류별 56회 H2D)와 조립 방식도 다르다 — 이 차이는 저장 형식 하나의 효과가 아니라 새 hit 경로 전체의 효과다.

PYR-b20 − U_packed의 TTFT = -0.006 ms [-0.039, +0.028]. 세 packed arm의 SSD 바이트와 GPU cache KV 바이트는 요청마다 서로 같고 B와도 같다(확인; Σ k_l = 28·ceil(N/4)). DONTNEED는 OS page-cache hint이며 NAND cold나 O_DIRECT가 아니다. 절대 TTFT는 다른 실행과 비교하지 않는다.

## 저장과 setup (이미지당 1회)

| Arm | 파일 수 | 기록 MB | outer persistence mean / p50 / p95 ms | writer ms | write / fsync ms | activation mean / p50 / p95 ms |
|---|---|---|---|---|---|---|
| FullLoad | 59 | 24.342 | 63.918 / 63.030 / 71.615 | 62.926 | 8.099 / 32.168 | 14.131 / 13.713 / 15.990 |
| Ours B (unpacked) | 59 | 6.400 | 45.310 / 45.443 / 48.591 | 44.361 | 2.849 / 26.726 | 5.109 / 5.087 / 5.763 |
| U_packed | 3 | 6.387 | 18.620 / 18.534 / 20.840 | 17.897 | 2.016 / 2.399 | 5.605 / 5.585 / 6.237 |
| PYR-b20_packed | 3 | 6.389 | 18.808 / 18.759 / 20.936 | 18.085 | 2.030 / 2.419 | 5.653 / 5.636 / 6.403 |
| PYR-b2p5_packed | 3 | 6.389 | 18.762 / 18.624 / 21.361 | 18.037 | 2.031 / 2.402 | 5.646 / 5.626 / 6.389 |

persistence는 writer wall + completion seal(기존 B와 같은 경계), activation은 메타 검증 + payload SHA256 전체 검증 + (packed) pinned buffer 할당이다. 파일 수는 payload + meta.json + completion.json.

## 3-turn session (shared-stream / standalone-equivalent)

| Arm | setup ms/image | shared-stream mean / p50 / p95 ms | standalone-equivalent (DERIVED) mean / p50 / p95 ms |
|---|---|---|---|
| ReComp | 0.000 | 398.852 / 401.520 / 451.279 | 398.852 / 401.520 / 451.279 |
| FullLoad | 78.049 | 322.097 / 312.035 / 393.223 | 392.706 / 392.443 / 439.594 |
| Ours B (unpacked) | 50.419 | 285.085 / 275.960 / 334.284 | 330.587 / 326.437 / 371.579 |
| U_packed | 24.224 | 268.553 / 260.766 / 307.195 | 290.453 / 285.768 / 331.557 |
| PYR-b20_packed | 24.460 | 268.453 / 260.422 / 307.952 | 290.595 / 285.784 / 332.298 |
| PYR-b2p5_packed | 24.408 | 268.582 / 260.706 / 308.351 | 290.663 / 286.016 / 332.519 |

| A − B | accounting | Δ ms [95% CI] |
|---|---|---|
| Ours B (unpacked) − ReComp | shared_stream_session_ms | -113.767 [-115.703, -111.815] |
| Ours B (unpacked) − ReComp | derived_standalone_session_ms | -68.265 [-69.951, -66.547] |
| U_packed − ReComp | shared_stream_session_ms | -130.299 [-132.336, -128.252] |
| U_packed − ReComp | derived_standalone_session_ms | -108.399 [-110.203, -106.562] |
| U_packed − Ours B (unpacked) | shared_stream_session_ms | -16.531 [-16.750, -16.318] |
| U_packed − Ours B (unpacked) | derived_standalone_session_ms | -40.134 [-40.329, -39.942] |
| PYR-b20_packed − ReComp | shared_stream_session_ms | -130.399 [-132.444, -128.341] |
| PYR-b20_packed − ReComp | derived_standalone_session_ms | -108.258 [-110.064, -106.448] |
| PYR-b20_packed − Ours B (unpacked) | shared_stream_session_ms | -16.632 [-16.932, -16.337] |
| PYR-b20_packed − Ours B (unpacked) | derived_standalone_session_ms | -39.992 [-40.279, -39.700] |
| PYR-b2p5_packed − ReComp | shared_stream_session_ms | -130.270 [-132.297, -128.230] |
| PYR-b2p5_packed − ReComp | derived_standalone_session_ms | -108.189 [-109.976, -106.375] |
| PYR-b2p5_packed − Ours B (unpacked) | shared_stream_session_ms | -16.503 [-16.779, -16.242] |
| PYR-b2p5_packed − Ours B (unpacked) | derived_standalone_session_ms | -39.924 [-40.181, -39.667] |

shared-stream은 이미지 setup을 source dialogue에 한 번 부과한 실측 합이다. standalone-equivalent는 같은 실측 setup을 dialogue마다 다시 부과한 **DERIVED** 값이며 non-source T1에 capture hook을 걸었을 때의 증분은 NOT_MEASURED이다.

## 질문 유형별 Δ (followup, LLaVA 보고서와 같은 regex 분류; 탐색적 수치)

| 유형 | n | U % | PYR-b20 % | PYR-b2p5 % | ReComp % | PYR-b20 − U [95% CI] | PYR-b2p5 − U [95% CI] |
|---|---|---|---|---|---|---|---|
| all | 8122 | 57.87 | 56.97 | 57.58 | 61.36 | -0.899 [-1.490, -0.302] | -0.283 [-0.738, +0.172] |
| yes/no | 3416 | 71.55 | 71.78 | 71.78 | 75.67 | +0.234 [-0.352, +0.803] | +0.234 [-0.294, +0.747] |
| color | 404 | 46.29 | 43.81 | 45.30 | 50.99 | -2.475 [-5.839, +0.978] | -0.990 [-3.769, +1.754] |
| spatial | 1678 | 51.97 | 49.28 | 49.94 | 55.01 | -2.682 [-4.092, -1.336] | -2.026 [-3.194, -0.901] |
| material/attribute | 408 | 52.21 | 53.43 | 53.19 | 58.58 | +1.225 [-1.266, +3.874] | +0.980 [-1.152, +3.202] |
| object | 1965 | 44.22 | 42.70 | 44.27 | 46.31 | -1.527 [-2.860, -0.208] | +0.051 [-1.001, +1.088] |
| other | 251 | 45.82 | 45.42 | 46.61 | 48.21 | -0.398 [-3.891, +2.941] | +0.797 [-1.916, +3.502] |

유형별 수치는 사전 등록한 가설 검정이 아니다. 다중 비교 보정을 하지 않았고 표본이 작은 유형은 CI가 넓다.

## layer별 k_l·바이트 (세 벡터)

| N | 균등 k | PYR-b20 첫 / 마지막 | PYR-b2p5 첫 / 마지막 | Σ k_l (세 벡터 동일) | payload MB (구조 포함, 세 벡터·B 동일) |
|---|---|---|---|---|---|
| 260 | 65 | 127 / 3 | 104 / 26 | 1820 | 4.932 |
| 345 | 87 | 169 / 5 | 139 / 35 | 2436 | 6.193 |
| 529 | 133 | 259 / 7 | 212 / 54 | 3724 | 8.831 |

HOLD 398장의 N은 260 / 345 / 529 (최소 / 실제 이미지 중간값 / 최대), 서로 다른 N 22개. layer 28개 전체의 비율·k_l·바이트는 [layer_k_bytes.csv](layer_k_bytes.csv), 그림은 [fig_vectors_qwen.png](fig_vectors_qwen.png). k_l = 0인 layer는 없다(최소 PYR-b20 3행).

## 정합성 사후 확인 — U_packed ≡ B

U_packed와 기존 B는 같은 KV를 쓰므로 예측이 토큰 단위로 같아야 한다. main 12,183 요청 중 **12,183건 일치 (100.000%)**, 불일치 0건.

gate에서도 같은 capture로 만든 두 store의 payload·조립 텐서·28 layer SDPA 입력·첫 logits이 bitwise 같았고(QG2), integration에서 scored arm 4개의 T1 capture가 비트 단위로 같았다.

## 결정성 참고 (10/02 Qwen 본실험의 같은 요청과 비교)

| Arm | 요청 | 토큰 일치 | 예측 일치 | followup 토큰 일치 |
|---|---|---|---|---|
| ReComp | 12183 | 100.000% | 100.000% | 100.000% |
| FullLoad | 12183 | 100.000% | 100.000% | 100.000% |
| Ours B (unpacked) | 12183 | 100.000% | 100.000% | 100.000% |

참고 수치일 뿐이다. 절대 TTFT는 실행 간 비교하지 않는다.

## 실행 기록

- 실제 요청 수: main **73,098 / 73,098** (398 images, 4,061 dialogues × 3 turns × 6 arms), smoke 72, integration 72.
- 실패·스킵: 0. 실패한 image attempt 기록: 없음. 중단 기록: 없음.
- 소요 시간: main 첫 시작부터 종료까지 2.42시간 (계약의 예상 2.7–3.4시간).
- 독립 감사: PASS (114,010 checks). Artifact protection: PASS (SHA 4700 files, kvstore inventory 16160, 다른 run inventory 1826177). 삭제: run-local scratch 51562 files / 406 image bundles, gate store 1040 files — 기존 산출물 삭제 없음.
- freeze amendment: 없음. 기존 소스 수정 없음(새 파일만 추가).

### 프롬프트와 다르게 한 점

1. 설계 문서를 디스크에 쓰기 전에 설계 A의 기능 점검을 이미지 1장으로 한 번 실행했다(판정·수치에 쓰지 않음). 'GPU 측정 전에 문서로 남긴다'의 순서를 이 1회만큼 지키지 못했다.
2. freeze 전에 별도 run root(`runs/qwen_layer_budget_pyr_dev_20261111T0900Z`)에서 gate·integration·smoke를 한 번 리허설했다(정합성 fixture만 사용, 개발 기록으로 보존). 공식 gate·integration·smoke는 freeze 뒤 이 run에서 처음부터 다시 실행했고 수정 없이 통과했다.
3. packed hit의 조립은 '파일 전체를 pinned buffer로 preadv → buffer 1회 H2D → GPU에서 행 index 1회'로 구현했다. 결과 텐서는 기존 `load_prefix`와 bitwise 같지만(QG2) 조립 연산 위치(CPU → GPU)가 B와 다르다. 그래서 'U_packed vs B'의 TTFT 차이는 읽기 호출 수뿐 아니라 조립 방식의 차이도 포함한다.
4. 기존 CPU 테스트 중 GLM-4.6V 3건은 이 변경 전부터 실패하므로 분리 보고하고 PASS 기준에서 뺐다(계약 8절).
5. 보고 스크립트(171)는 측정 뒤 실행하는 분석 코드라서 freeze 대상 runtime source에서 제외했다. 결과를 본 뒤 음수 Δ에 맞게 문구 세 곳(판정 표의 CI 행, 판정 설명 한 줄, 손실 회복 문장)과 유형 표의 강조를 고쳐 보고서를 다시 생성했다 — 수치·판정·감사 결과는 첫 생성과 같다. 그림 색 팔레트의 자동 검증 스크립트는 node가 없어 실행하지 못했다(선 모양·marker·직접 표기를 함께 썼다).
6. QG2·QG4는 요구보다 넓게 했다(QG2에 28 layer SDPA 입력 비교 추가, QG4를 U에도 적용).

### NOT_MEASURED

- PyramidKV의 attention 기반 token 선택(여기서는 layer별 자르는 지점만 사용).
- H2D 시간 단독(조립과 함께 측정), NAND/controller cold, O_DIRECT.
- standalone session의 non-source T1 capture 증분(DERIVED 값에 포함되지 않음).
- 동시 요청 throughput·메모리 포화.
- MT-VQA, GLM, LLaVA (이번 범위 아님).
