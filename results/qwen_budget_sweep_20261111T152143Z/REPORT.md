# Qwen2.5-VL 보존 비율 sweep (25 / 30 / 40 / 50 / 100%) — MT-GQA HOLD

Run `runs/qwen_budget_sweep_20261111T152143Z`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `efe93f167701ad8f…`, gate·smoke·main 전에 freeze). 단위: MB = 10^6 B, ms, 정확도 차이는 pp. followup = T2–T3. CI = 이미지 단위 paired bootstrap 95% (10,000회, seed 1234).

**판정은 없다. 곡선을 보고한다.** 상태: **FINAL** — main 85,281 / 85,281 요청, 실패·스킵 0, 독립 감사 PASS, U100_packed ≡ FullLoad 12,183 / 12,183.

## 곡선 (HOLD followup)

| Arm | 비율 | layer당 content 행 (N=345) | Followup % (정답/N) | vs ReComp Δpp [CI] | vs FullLoad Δpp [CI] | vs U25 Δpp [CI] | TTFT mean ms | TTFT / ReComp | ReComp 대비 단축 | SSD MB / followup |
|---|---|---|---|---|---|---|---|---|---|---|
| ReComp | – | – | 61.364 (4984/8122) | – | – | – | 113.222 | 1.000× | 0.0% | 0.000 |
| FullLoad | 100% (전체) | 345 | 61.401 (4987/8122) | +0.037 [-0.151, +0.221] | – | – | 70.770 | 0.625× | 37.5% | 24.407 |
| U25_packed | 25% | 87 | 57.868 (4700/8122) | -3.497 [-4.260, -2.743] | -3.534 [-4.302, -2.784] | – | 46.614 | 0.412× | 58.8% | 6.371 |
| U30_packed | 30% | 104 | 58.963 (4789/8122) | -2.401 [-3.098, -1.700] | -2.438 [-3.137, -1.741] | +1.096 [+0.663, +1.529] | 46.956 | 0.415× | 58.5% | 7.403 |
| U40_packed | 40% | 138 | 60.133 (4884/8122) | -1.231 [-1.855, -0.625] | -1.268 [-1.879, -0.674] | +2.265 [+1.681, +2.851] | 47.679 | 0.421× | 57.9% | 9.448 |
| U50_packed | 50% | 173 | 60.859 (4943/8122) | -0.505 [-1.050, +0.026] | -0.542 [-1.079, -0.012] | +2.992 [+2.317, +3.682] | 48.234 | 0.426× | 57.4% | 11.507 |
| U100_packed | 100% | 345 | 61.401 (4987/8122) | +0.037 [-0.151, +0.221] | +0.000 [+0.000, +0.000] | +3.534 [+2.784, +4.302] | 52.270 | 0.462× | 53.8% | 21.765 |

Δ는 (행의 arm − 기준 arm)이다. CI가 0을 포함하는 차이는 '차이 미입증'이며 동등하다는 뜻이 아니다. T1은 모든 arm이 full-pixel이다. 차이는 generated history가 갈라지는 효과를 포함한 method 수준 비교이다.

그림: [fig_tradeoff_qwen.png](fig_tradeoff_qwen.png) (가로축 TTFT), [fig_tradeoff_qwen_ssd.png](fig_tradeoff_qwen_ssd.png) (가로축 SSD MB). 수치 전체는 [summary.json](summary.json).

이 실험에서 반드시 알아야 할 것:

- **모든 packed arm은 같은 중요도 순서(image-only saliency)의 앞 k개를 모든 layer에 똑같이 쓰고 비율만 다르다.** k = ceil(비율 × N). 구조 행 21개는 항상 전부 남긴다. 큰 비율의 집합은 작은 비율의 집합을 포함한다(HOLD 398장 전부에서 감사).
- **U100_packed와 FullLoad의 TTFT 차이는 packed 경로(파일 1개·preadv 1회·GPU 조립)의 효과이며 토큰 선택과 무관하다.** 두 arm은 같은 KV를 쓴다.
- **Qwen의 N은 260–529이다(중간값 이미지 345).** layer당 남는 content 토큰 수(중간값 이미지): 25% → 87개, 30% → 104개, 40% → 138개, 50% → 173개, 100% → 345개. LLaVA 25%는 중간값 이미지에서 552개다(N 1,344–2,880; 이 실행에서 다시 재지 않은 선행 값).
- MT-VQA·GLM·LLaVA는 실행하지 않았다.

## 사전 고정한 요약 기준선 (기술 통계)

| 기준선 | 규칙 | 결과 |
|---|---|---|
| (a) 손실 미검출 | ReComp 대비 Δ의 95% CI가 0을 포함하는 가장 작은 비율 | **U50_packed** |
| (b) LLaVA 25% 수준 | ReComp 대비 손실 점추정이 2.30 pp 이하인 가장 작은 비율 | **U40_packed** |

| 비율 | ReComp 대비 Δpp [95% CI] | CI가 0 포함 (a) | 손실 ≤ 2.30 pp (b) |
|---|---|---|---|
| U25_packed | -3.497 [-4.260, -2.743] | 아니오 | 아니오 |
| U30_packed | -2.401 [-3.098, -1.700] | 아니오 | 아니오 |
| U40_packed | -1.231 [-1.855, -0.625] | 아니오 | 예 |
| U50_packed | -0.505 [-1.050, +0.026] | 예 (차이 미입증) | 예 |
| U100_packed | +0.037 [-0.151, +0.221] | 예 (차이 미입증) | 예 |

두 기준선은 실행 전에 고정했고 HOLD 결과를 보고 바꾸지 않았다. **둘 다 HOLD에서 관찰한 값이며 다른 데이터에서 확인되지 않았다.** (a)는 손실이 없다는 뜻이 아니라 이 표본 크기에서 손실을 검출하지 못했다는 뜻이다.

**(a)는 경계선 결과다.** U50_packed의 ReComp 대비 CI 상한은 +0.026 pp로 0을 겨우 넘고, 같은 arm을 FullLoad(= U100_packed와 같은 KV)와 비교하면 Δ -0.542 pp, CI [-1.079, -0.012]로 0을 포함하지 않는다. 점추정으로는 U50_packed도 ReComp보다 0.505 pp 낮다. 기준선 (a)의 규칙(ReComp 대비 CI)은 그대로 적용했고, 이 문장은 결과를 본 뒤 해석을 돕기 위해 덧붙인 것이다.

## 비율을 올릴 때의 변화 (인접 비율 비교)

| 구간 | Followup Δpp [95% CI] | T2 Δpp | T3 Δpp | TTFT Δ ms | SSD MB Δ | 비고 |
|---|---|---|---|---|---|---|
| U25_packed → U30_packed | +1.096 [+0.663, +1.529] | +0.960 | +1.231 | +0.343 | +1.032 | 상승, CI 0 제외 |
| U30_packed → U40_packed | +1.170 [+0.679, +1.654] | +1.379 | +0.960 | +0.722 | +2.045 | 상승, CI 0 제외 |
| U40_packed → U50_packed | +0.726 [+0.275, +1.180] | +0.665 | +0.788 | +0.555 | +2.059 | 상승, CI 0 제외 |
| U50_packed → U100_packed | +0.542 [+0.012, +1.079] | +0.813 | +0.271 | +4.036 | +10.259 | 상승, CI 0 제외 |

비율이 커지는데 followup 정확도 점추정이 떨어진 구간은 없다.

## 정확도 (turn별)

| Arm | T1 % | T2 % | T3 % | All % (정답/N) | Followup % [95% CI] |
|---|---|---|---|---|---|
| ReComp | 60.749 | 61.660 | 61.069 | 61.159 (7451/12183) | 61.364 [59.750, 62.933] |
| FullLoad | 60.749 | 61.709 | 61.093 | 61.184 (7454/12183) | 61.401 [59.786, 62.970] |
| U25_packed | 60.749 | 57.892 | 57.843 | 58.828 (7167/12183) | 57.868 [56.343, 59.417] |
| U30_packed | 60.749 | 58.852 | 59.074 | 59.558 (7256/12183) | 58.963 [57.434, 60.525] |
| U40_packed | 60.749 | 60.231 | 60.034 | 60.338 (7351/12183) | 60.133 [58.571, 61.711] |
| U50_packed | 60.749 | 60.896 | 60.822 | 60.822 (7410/12183) | 60.859 [59.260, 62.455] |
| U100_packed | 60.749 | 61.709 | 61.093 | 61.184 (7454/12183) | 61.401 [59.786, 62.970] |

| 비교 (A − B) | Followup Δpp [95% CI] | All Δpp [95% CI] | T2 Δpp | T3 Δpp | A만 / B만 정답 (followup) | 예측 일치율 (followup) |
|---|---|---|---|---|---|---|
| U25_packed − ReComp | -3.497 [-4.260, -2.743] | -2.331 [-2.840, -1.829] | -3.768 | -3.226 | 279 / 563 | 82.75% |
| U30_packed − ReComp | -2.401 [-3.098, -1.700] | -1.601 [-2.065, -1.133] | -2.807 | -1.995 | 265 / 460 | 85.10% |
| U40_packed − ReComp | -1.231 [-1.855, -0.625] | -0.821 [-1.237, -0.417] | -1.428 | -1.034 | 239 / 339 | 87.91% |
| U50_packed − ReComp | -0.505 [-1.050, +0.026] | -0.337 [-0.700, +0.017] | -0.763 | -0.246 | 200 / 241 | 90.70% |
| U100_packed − ReComp | +0.037 [-0.151, +0.221] | +0.025 [-0.101, +0.148] | +0.049 | +0.025 | 24 / 21 | 99.08% |
| U25_packed − FullLoad | -3.534 [-4.302, -2.784] | -2.356 [-2.868, -1.856] | -3.817 | -3.250 | 277 / 564 | 82.85% |
| U30_packed − FullLoad | -2.438 [-3.137, -1.741] | -1.625 [-2.092, -1.161] | -2.856 | -2.019 | 265 / 463 | 85.09% |
| U40_packed − FullLoad | -1.268 [-1.879, -0.674] | -0.845 [-1.253, -0.449] | -1.477 | -1.059 | 240 / 343 | 87.82% |
| U50_packed − FullLoad | -0.542 [-1.079, -0.012] | -0.361 [-0.719, -0.008] | -0.813 | -0.271 | 204 / 248 | 90.64% |
| U100_packed − FullLoad | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 | +0.000 | 0 / 0 | 100.00% |
| FullLoad − ReComp | +0.037 [-0.151, +0.221] | +0.025 [-0.101, +0.148] | +0.049 | +0.025 | 24 / 21 | 99.08% |

## Followup 성능과 I/O

| Arm | TTFT mean / p50 / p95 ms | E2E mean / p95 ms | TTFT / ReComp (단축) | TTFT / FullLoad (단축) | SSD MB | 읽기 호출 (pread / preadv) | GPU cache KV MB | peak GPU alloc MB (followup 평균 / 전체 최대) |
|---|---|---|---|---|---|---|---|---|
| ReComp | 113.222 / 113.164 / 126.547 | 133.529 / 157.408 | 1.000× (0.0%) | 1.600× (-60.0%) | 0.000 | 0 / 0 | 0.000 | 6165.4 / 6202.5 |
| FullLoad | 70.770 / 70.854 / 74.660 | 91.191 / 107.484 | 0.625× (37.5%) | 1.000× (0.0%) | 24.407 | 57 / 0 | 21.765 | 6116.9 / 6192.5 |
| U25_packed | 46.614 / 46.508 / 48.547 | 67.007 / 80.940 | 0.412× (58.8%) | 0.659× (34.1%) | 6.371 | 0 / 1 | 6.371 | 6101.5 / 6203.5 |
| U30_packed | 46.956 / 47.007 / 48.926 | 67.359 / 81.294 | 0.415× (58.5%) | 0.664× (33.6%) | 7.403 | 0 / 1 | 7.403 | 6102.5 / 6203.5 |
| U40_packed | 47.679 / 47.896 / 49.826 | 68.097 / 82.176 | 0.421× (57.9%) | 0.674× (32.6%) | 9.448 | 0 / 1 | 9.448 | 6104.6 / 6203.5 |
| U50_packed | 48.234 / 48.278 / 50.400 | 68.676 / 82.720 | 0.426× (57.4%) | 0.682× (31.8%) | 11.507 | 0 / 1 | 11.507 | 6106.7 / 6203.5 |
| U100_packed | 52.270 / 51.966 / 54.931 | 72.898 / 87.948 | 0.462× (53.8%) | 0.739× (26.1%) | 21.765 | 0 / 1 | 21.765 | 6116.9 / 6203.5 |

구간별 평균 ms (followup; 구간은 서로 겹칠 수 있어 합산하지 않는다. `읽기 포함`은 계획 + 읽기 + (FullLoad는) CPU 조립):

| Arm | prompt 준비 | 읽기 포함 | raw read | 계획 | H2D·조립 | hook 설치 | suffix prefill | decode |
|---|---|---|---|---|---|---|---|---|
| ReComp | 4.462 | – | – | – | – | – | 106.445 (full prefill) | 20.300 |
| FullLoad | 0.908 | 24.097 | 20.576 | 0.050 | 3.153 | – | 42.379 | 20.414 |
| U25_packed | 0.914 | 2.874 | 2.399 | 0.440 | 0.670 | 0.076 | 41.744 | 20.388 |
| U30_packed | 0.895 | 3.186 | 2.706 | 0.446 | 0.678 | 0.079 | 41.789 | 20.397 |
| U40_packed | 0.902 | 3.659 | 3.137 | 0.487 | 0.698 | 0.076 | 42.010 | 20.413 |
| U50_packed | 0.901 | 4.223 | 3.727 | 0.462 | 0.762 | 0.076 | 41.935 | 20.436 |
| U100_packed | 0.902 | 7.373 | 6.835 | 0.503 | 1.245 | 0.078 | 42.320 | 20.623 |

**packed 경로 효과 (100%에서, 같은 KV)**: U100_packed − FullLoad의 TTFT = -18.500 ms [-18.665, -18.336] (0.739×; p50 51.966 vs 70.854, p95 54.931 vs 74.660). 읽기 호출은 57회 pread → 1회 preadv. 읽은 바이트는 U100_packed 21.765 MB, FullLoad 24.407 MB이다(FullLoad는 64행 chunk 단위로 padding 행까지 읽는다). 이 차이는 파일 배치·읽기 호출 수·조립 위치(CPU → GPU)를 합친 packed 경로 전체의 효과이며, 어느 토큰을 남기는지와 무관하다.

DONTNEED는 OS page-cache hint이며 NAND cold나 O_DIRECT가 아니다. 절대 TTFT는 다른 실행과 비교하지 않는다.

## 저장과 setup (이미지당 1회)

| Arm | 파일 수 | store MB / 이미지 | persistence mean / p50 / p95 ms | writer ms | write / fsync ms | activation mean / p50 / p95 ms |
|---|---|---|---|---|---|---|
| FullLoad | 59 | 24.342 | 63.765 / 63.282 / 70.985 | 62.789 | 8.109 / 32.045 | 14.095 / 13.765 / 15.958 |
| U25_packed | 3 | 6.388 | 18.842 / 18.714 / 21.921 | 18.082 | 2.036 / 2.442 | 5.689 / 5.712 / 6.369 |
| U30_packed | 3 | 7.416 | 19.888 / 19.863 / 22.062 | 19.162 | 2.380 / 2.625 | 6.333 / 6.339 / 7.224 |
| U40_packed | 3 | 9.453 | 22.224 / 22.265 / 25.918 | 21.475 | 3.096 / 3.122 | 7.618 / 7.582 / 8.663 |
| U50_packed | 3 | 11.505 | 24.301 / 24.290 / 27.552 | 23.553 | 3.852 / 3.383 | 8.858 / 8.817 / 10.012 |
| U100_packed | 3 | 21.724 | 35.365 / 35.055 / 40.087 | 34.607 | 7.843 / 5.117 | 14.776 / 14.538 / 16.882 |

persistence는 writer wall + completion seal, activation은 메타 검증 + payload SHA256 전체 검증 + (packed) pinned buffer 할당이다. 파일 수는 payload + meta.json + completion.json.

## 3-turn session (shared-stream / standalone-equivalent)

| Arm | setup ms/image | shared-stream mean / p50 / p95 ms | standalone-equivalent (DERIVED) mean / p50 / p95 ms |
|---|---|---|---|
| ReComp | 0.000 | 399.398 / 401.848 / 451.708 | 399.398 / 401.848 / 451.708 |
| FullLoad | 77.860 | 322.197 / 311.748 / 393.309 | 392.649 / 392.170 / 438.865 |
| U25_packed | 24.531 | 268.763 / 260.628 / 307.744 | 290.944 / 285.937 / 331.998 |
| U30_packed | 26.221 | 269.743 / 261.632 / 308.994 | 293.477 / 288.659 / 334.838 |
| U40_packed | 29.841 | 271.478 / 263.580 / 312.060 | 298.519 / 293.924 / 340.286 |
| U50_packed | 33.159 | 272.931 / 264.666 / 316.693 | 302.970 / 298.737 / 345.900 |
| U100_packed | 50.141 | 283.038 / 274.030 / 333.522 | 328.455 / 326.310 / 374.169 |

| A − B | accounting | Δ ms [95% CI] |
|---|---|---|
| U25_packed − ReComp | shared_stream_session_ms | -130.635 [-132.657, -128.596] |
| U25_packed − ReComp | derived_standalone_session_ms | -108.454 [-110.224, -106.633] |
| U25_packed − FullLoad | shared_stream_session_ms | -53.434 [-54.082, -52.812] |
| U25_packed − FullLoad | derived_standalone_session_ms | -101.705 [-102.626, -100.800] |
| U30_packed − ReComp | shared_stream_session_ms | -129.655 [-131.654, -127.634] |
| U30_packed − ReComp | derived_standalone_session_ms | -105.921 [-107.655, -104.163] |
| U30_packed − FullLoad | shared_stream_session_ms | -52.455 [-53.068, -51.849] |
| U30_packed − FullLoad | derived_standalone_session_ms | -99.172 [-100.040, -98.318] |
| U40_packed − ReComp | shared_stream_session_ms | -127.919 [-129.900, -125.938] |
| U40_packed − ReComp | derived_standalone_session_ms | -100.878 [-102.553, -99.185] |
| U40_packed − FullLoad | shared_stream_session_ms | -50.719 [-51.306, -50.149] |
| U40_packed − FullLoad | derived_standalone_session_ms | -94.130 [-94.948, -93.324] |
| U50_packed − ReComp | shared_stream_session_ms | -126.466 [-128.389, -124.533] |
| U50_packed − ReComp | derived_standalone_session_ms | -96.428 [-97.982, -94.846] |
| U50_packed − FullLoad | shared_stream_session_ms | -49.266 [-49.789, -48.761] |
| U50_packed − FullLoad | derived_standalone_session_ms | -89.680 [-90.397, -88.966] |
| U100_packed − ReComp | shared_stream_session_ms | -116.359 [-118.146, -114.563] |
| U100_packed − ReComp | derived_standalone_session_ms | -70.943 [-72.128, -69.725] |
| U100_packed − FullLoad | shared_stream_session_ms | -39.159 [-39.501, -38.821] |
| U100_packed − FullLoad | derived_standalone_session_ms | -64.194 [-64.621, -63.751] |

shared-stream은 이미지 setup을 source dialogue에 한 번 부과한 실측 합이다. standalone-equivalent는 같은 실측 setup을 dialogue마다 다시 부과한 **DERIVED** 값이며 non-source T1에 capture hook을 걸었을 때의 증분은 NOT_MEASURED이다.

## 비율별 k와 바이트

| N | U25 k | U30 k | U40 k | U50 k | U100 k | U25 MB | U30 MB | U40 MB | U50 MB | U100 MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 260 | 65 | 78 | 104 | 130 | 260 | 4.932 | 5.677 | 7.168 | 8.659 | 16.114 |
| 345 | 87 | 104 | 138 | 173 | 345 | 6.193 | 7.168 | 9.118 | 11.125 | 20.988 |
| 529 | 133 | 159 | 212 | 265 | 529 | 8.831 | 10.322 | 13.361 | 16.400 | 31.539 |

HOLD 398장의 N은 260 / 345 / 529 (최소 / 실제 이미지 중간값 / 최대), 서로 다른 N 22개. packed 파일 크기 = (k + 21) × 57,344 B. k는 부동소수 `layer_keep_counts`와 정수 분수 `ceil(p·N/q)` 두 가지로 계산해 N = 1–3,000 전부에서 일치함을 확인했다. 전체 표는 [phase0/k_tables.csv](phase0/k_tables.csv).

## Gate

| Gate | 결과 |
|---|---|
| SG1 CPU (벡터 SHA·합, N=1–3,000 정수 분수 k, validator, 기본값 미사용) | PASS |
| SG2 25% 회귀 (payload·K/V·SDPA 입력·logits·ID bitwise, PyramidKV run 코드 대비) | PASS |
| SG3 100% 동치 (U100_packed ≡ FullLoad: K/V·SDPA 입력·logits·ID bitwise) | PASS |
| SG4 포함 관계 (HOLD 398장, U25 ⊂ U30 ⊂ U40 ⊂ U50 ⊂ U100) | PASS |
| SG5 I/O (preadv 1회·pread 0회, 바이트, vision 0·score 0, 질문 불변) | PASS |
| SGR 참조 검증 (추가: 독립 in-memory 참조 bitwise, 30/40/50% FP64 1e-12) | PASS |
| SG6 통합 (7 arm × 12 요청 × 4 dialogue = 84 요청 + 독립 감사) | PASS |
| 기존 CPU 테스트 (non-GLM 전체) | PASS |
| Smoke 84 요청 | PASS |
| 독립 최종 감사 (85,281 요청) | PASS |
| Artifact protection before/after | PASS |
| Deletion log (run-local scratch·gate store만) | PASS |

기존 CPU 테스트: Ran 556 tests in 9.979s (새 테스트 14개 포함), returncode 0. GLM-4.6V 테스트는 분리 실행했고 3건이 PyramidKV run 때와 똑같이 실패한다(설치된 transformers API 문제, 새 파일을 import하지 않음). SG2의 meta 차이는 문서 필드뿐이며 목록은 `gates/gpu_cases/*.json`의 `SG2.meta_differs_only_in_document_fields`에 있다.

## 정합성 사후 확인 — U100_packed ≡ FullLoad

U100_packed와 FullLoad는 같은 KV를 쓰므로 예측이 토큰 단위로 같아야 한다. main 12,183 요청 중 **12,183건 일치 (100.000%)**, 불일치 0건.

gate에서도 같은 capture로 만든 두 store의 layer별 K/V·28 layer SDPA 입력·첫 logits이 bitwise 같았고(SG3), integration에서 stored arm 6개의 T1 capture KV가 비트 단위로 같았다.

## 결정성 참고 (PyramidKV run의 같은 요청과 비교)

| 이 실행의 arm | PyramidKV run의 arm | 요청 | 토큰 일치 | 예측 일치 | followup 토큰 일치 |
|---|---|---|---|---|---|
| ReComp | recompute | 12183 | 100.000% | 100.000% | 100.000% |
| FullLoad | fullload | 12183 | 100.000% | 100.000% | 100.000% |
| U25_packed | qwen_ours_prefixstore_kv25_U_packed | 12183 | 100.000% | 100.000% | 100.000% |

참고 수치일 뿐이다. 절대 TTFT는 실행 간 비교하지 않는다.

## 실행 기록

- 실제 요청 수: main **85,281 / 85,281** (398 images, 4,061 dialogues × 3 turns × 7 arms), smoke 84, integration 84.
- 실패·스킵: 0. 실패한 image attempt 기록: 없음. 중단 기록: 없음.
- 소요 시간: main 첫 시작부터 종료까지 2.80시간 (계약의 예상 2.9–3.8시간).
- 독립 감사: PASS (111,988 checks). Artifact protection: PASS (SHA 4771 files, kvstore inventory 16160, 다른 run inventory 1841831). 삭제: run-local scratch 30044 files / 406 image bundles, gate store 700 files — 기존 산출물 삭제 없음.
- freeze amendment: 없음. 기존 소스 수정 없음(새 파일만 추가).

### 프롬프트와 다르게 한 점

1. freeze 전에 별도 run root(`runs/qwen_budget_sweep_dev_20261111T1530Z`)에서 gate·integration·smoke를 한 번 리허설했다(정합성 fixture만 사용, 개발 기록으로 보존). 공식 gate·integration·smoke는 freeze 뒤 이 run에서 처음부터 다시 실행했다. 자세한 내용은 계약 15절.
2. SG4(CPU)의 점수는 PyramidKV run이 각 HOLD 이미지의 T1에서 capture한 image-only saliency를 썼다(CPU gate에서는 모델을 돌리지 않으므로). 이 실행에서 실제로 capture한 점수로는 main의 이미지별 독립 감사가 398장 전부에서 같은 포함 관계를 다시 확인했다.
3. 요구된 gate에 더해 SGR(다섯 비율 각각의 독립 in-memory 참조 bitwise 비교, 30/40/50%의 FP64 의미 검증)을 넣었다. SG2·SG3에는 28 layer SDPA 입력 비교를 더했다.
4. 기존 CPU 테스트 중 GLM-4.6V 3건은 이 변경 전부터 실패하므로 PyramidKV run과 같이 분리 보고하고 PASS 기준에서 뺐다.
5. 보고 스크립트(176)는 측정 뒤 실행하는 분석 코드라서 freeze 대상 runtime source에서 제외했다. 그림 색 팔레트의 자동 검증 스크립트는 node가 없어 실행하지 못했다(검증된 기준 팔레트의 앞 세 색을 쓰고 marker 모양과 점별 직접 표기를 함께 썼다). main 결과로 보고서를 한 번 만든 뒤 176을 두 군데 고쳤다: 그림의 글자 위치(50%·ReComp label이 선·축과 겹침)와 기준선 (a)의 경계선 설명 문장. 수치·규칙·기준선 결과는 바뀌지 않았다.
6. 프롬프트 문서가 저장소 파일이 아니라 대화로 전달되어 '결과 기록'을 그 문서에 직접 적지 못했다. 같은 내용을 [RESULT_RECORD.md](RESULT_RECORD.md)에 두었다.

### NOT_MEASURED

- H2D 시간 단독(조립과 함께 측정), NAND/controller cold, O_DIRECT.
- standalone session의 non-source T1 capture 증분(DERIVED 값에 포함되지 않음).
- 동시 요청 throughput·메모리 포화.
- 25–100% 사이의 다른 비율(60–90% 등)과 25% 미만.
- LLaVA 25%의 손실(−2.30 pp)과 토큰 수(552)는 선행 실행의 값이며 이 실행에서 다시 재지 않았다.
- MT-VQA, GLM, LLaVA (이번 범위 아님).
