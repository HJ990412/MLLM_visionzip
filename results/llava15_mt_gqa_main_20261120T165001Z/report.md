# LLaVA-1.5-7B × MT-GQA 본실험: ReComp / FullLoad / IMPRESS / Ours

측정 보고다. GO/NO-GO 판정은 두지 않는다. 측정하지 않은 값은 "미실행" 또는 "—"로 적는다.

**본실험 완료**: 48,732 / 48,732 요청 (398 images, 4,061 dialogues × 3턴 × 4 arm).

## 설정

- 실행: `runs/llava15_mt_gqa_main_20261120T165001Z`(요청 단위 원본), `results/llava15_mt_gqa_main_20261120T165001Z`(집계·보고서·status.json·driver.log). 계약: `docs/llava15_mt_gqa_main_contract.md`(사본 `CONTRACT.md`).
- 환경: Python 3.10.20, torch 2.5.1+cu121, transformers 4.57.6 (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`) — 11/18 LLaVA-1.5 실행과 같다(freeze에서 대조).
- 모델 `llava-hf/llava-1.5-7b-hf` revision `b234b804`, NF4/BF16/eager, FP16 SSD KV, greedy cap16, HF 기본 336×336 전처리, visual token 576개, separator 없음.
- 데이터: MT-GQA 398 images / 4,061 dialogues × 3턴, method-local generated history, strict normalized EM, 대화 단위 arm 순서 회전. 모든 방법이 매 대화 T1을 픽셀부터 계산하고, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건이다.
- ReComp: 매 턴 이미지부터 다시 계산. FullLoad: Visual KV 전체 저장, 워커가 layer 순서로 전부 읽음(11/18 구현).
- IMPRESS: 11/18 실행의 구현·설정 그대로(probe head 0·1·2, 전체 행 열 합 중요도, 상위 25%, 평균 Jaccard ≥ 0.3이면 투표 선택 아니면 fallback, K/V 분리, 64토큰 chunk, 첫 대화 T1 기준 layer별 재배치, 다음 layer probe key 미리 읽기).
- Ours: image-only CLIP CLS→patch attention 중요도, PYR-b20 layer별 배분(총 KV 예산 25%), layer l은 상위 k_l개 토큰을 원래 토큰 인덱스 오름차순으로 저장, 재사용 턴에는 각 행을 GPU cache의 원래 위치에 올리고 나머지는 mask. 파일 구성·비동기 읽기 순서·pinned buffer·읽기 호출 수는 기존 LLaVA-1.5 비동기 경로와 같다.

## Gate

| gate | 4장 | DEV 40장 | 내용 |
|---|---|---|---|
| 1. Ours 저장 내용 일치 | PASS | PASS | 4장 4/4장, 비교한 K·V 행 36,864개 중 다른 행 0개 · DEV 40/40장, 비교한 K·V 행 368,640개 중 다른 행 0개 |
| 2. Ours 선택·배치가 질문과 무관 | PASS | PASS | 4장 4/4장(질문 3개씩) · DEV 40/40장(질문 3개씩) |
| 3. Ours 동기 = 비동기 | PASS | PASS | 4장 147건 중 토큰 불일치 0건, 읽기 바이트 불일치 0건 · DEV 1,134건 중 토큰 불일치 0건, 읽기 바이트 불일치 0건 |
| 4. Ours 두 번 실행 | PASS | PASS | 4장 147건 중 토큰 불일치 0건 · DEV 1,134건 중 토큰 불일치 0건 |
| 5. IMPRESS | PASS | PASS | 유지 100% vs 같은 재배치 store의 FullLoad loader: 4장 147건 중 토큰 불일치 0건 · DEV 1,134건 중 토큰 불일치 0건; 동기 vs 비동기: 4장 147건 중 토큰 불일치 0건, 읽기 바이트 불일치 0건 · DEV 1,134건 중 토큰 불일치 0건, 읽기 바이트 불일치 0건 |
| 6. ReComp = 11/12 실행 | PASS | PASS | 4장 147건 중 토큰 불일치 0건 · DEV 1,134건 중 토큰 불일치 0건 |
| 7. 기존 테스트 | PASS | (공통) | transformers 4.57.6: 795개 실행, 실패 3개(이 작업 전부터 실패하던 것 3개), 새 실패 0개; transformers 5.13.1: 813개 실행, 실패 2개(이 작업 전부터 실패하던 것 2개), 새 실패 0개; 새 테스트 파일 22개 통과 |
| 8. DEV 40장 파일럿 | — | 실행됨 | 5,670건 감사 PASS (판정 항목 아님, 표는 아래) |

본실험 진입 판정: **PASS** (gate 1–7이 4장과 DEV 40장에서 모두 PASS이고 DEV 파일럿이 실행됐을 때만 본실험을 시작한다).

transformers 4.57.6 환경에서 이 작업 전부터 실패하던 테스트(목록만 적음): `setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)`; `test_actual_native_adjacent_pair_partial_rope_keeps_tail_bits (test_glm46v_validation.GlmValidationTests)`; `test_actual_native_sdpa_hook_preserves_output_and_qk_score (test_glm46v_vision.GlmVisionTests)`.

transformers 5.13.1 환경에서 이 작업 전부터 실패하던 테스트(목록만 적음): `test_real_tiny_clip_matches_full_penultimate_attention (test_visdial_turn1_piggyback_core.VisionForwardCaptureTests)`; `test_single_stock_forward_capture_equals_dense_visionzip_formula (test_qwen25_vision.StockVisionCaptureTests)`.

## 본실험 표 (398장)

| 방법 | Followup EM (%) | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 65.279 | 126.1 | 134.5 | 136.7 | 0.00 | 0.0 |
| FullLoad | 65.366 | 116.1 | 115.4 | 123.2 | 301.99 | 64.0 |
| IMPRESS | 64.516 | 244.8 | 243.8 | 260.9 | 267.62 | 99.0 |
| Ours | 63.457 | 74.0 | 73.5 | 79.7 | 75.50 | 32.0 |

Followup = T2–T3, strict normalized EM. SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균, 읽기 호출 수는 pread + preadv 횟수의 요청 평균.

턴별 EM과 참고값:

| 방법 | T1 EM | T2 EM | T3 EM | T1 TTFT 평균 (ms) | followup E2E 평균 (ms) |
|---|---:|---:|---:|---:|---:|
| ReComp | 59.394 | 64.861 | 65.698 | 108.4 | 154.1 |
| FullLoad | 59.394 | 64.836 | 65.895 | 108.3 | 144.8 |
| IMPRESS | 59.394 | 63.605 | 65.427 | 108.7 | 274.2 |
| Ours | 59.394 | 63.137 | 63.777 | 108.4 | 103.7 |

IMPRESS의 종류별 읽기량(followup 평균): probe key 사본 14.38 MB / K 126.62 MB / V 126.62 MB. 읽기 호출 수는 요청마다 다르다(최소 96, 최대 118).

### 짝지은 비교 (이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI, followup)

| 비교 | EM 차이 (pp) | TTFT 차이 (ms) |
|---|---:|---:|
| Ours − IMPRESS | -1.059 [-1.804, -0.336] | -170.9 [-171.4, -170.3] |
| Ours − FullLoad | -1.908 [-2.652, -1.183] | -42.1 [-42.4, -41.8] |
| Ours − ReComp | -1.822 [-2.549, -1.115] | -52.1 [-52.4, -51.8] |
| IMPRESS − FullLoad | -0.850 [-1.448, -0.262] | +128.8 [+128.1, +129.4] |
| FullLoad − ReComp | +0.086 [-0.180, +0.356] | -10.0 [-10.4, -9.6] |

### TTFT 비율 (followup 평균의 비, 같은 실행; 괄호는 이미지 단위 bootstrap 95% CI)

| 비율 | 값 |
|---|---:|
| Ours / FullLoad | 0.637× [0.635, 0.639] |
| Ours / ReComp | 0.587× [0.585, 0.588] |
| Ours / IMPRESS | 0.302× [0.301, 0.303] |
| IMPRESS / FullLoad | 2.109× [2.102, 2.117] |

### IMPRESS 통계 (followup)

- fallback 비율(layer·요청 평균): 51.52% — 한 layer라도 fallback인 요청 100.00%, 32개 layer 모두 fallback인 요청 0.00%
- probe head 간 평균 Jaccard: 0.3368 (임계값 0.3, 이상이면 투표 선택)
- K·V chunk 읽기 비율(읽은 chunk / layer당 9 chunk): K 83.86% · V 83.86% · 평균 83.86% (투표 선택 layer만 66.70%)
- 중요도 중 prefix 행이 차지하는 비율(probe head 3개 × 이미지 토큰 576개, layer·요청 평균): 93.47%
- 다음 layer probe key 미리 읽기: 요청당 31.0회 중 30.94회가 사용 시점에 완료

| layer | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | prefix 행 비율 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.7806 | 33.33% | 33.33% | 91.56% |
| 1 | 0.00% | 0.7946 | 34.64% | 34.64% | 95.04% |
| 2 | 0.00% | 0.6763 | 45.79% | 45.79% | 96.22% |
| 3 | 21.04% | 0.3404 | 67.38% | 67.38% | 94.99% |
| 4 | 56.16% | 0.2961 | 79.02% | 79.02% | 96.55% |
| 5 | 0.79% | 0.3775 | 65.63% | 65.63% | 97.20% |
| 6 | 0.00% | 0.4848 | 61.63% | 61.63% | 94.79% |
| 7 | 83.66% | 0.2622 | 92.21% | 92.21% | 96.61% |
| 8 | 57.24% | 0.2962 | 88.94% | 88.94% | 97.50% |
| 9 | 92.02% | 0.2487 | 97.95% | 97.95% | 96.30% |
| 10 | 77.01% | 0.2767 | 92.53% | 92.53% | 97.17% |
| 11 | 97.69% | 0.2257 | 99.09% | 99.09% | 95.83% |
| 12 | 100.00% | 0.1995 | 100.00% | 100.00% | 97.53% |
| 13 | 6.80% | 0.3517 | 82.69% | 82.69% | 95.47% |
| 14 | 79.52% | 0.2753 | 93.52% | 93.52% | 96.21% |
| 15 | 92.43% | 0.2521 | 97.48% | 97.48% | 95.06% |
| 16 | 30.45% | 0.3216 | 81.19% | 81.19% | 93.89% |
| 17 | 41.64% | 0.3101 | 83.40% | 83.40% | 90.64% |
| 18 | 88.88% | 0.2573 | 96.33% | 96.33% | 93.38% |
| 19 | 100.00% | 0.2035 | 100.00% | 100.00% | 94.81% |
| 20 | 26.39% | 0.3258 | 72.35% | 72.35% | 88.86% |
| 21 | 42.51% | 0.3081 | 91.68% | 91.68% | 81.83% |
| 22 | 42.71% | 0.3114 | 98.05% | 98.05% | 90.71% |
| 23 | 100.00% | 0.1835 | 100.00% | 100.00% | 96.32% |
| 24 | 24.42% | 0.3294 | 86.42% | 86.42% | 83.80% |
| 25 | 23.16% | 0.3261 | 87.72% | 87.72% | 81.21% |
| 26 | 6.93% | 0.3742 | 81.64% | 81.64% | 92.47% |
| 27 | 84.94% | 0.2611 | 99.32% | 99.32% | 94.09% |
| 28 | 55.84% | 0.2999 | 98.10% | 98.10% | 92.22% |
| 29 | 40.69% | 0.3134 | 83.08% | 83.08% | 91.51% |
| 30 | 85.03% | 0.2650 | 95.57% | 95.57% | 96.74% |
| 31 | 90.78% | 0.2476 | 96.74% | 96.74% | 94.48% |

### T1 저장 비용 (이미지당 평균; store는 이미지의 첫 대화 T1에서 한 번 만든다)

| 방법 | 저장 크기 (MB) | Visual KV (MB) | probe key 사본 (MB) | 쓰기 (ms) | 재배치 추가 (ms) | 저장 시간 합 (ms) | p95 | OS 기록량 (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 318.79 | 301.99 | 14.16 | 290.5 | — | 290.5 | 353.1 | 318.79 |
| IMPRESS | 319.16 | 301.99 | 14.16 | 288.6 | 244.5 | 533.1 | 585.9 | 635.76 |
| Ours | 78.14 | 75.50 | 0.00 | 117.3 | — | 117.3 | 162.6 | 78.14 |

- ReComp는 store가 없다. "쓰기"는 writer가 파일을 쓰고 fsync·rename까지 마친 시간이다. "재배치 추가"는 IMPRESS가 T1 통계로 순서를 정해 K·V·probe 파일을 다시 쓰는 시간이다.
- FullLoad store에도 probe key 사본이 들어 있다(11/18 구현의 writer가 같다). FullLoad는 그 파일을 읽지 않는다.
- 실험 하니스가 증거용으로 하는 전체 파일 해시(무결성 봉인)는 위 시간에 넣지 않았다: FullLoad 161.3 ms · IMPRESS 159.1 ms · Ours 40.3 ms.
- IMPRESS의 중요도 통계 캡처는 T1 요청 안에서 일어난다(이미지당 평균 1.4 ms, 그 요청의 TTFT에 포함).

### 재사용 턴(followup) 시간 분해

| 방법 | TTFT 평균 (ms) | 읽기 대기 평균 (ms) | p95 | GPU 적재 평균 (ms) | layer hook 합 (ms) | 워커 읽기 합 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 126.1 | — | — | — | — | — |
| FullLoad | 116.1 | 5.20 | 11.52 | 75.79 | 80.99 | 110.04 |
| IMPRESS | 244.8 | 104.55 | 120.53 | — | 204.36 | — |
| Ours | 74.0 | 0.55 | 2.96 | 38.95 | 39.61 | 26.94 |

- 읽기 대기: 요청 스레드가 읽기를 기다리며 멈춘 시간(동기 읽기는 읽는 시간 전체, 비동기 읽기는 워커가 아직 못 읽은 구간을 기다린 시간).
- GPU 적재: Ours는 selector의 scatter 타이머(GPU cache에 행 쓰기 + mask), FullLoad는 layer hook 시간에서 읽기 대기를 뺀 값(전체 행의 host→GPU 복사).
- IMPRESS는 layer hook 안에서 probe 점수 계산·선택·K/V 읽기·적재를 함께 하므로 GPU 적재만 따로 잰 값이 없다(hook 합 − 읽기 대기 = 99.81 ms가 선택 계산 + 적재).
- 워커 읽기 합은 연산과 겹쳐 진행되므로 TTFT에 더해지는 값이 아니다. 항목들의 합을 TTFT로 해석하지 않는다.

T1 생성 토큰이 네 방법에서 모두 같은 대화: 4,061 / 4,061.

참고(판정 아님): 본실험 ReComp의 생성 토큰을 11/12 실행의 같은 요청과 비교하면 12,183건 중 0건이 다르다.

## DEV 40장 파일럿 (gate 8; 본실험 분모가 아니다)

| 방법 | Followup EM (%) | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 66.534 | 126.3 | 134.5 | 136.9 | 0.00 | 0.0 |
| FullLoad | 67.328 | 115.8 | 114.5 | 125.0 | 301.99 | 64.0 |
| IMPRESS | 66.270 | 239.4 | 237.9 | 253.1 | 267.13 | 98.5 |
| Ours | 66.138 | 74.3 | 73.9 | 81.2 | 75.50 | 32.0 |
| Ours(재배열) | 65.476 | 74.0 | 73.6 | 79.4 | 75.50 | 32.0 |

DEV 40장(756 followup/arm)의 짝지은 비교(참고, 구간이 넓다):

| 비교 | EM 차이 (pp) | TTFT 차이 (ms) |
|---|---:|---:|
| Ours − IMPRESS | -0.132 [-1.729, +1.333] | -165.1 [-166.7, -163.4] |
| Ours − FullLoad | -1.190 [-2.849, +0.387] | -41.5 [-42.3, -40.7] |
| Ours − ReComp | -0.397 [-1.940, +1.139] | -52.0 [-53.0, -51.0] |
| IMPRESS − FullLoad | -1.058 [-2.450, +0.264] | +123.6 [+121.6, +125.6] |
| FullLoad − ReComp | +0.794 [+0.000, +1.743] | -10.5 [-12.0, -9.0] |

### DEV 참고값: Ours(재배열)과 Ours의 차이 (gate가 아니며 본실험 arm에도 없다)

- Followup EM: Ours 66.138% · Ours(재배열) 65.476% · 차이(Ours − Ours(재배열)) +0.661 pp [+0.000, +1.339]
- Followup TTFT 평균: Ours 74.34 ms · Ours(재배열) 74.00 ms · 차이 +0.34 ms [+0.11, +0.59] · 비율 1.0046× [1.0015, 1.0080]
- 생성 토큰이 다른 followup 요청: 10 / 756 · 읽기 바이트나 호출 수가 다른 요청: 0
- 두 store의 같은 토큰 K·V 행 비교: 368,640행 중 다른 행 0개

## 실행 기록

- 요청 수: gate 1,470 / 1,470 · devgate 7,938 / 7,938 · dev 5,670 / 5,670 · main 48,732 / 48,732. gate·devgate·dev 요청은 본실험 분모에 넣지 않았다.
- gate: 프로세스 시작 1회 (2026-11-20 16:51:58Z), 종료 2026-11-20 16:56:33Z, 처음 시작부터 0.08 h.
- devgate: 프로세스 시작 1회 (2026-11-20 16:57:05Z), 종료 2026-11-20 17:22:47Z, 처음 시작부터 0.43 h.
- dev: 프로세스 시작 1회 (2026-11-20 17:25:03Z), 종료 2026-11-20 17:42:00Z, 처음 시작부터 0.28 h.
- main: 프로세스 시작 1회 (2026-11-20 17:42:18Z), 종료 2026-11-20 20:17:40Z, 처음 시작부터 2.59 h.
- 드라이버 실행 1회: 2026-11-20T16:50:46Z (fresh, 당시 단계 None).
- 이어서 실행한 횟수: 드라이버 재실행 0회, 요청 단계 프로세스 재시작 0회 (중단 없이 한 번에 끝났다).
- 다시 실행한 대화: 없음.

프로세스 시작 직후의 followup 요청(방법당 처음 40건)과 나머지 요청의 TTFT (본실험):

| 프로세스 | 첫 요청 시각 (UTC) | 방법 | 처음 n건 평균 (ms) | 그 방법의 첫 요청 (ms) | 나머지 평균 (ms) | 평균 차이 (ms) | 나머지 p95 (ms) | 처음 n건 중 p95 초과 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 (처음 시작) | 2026-11-20 17:42:20Z | ReComp | 125.9 | 109.8 | 126.1 | -0.1 | 136.7 | 3 / 40 |
| 1 (처음 시작) | 2026-11-20 17:42:20Z | FullLoad | 112.3 | 115.7 | 116.1 | -3.7 | 123.2 | 0 / 40 |
| 1 (처음 시작) | 2026-11-20 17:42:20Z | IMPRESS | 233.4 | 242.7 | 244.9 | -11.5 | 260.9 | 1 / 40 |
| 1 (처음 시작) | 2026-11-20 17:42:20Z | Ours | 73.1 | 72.0 | 74.0 | -0.9 | 79.8 | 0 / 40 |

"나머지"는 그 방법의 본실험 followup 요청 중 해당 처음 n건을 뺀 전부다. "p95 초과"는 처음 n건 중 나머지의 p95보다 느린 요청 수다(우연히도 n의 5%쯤은 넘는다).
이어서 실행한 적이 없으므로 "이어서 실행한 직후"의 영향은 측정 대상이 없다. 위 표는 처음 시작 직후의 값이다(모델 로드 뒤 합성 요청 1건으로 예열, 측정에서 제외).

- SSD 기록량(실험 전체, `/sys/block/nvme0n1/stat`의 기록 섹터 차이, 부팅 1회 구간 합): 524.2 GB. 이 값은 장치 전체의 기록량이라 같은 SSD에 쓴 다른 프로세스의 기록도 포함한다.
- 실행 전 예상 기록량(store 크기 × 이미지 수): 509.1 GB (gate 4.4 · devgate 28.6 · dev 44.4 · main 411.1 · 질문 무관 확인 20.5). 본실험에서 store 쓰기 때 프로세스가 OS에 넘긴 바이트의 합(FullLoad·IMPRESS·Ours store): 411.0 GB. 예상값에는 단위 테스트가 임시로 쓰는 store, 요청 기록, 이어 실행으로 다시 만든 store가 들어 있지 않다.
- 드라이버 단계별 장치 기록량: tests 4.8 GB · gate_phase 4.6 GB · gate 0.0 GB · qindep_gate 1.9 GB · devgate_phase 29.6 GB · devgate 0.0 GB · qindep_devgate 18.8 GB · dev_phase 45.4 GB · dev 0.0 GB · main_phase 419.2 GB · finish 0.0 GB.
- 디스크 여유: 실행 전 416.8 GiB, 보고 시점 416.4 GiB. scratch store는 이미지마다 감사·commit 뒤 삭제했다.
- 기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로, 크기, mtime) 비교: 변화 없음 (3,093,142개 → 3,093,142개).
- 소스 freeze 대조(보고 시점): 일치 (freeze 때 기록한 파일의 SHA256 비교; 그 뒤 새로 생긴 파일과 보고서 스크립트는 대상이 아니다).

## 코드에서 추가하거나 바꾼 부분

바꾼 파일 (기본 동작은 그대로, 옵션 추가):

- `mmimpress/llava15/store.py`: `write_llava15_packed_store(..., row_order="importance")`, `persist_captured_llava15_prefix(..., row_order="importance")`에 `row_order="original"` 옵션. layer l이 중요도 순서의 앞 k_l개 토큰(동점 처리 포함, 기존 `visionzip_repack_order`의 순서)을 원래 토큰 인덱스 오름차순으로 저장한다. 메타데이터에 중요도 순열을 남기지 않고(`stored_original_indices`, `stored_min_keep_count`만), `visionzip_layout.pt`를 쓰지 않는다. 새 store format `llava15_prefixstore_ratio_packed_original_v1`, 새 검증 함수 `validate_llava15_packed_original_meta`.
- `mmimpress/llava15/serve.py`: `Llava15PackedContext`가 store의 `row_order`를 읽어 layer별 원래 인덱스를 준비하고, `Llava15PrefixSelector._scatter_layer`가 원래 순서 store의 행을 GPU cache의 원래 토큰 위치에 쓰고 그 위치만 보이게 mask를 만든다. 읽기 계획(`PackedReader.stream_plan`), 워커, pinned buffer, hook은 기존 코드 그대로다.

바꾸지 않은 것: LLaVA-NeXT 경로(`mmimpress/packed_store.py`, `packed_serve.py`, `piggyback.py`), `mmimpress/impress_official.py`, `mmimpress/llava15/raster.py`, `mmimpress/serve.py`, 11/18 실행의 하니스 `scripts/199_eval_llava15_mtvqa_impress_code.py`(이 실험이 import해서 그대로 호출), 기존 store·결과 디렉터리·동결 파일.

새 파일:

- `scripts/206_eval_llava15_mt_gqa_main.py`: 이 실험의 실행 하니스(gate / devgate / dev / main, 대화 단위 이어 실행).
- `scripts/207_validate_llava15_mt_gqa_main.py`: gate 1–8 판정과 본실험 진입 판정.
- `scripts/208_report_llava15_mt_gqa_main.py`: 이 보고서(측정에 관여하지 않으며 소스 freeze 대상이 아니다).
- `scripts/llava15_main_status.py`: status.json, SSD 기록량 표본, 드라이버 실행 기록.
- `scripts/run_llava15_mt_gqa_main.sh`: 드라이버(터미널과 분리, 잠금, 단계별 자동 판정, 이어 실행).
- `tests/test_llava15_original_order.py`: 새 테스트.
- `docs/llava15_mt_gqa_main_contract.md`: 실행 전 동결한 방법 정의와 gate.

새 테스트 (`tests/test_llava15_original_order.py`):

- 저장: 원래 순서 store의 layer별 집합이 재배열 store의 앞 k_l개와 같다(동점 포함), 같은 토큰의 K·V 바이트·byte table·payload 크기가 같다, 메타데이터에 중요도 순열이 없고 `visionzip_layout.pt`가 없다, 기본 경로의 메타데이터·바이트가 그대로다, 변조·잘못된 옵션은 거부된다.
- 적재: 각 행이 GPU cache의 원래 위치에 올라가고 저장하지 않은 위치는 mask된다, 두 store가 layer별로 같은 토큰을 같은 K/V 값으로 보여 준다, 비동기 읽기가 동기 읽기와 같은 buffer·cache·mask를 만들고 읽기 구간·순서·호출 수(32회)가 재배열 store의 비동기 읽기와 같다.
- 질문 무관: 서로 다른 프롬프트의 캡처 세 개로 만든 store의 layer별 인덱스 목록·byte table·파일 구성이 같다.
- 하니스: arm·단계·요청 수, gate 비교 함수, 이어 실행 계획(완료 대화 유지, 끊긴 대화의 행 버리기, store 검증 실패 시 이미지 재시작).

## 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계

지시와 다르게 했거나 지시에 없던 것:

- 이어 실행: 완료된 대화는 건너뛰고 끊긴 대화는 T1부터 다시 한다. 다만 store는 이미지의 첫 대화 T1에서 한 번 만들어 그 이미지의 모든 대화가 쓰므로, 끊긴 지점이 첫 대화이거나 디스크의 store가 만들 때 기록한 SHA256과 맞지 않으면 그 이미지의 store와 기록을 모두 버리고 이미지의 첫 대화부터 다시 한다(이 경우 그 이미지에서 이미 끝났던 대화도 다시 실행된다). 버린 기록은 삭제하지 않고 `discarded_rows_resume_*.jsonl` / `ABANDONED.json`으로 남겨 결과에서만 제외했다.
- 드라이버는 요청 단계가 오류로 끝나면 같은 실행 안에서 최대 3회까지 스스로 다시 시작한다(지시에 없던 것). 각 재시작은 위 "실행 기록"의 프로세스 시작 횟수에 들어 있다. GPU를 쓰는 단계는 다른 프로세스가 GPU를 쓰고 있으면 비워질 때까지(최대 2시간) 기다렸다가 시작한다.
- 소스 freeze 대조는 freeze 때 기록한 파일만 비교한다. 그 뒤 새로 생긴 파일과 보고서 스크립트는 대상이 아니다(저장소의 다른 작업이 실험을 멈추지 않게 하기 위함).
- gate 판정용 arm(Ours 동기 읽기, Ours 두 번째 실행, IMPRESS 동기 읽기, IMPRESS 유지 100%, 재배치 store를 읽는 FullLoad loader)을 4장에서는 본실험 arm과 한 단계에서, DEV 40장에서는 별도 단계(devgate, 7 arm)로 돌렸다. gate 1과 6의 DEV 판정은 파일럿 단계의 기록으로 했다.
- DEV 파일럿은 네 방법에 Ours(재배열)을 더한 5 arm을 한 단계에서 돌렸다. 따라서 파일럿의 arm 회전은 5 arm 기준이고 본실험은 4 arm 기준이다.
- gate 2는 Ours store를 질문 3개로 새로 만들어 layer별 인덱스 목록·행 수·byte table·파일 구성·메타데이터(요청 기록과 payload 해시 제외)를 비교했다. payload 바이트는 보고만 한다.
- gate 7은 실행 환경(transformers 4.57.6)과 교차 확인용 5.13.1 환경 두 곳에서 돌렸고 두 곳 모두 새 실패 0개를 요구했다.
- gate 검사 프로그램이 결과 파일을 만들지 못하고 끝난 경우(판정이 아닌 실행 오류)는 FAIL로 확정하지 않고 멈추며, 같은 명령으로 그 검사부터 다시 한다. 판정이 기록된 gate는 다시 판정하지 않는다.
- 모델을 올린 직후 합성 이미지 요청 1건으로 예열한다(11/18 실행과 같음, 모든 지표에서 제외). 저장 KV의 예열은 없다.
- status.json의 예상 종료 시각은 현재 단계의 요청당 시간을 남은 요청 수에 곱한 근사값이다.
- TTFT 비율에 이미지 단위 bootstrap 95% CI를 함께 적었다(지시에 없던 것).

측정하지 않은 것과 한계:

- 한 번의 실행이다. 실행 간 TTFT 변동은 재지 않았고, 다른 실행의 절대 TTFT와 비교하지 않는다.
- FullLoad의 동기 읽기 arm은 돌리지 않았다(지시의 gate에 없음). FullLoad store의 probe key 사본은 저장만 되고 읽히지 않는다.
- IMPRESS의 GPU 적재 시간은 선택 계산과 한 타이머 안에 있어 따로 재지 못했다.
- SSD 기록량은 장치 전체의 값이며 이 실험만의 기록량을 분리한 값이 아니다. 강제 종료 직전 마지막 표본 이후의 기록은 빠진다.
- 다른 KV 예산(25% 외), 다른 layer 배분, warm page cache 조건, 다른 데이터셋·모델은 이 실험의 범위가 아니다.
- GO/NO-GO 판정은 두지 않았다. 이 문서는 측정 보고다.

