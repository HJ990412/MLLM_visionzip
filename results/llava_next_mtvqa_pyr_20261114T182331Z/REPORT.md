# LLaVA-NeXT-7B + PyramidKV 배분(PYR-beta-raw packed) — MT-VQA-v2-reconstructed-25k 단일 arm

Run `runs/llava_next_mtvqa_pyr_20261114T182331Z`, 결과 `results/llava_next_mtvqa_pyr_20261114T182331Z`. 계약 `docs/llava_next_mtvqa_pyr_contract.md` (SHA256 `4f973e24a702225ccb84448f909af0362e6883c0731dd562d231232534cdd757`). 작성 2026-11-16 03:35:33 KST.

## 요약

- main 요청 **75,000/75,000** (25,000 images × 3 turns), 실패 0건·스킵 0건, 독립 감사 **PASS**. smoke 12·pilot40 120·G-det-U 120 요청은 main 분모에 넣지 않았다.
- main 소요 시간 23.55 h (이미지 실행·감사·hash·gzip·삭제 합 23.52 h; 모델 로딩·중간 보고 포함 경과).
- **(A) 주 판정: IMPROVED** — D = PYR-beta-raw_packed − [B] = +0.7364 pp [+0.5916, +0.8824] (followup 공식 VQA score, cross-run, 결정성 확인됨).
- 근거 gate: G-det-T1 25,000/25,000, G-det-U(pilot40) 120/120.

## 반드시 함께 읽을 사실

- main에서 실행한 arm은 PYR-beta-raw_packed 하나다. ReComp·FullLoad·SparseVLM·[B] 수치는 v3 run(2026-10)의 기록이며 이번에 다시 실행하지 않았다.
- 정확도 비교는 요청 단위로 짝지은 cross-run 비교이고, 근거는 G-det-T1(25,000/25,000건)과 G-det-U(pilot40 120/120건)다.
- TTFT·session의 v3 대비 비교는 cross-run 참고치다. 같은 실행 안 기준은 T1 TTFT뿐이다.
- PYR-beta-raw는 PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 attention 기반 token 선택은 쓰지 않았다. [B]와 같은 token 순서·같은 총 바이트이고, layer별 자르는 지점과 I/O 경로(packed)만 다르다. 따라서 D에는 layer 배분 효과만 들어가지만(G-det-U로 packed 경로 자체는 예측을 바꾸지 않음을 확인), TTFT 차이에는 packed 경로 효과가 섞여 있다.
- MPIC는 비교표에서 제외했다(기존 MPIC 결과는 지우지 않았다). Qwen·GLM·LLaVA-1.5·MT-GQA는 이번에 실행하지 않았다.
- v3와 같은 한계: 기존 250장 개발 데이터와 이미지 166개 중복(unseen holdout 아님), DONTNEED는 cold NAND 보장이 아님, dense GPU 버퍼 유지(GPU 메모리 절감 주장 없음).
- CI가 0을 포함하는 비교는 "차이 미입증"이며 동등성의 증거가 아니다.

## 판정과 표 (계약 8절, 실행 전 고정)

정합성 G-det-T1: PYR arm T1 생성 token = v3 T1 **25,000/25,000** → PASS (cross-run 정확도 비교: ALLOWED).

### (A) PyramidKV 배분 효과 — D = PYR-beta-raw_packed − [B] (followup 공식 VQA score, cross-run, 결정성 확인됨)

| 항목 | 값 |
|---|---|
| PYR-beta-raw packed followup | 50.8466 % |
| [B] v3 followup (같은 이미지) | 50.1102 % |
| D, 95% CI | +0.7364 pp [+0.5916, +0.8824] |
| 판정 규칙 결과 | **IMPROVED** |
| 참고: D ≥ +1.0 pp (MT-GQA GO 기준) | 아니오 |
| 참고: 손실 회복률 D / (ReComp − [B]) | 44.3 % (D / (ReComp - [B]) on the same images; ReComp - [B] = 1.6626 pp) |
| 조건: SSD MB/followup ≤ [B] × 1.02 | 307.75 vs 307.75 MB → 충족 |

### (B) 원래 성능 대비 (followup, cross-run, 결정성 확인됨)

| 비교 | Δ, 95% CI | baseline % | baseline 대비 % | CI |
|---|---:|---:|---:|---|
| PYR − ReComp | -0.9262 pp [-1.0846, -0.7638] | 51.7728 | 98.21 | CI excludes 0 |
| PYR − FullLoad | -0.9340 pp [-1.0954, -0.7708] | 51.7806 | 98.20 | CI excludes 0 |
| PYR − SparseVLM-SSD-KV25-AllHead | -0.7246 pp [-0.8830, -0.5618] | 51.5712 | 98.59 | CI excludes 0 |

### (C) turn별 공식 VQA score (%)

| arm | T1 | T2 | T3 | All | Followup |
|---|---:|---:|---:|---:|---:|
| PYR-beta-raw packed (this run) | 51.7940 | 50.7968 | 50.8964 | 51.1624 | 50.8466 |
| ReComp (v3) | 51.7940 | 51.6864 | 51.8592 | 51.7799 | 51.7728 |
| FullLoad (v3) | 51.7940 | 51.6852 | 51.8760 | 51.7851 | 51.7806 |
| SparseVLM-SSD-KV25-AllHead (v3) | 51.7940 | 51.4476 | 51.6948 | 51.6455 | 51.5712 |
| PrefixStore-KV25 [B] (v3) | 51.7940 | 49.8396 | 50.3808 | 50.6715 | 50.1102 |

turn별 Δ (PYR − baseline, cross-run, 결정성 확인됨):

| baseline | T1 | T2 | T3 | All | Followup |
|---|---|---|---|---|---|
| ReComp | +0.000 pp [+0.000, +0.000] | -0.890 pp [-1.122, -0.660] | -0.963 pp [-1.188, -0.741] | -0.617 pp [-0.723, -0.509] | -0.926 pp [-1.085, -0.764] |
| FullLoad | +0.000 pp [+0.000, +0.000] | -0.888 pp [-1.121, -0.656] | -0.980 pp [-1.204, -0.756] | -0.623 pp [-0.730, -0.514] | -0.934 pp [-1.095, -0.771] |
| SparseVLM-SSD-KV25-AllHead | +0.000 pp [+0.000, +0.000] | -0.651 pp [-0.878, -0.423] | -0.798 pp [-1.023, -0.574] | -0.483 pp [-0.589, -0.375] | -0.725 pp [-0.883, -0.562] |
| PrefixStore-KV25 [B] | +0.000 pp [+0.000, +0.000] | +0.957 pp [+0.745, +1.169] | +0.516 pp [+0.318, +0.713] | +0.491 pp [+0.394, +0.588] | +0.736 pp [+0.592, +0.882] |

answer_type별 followup Δ (탐색적, 다중비교 보정 없음):

| answer_type | 요청 수 | PYR % | [B] % | PYR − [B] | ReComp % | PYR − ReComp |
|---|---:|---:|---:|---|---:|---|
| yes/no | 18,623 | 43.9709 | 43.4055 | +0.565 pp [+0.421, +0.712] | 44.2206 | -0.250 pp [-0.398, -0.100] |
| number | 6,391 | 61.2158 | 59.8326 | +1.383 pp [+0.840, +1.919] | 62.6318 | -1.416 pp [-2.128, -0.712] |
| other | 24,986 | 53.3191 | 52.6207 | +0.698 pp [+0.467, +0.931] | 54.6242 | -1.305 pp [-1.547, -1.060] |

PYR과 [B]의 예측 비교 (정답 = 그 요청의 공식 VQA score > 0):

| 범위 | 요청 수 | 예측 문자열 일치 | 일치율 | PYR만 정답 | [B]만 정답 | PYR score 더 높음 | [B] score 더 높음 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2 | 25,000 | 22,493 | 89.97 % | 514 | 319 | 928 | 562 |
| T3 | 25,000 | 22,833 | 91.33 % | 445 | 342 | 775 | 574 |
| Followup | 50,000 | 45,326 | 90.65 % | 959 | 661 | 1,703 | 1,136 |

### (D) 지연·I/O

| 지표 (이번 실행, PYR-beta-raw packed) | mean | p50 | p95 |
|---|---:|---:|---:|
| followup TTFT ms | 228.34 | 228.67 | 269.88 |
| followup E2E ms | 261.08 | 259.19 | 314.28 |
| T2 TTFT ms | 227.15 | 227.20 | 268.23 |
| T3 TTFT ms | 229.53 | 229.97 | 271.58 |
| T1 TTFT ms (full pixel) | 527.00 | 528.52 | 663.08 |
| T1 E2E ms | 558.79 | 560.78 | 706.47 |
| packed read ms (preadv) | 100.01 | 100.51 | 123.54 |
| scatter ms | 82.60 | 82.74 | 100.34 |
| suffix prefill ms | 39.68 | 39.06 | 43.68 |
| prepare ms (read + scatter + mask) | 187.81 | 188.22 | 227.49 |

- SSD 307.75 MB/followup, preadv 1.00회·pread 0.00회/followup, T1의 store 읽기 0 bytes.
- **같은 실행 안 비율 R_in = followup TTFT / T1 TTFT = 228.34 / 527.00 = 0.4333** (T1은 full pixel 재계산). 참고로 v3 같은 이미지에서 ReComp followup / T1 = 1.0165, [B] followup / T1 = 0.5033.

| v3 arm (cross-run 참고치, 절대 TTFT는 실행 간 비교하지 않는다) | followup TTFT ms | T1 TTFT ms | SSD MB/followup | read 호출/followup |
|---|---:|---:|---:|---:|
| ReComp | 530.15 | 521.56 | 0.00 | 0.00 |
| FullLoad | 755.79 | 522.03 | 1172.38 | 64.00 |
| SparseVLM-SSD-KV25-AllHead | 844.59 | 528.19 | 1183.51 | 77.07 |
| PrefixStore-KV25 [B] | 262.46 | 521.48 | 307.75 | 65.00 |

- 환경 변동 지표(판정 아님): 이번 T1 TTFT 527.00 ms / v3 T1 524.33 ms(5 arm 평균, 같은 이미지) = 1.0051 ([B] arm T1 521.48 ms 대비 1.0106). packed read 처리율 3077 MB/s (MT-GQA PYR run 308.05 MB / 97.5 ms = 3159 MB/s).

### (E) 저장·session

| 지표 (이번 실행) | mean | p50 | p95 |
|---|---:|---:|---:|
| store MB/이미지 (logical) | 310.48 | 323.59 | 380.23 |
| persistence ms | 711.49 | 709.11 | 884.19 |
| activation ms | 189.53 | 190.29 | 231.07 |
| 3 turn 요청 E2E 합 ms | 1080.96 | 1087.92 | 1312.94 |
| 3-turn session ms = 3 E2E + persistence + activation | 1981.98 | 1996.95 | 2427.29 |

이미지당 dialogue 1개라 분할 상각이 없다. v3 session(cross-run 참고치, 같은 이미지): ReComp 1679.79 ms, FullLoad 4541.38 ms, SparseVLM-SSD-KV25-AllHead 4771.00 ms, PrefixStore-KV25 [B] 1954.00 ms.

### (F) MT-GQA와의 일관성

MT-GQA(EM)에서는 PYR − U = +1.244 pp [+0.624, +1.864]였다. MT-VQA(공식 VQA score)의 D = +0.736 pp [+0.592, +0.882]. 부호 같음. 지표가 EM과 VQA score로 다르므로 크기를 직접 비교하지 않는다.

## Gate (Phase 1)

| gate | 결과 |
|---|---|
| vg1_data_scoring_identity | PASS |
| vg2_budget | PASS |
| vg3_floor_propagation | PASS |
| vg4_packed_path | PASS |
| vg5_g_det_U | PASS |
| vg6_regression | PASS |

freeze 수정(amendment) 1건. freeze_amendment_01.json: 분석 코드 수정(요청 경로 무관): smoke(4장)·pilot40(40장) 소표본 분석에서 answer_type별 bootstrap 재표본에 해당 유형 요청이 0개인 경우 CI가 NaN이 되어 JSON 기록이 실패했다. CI가 정의되지 않으면 None + ci_defined=false로 기록하게 하고, JSON은 먼저 인코딩한 뒤 쓰게 했으며, 보고서 문구 2곳(계약 절 번호, SSD 온도 표기)과 REPRODUCE.md 작성을 추가했다.

## 실행 기록

- 실제 요청 수: main 75,000, smoke 12, pilot40 120, G-det-U gate 120 (합 75,252).
- 실패 attempt 0건, 재시도로 commit된 이미지 0장, 스킵 0. 16 token cap 도달 요청 4건.
- 중단·재개: main 프로세스 시작 1회.
- scratch store 삭제: 25,000장(정확한 SHA allowlist, 감사·commit 후). 남은 scratch 파일 0개.
- 보호 검사: 기존 소스 변경 0건, 기존 runs/results 파일 변경 0건·삭제 0건.
- 환경(판정 아님): 시작 시 여유 533.1 GB → 종료 시 527.0 GB, SSD SHPP41-2000GM, SSD 온도(hwmon 최고값) 시작 40.9 °C / 종료 46.9 °C. 2,500장마다의 T1 TTFT·packed read 처리율은 `environment_log.jsonl`.

## 프롬프트와 다르게 한 점

- 164의 구조 감사는 score를 strict EM으로 가정하므로 EM으로 다시 채점한 사본에 실행하고, 채택 행의 공식 VQA score·annotation·생성 ID는 adapter가 별도로 독립 검사한다(GLM auditor와 같은 방식). 108 adapter처럼 감사 내부 채점 함수를 바꿔 끼우는 방식은 164가 gold 첫 답만 넘기기 때문에 쓸 수 없었다.
- VG4 고정 사례: pilot40에서 가장 가로로 긴 이미지가 최소 N 이미지와 같아서 "가로 극단"은 그다음으로 가로로 긴 pilot40 이미지를 썼다(서로 다른 5장). VG4는 PYR-beta-raw와 U 두 벡터 모두에 적용했다.
- VG3는 166 --stage grmin을 그대로 부르지 않고(HOLD 전용 N·벡터 목록이 고정되어 있음) 같은 fixture·같은 fail-closed 목록을 MT-VQA N(960/2304/2880)과 새 감사 경로에 적용한 새 stage로 실행했다.
- 중간 보고(2,500장)는 runner가 이미지 사이에서 동기적으로 분석을 실행해 만든다(약 1–2분 동안 요청 실행이 멈춤; timed 요청과 겹치지 않게 하기 위함). main 소요 시간에는 이 시간이 포함된다.
- G-det-U와 G-det-T1의 "입력 token ID"는 이미지 뒤 text 쪽 input id(pilot40은 전체 리스트와 prompt 문자열, main은 SHA256)로 대조했다. 이미지 쪽 입력은 이미지 파일 SHA256과 같은 processor/모델 freeze로 묶인다(픽셀 tensor를 v3 raw가 보존하지 않음).
- v3 대비 표의 v3 수치는 프롬프트에 적힌 요약값이 아니라 v3 raw에서 같은 이미지 집합으로 다시 계산한 값이다(25,000장 전체에서는 프롬프트 수치와 같다). session의 v3 값은 v3 results의 session.csv(standalone_equivalent_session_ms)를 읽었다.
- "한쪽만 정답"은 그 요청의 공식 VQA score > 0을 정답으로 정의했다(score가 더 높은 쪽 건수도 함께 보고).
- 기존 runs/results 보호는 파일 크기·mtime_ns inventory 전후 비교로 했다(약 188만 개 파일; 큰 payload의 전체 SHA는 아님). 기존 소스는 전체 SHA256.
- Phase 0 prepare가 마지막 기록 단계에서 한 번 중단되어(계약 파일이 아직 없었음) 그 단계만 따로 실행했다(phase0/prepare_note.json). 검사 입력·결과는 바뀌지 않았다.
- freeze 수정 1건(freeze_amendment_01.json): 소표본(smoke·pilot40) 분석에서 answer_type CI가 NaN이 되는 분석 코드 결함을 고치고 재freeze했다. 요청 경로·규칙은 그대로이며, 모든 gate와 smoke·pilot40을 새 freeze에서 다시 실행했다(이전 실행은 superseded_01에 보존). 그래서 실제 GPU 요청 수는 gate/smoke/pilot40이 두 번씩이다.
- 계약 문서의 절 번호가 프롬프트와 다르다(timing 경계를 5절로 분리해 gate = 6절, phase = 7절, 판정 = 8절). 내용은 프롬프트 5·6·7절과 같다.

## NOT_MEASURED

- 같은 실행 안의 ReComp·FullLoad·SparseVLM·[B] TTFT/session (실행하지 않음; v3 값은 cross-run 참고치)
- PyramidKV의 attention 기반 token 선택 (layer별 자르는 지점만 PyramidKV)
- SSD SMART 사용률·NAND 수준 cold 여부 (nvme-cli/smartctl은 root 필요; hwmon 온도만 기록), O_DIRECT
- H2D 전송 바이트·GPU footprint 절감 (dense GPU 버퍼 유지)
- Qwen·GLM·LLaVA-1.5·MT-GQA, MPIC 비교
- PYR-beta(하한 0.10)·U_packed의 MT-VQA main 수치 (U_packed는 pilot40 gate 120요청뿐)

