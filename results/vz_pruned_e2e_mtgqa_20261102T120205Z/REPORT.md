# VZ-Pruned-E2E vs Ours-KV25 vs ReComp — MT-GQA generated history (Stage 2)

Run `vz_pruned_e2e_mtgqa_20261102T120205Z`, contract `docs/vz_pruned_e2e_mtgqa_contract.md` (approved, amendments 1–4). Workload 398 images / 4,061 dialogues / 3 turns, strict normalized EM, method-local generated history, seed 1234, greedy, 16 tokens. 모든 숫자는 raw rows를 독립 재채점한 값이다. CI는 image-cluster paired bootstrap 10,000회(공동 resample, pooled sums/counts)이며, CI의 0 포함은 동등성 증거가 아니고 1 pp 미만 차이는 해석하지 않는다.

## 1. 실행 상태와 gate

| 항목 | 상태 |
|---|---|
| Main requests | 36549 / 36549 (VALID) |
| Phase counts | {"correctness": 45, "smoke": 36, "integration": 18, "main": 36549} |
| Image attempts / failed | {"correctness_image_attempts": 5, "smoke_image_attempts": 4, "integration_image_attempts": 1, "main_image_attempts": 398} |
| Independent audit (148, re-run at report time) | PASS |
| G1 freeze / source hashes | PASS |
| G4 correctness | PASS (G4-a evaluable samples 5) |
| Stages | {"G4 GPU CORRECTNESS": "PASS", "SMOKE": "PASS", "INTEGRATION": "PASS", "MAIN": "PASS"} |
| G5 protection (before/after) | PASS |

G2 (per image, 398 main images + smoke/integration/correctness = 408): all PASS = True. Realigned layer-0 K rel L2 max S 4.48e-03 / sep 4.56e-03 (bound 1e-2); oracle ratio max 1.053 / 1.070 (bound 1.5); negative control min 1.095 (> 0.1); layer-0 V rel L2 max 2.5e-03; structural rows bit-identical in every layer (FP16 store) and in BF16 memory (True); prefix embeddings bit-identical to the HF merge in every image (True); sentinel rows 0. Report-only: deep-layer system-row drift vs Ours max K 8.56e-03 / V 1.52e-02; images above the rev.-2 bound 3e-2: 0. Control prefill 101.6 / 114.7 / 118.2 ms and whole gate 2394.3 / 2403.4 / 2787.1 ms per image, outside every timer.

## 2. 품질 (strict EM, %)

| Arm | T1 | T2 | T3 | Hit (T2–T3) | All | Dialogue all-3-correct | cap |
|---|---|---|---|---|---|---|---|
| ReComp | 63.19 | 68.11 | 68.85 | 68.48 (5562/8122) | 66.72 (8128/12183) | 35.48 | 0 |
| Ours-KV25 | 63.19 | 65.70 | 66.66 | 66.18 (5375/8122) | 65.18 (7941/12183) | 33.22 | 0 |
| VZ-Pruned-E2E | 60.99 | 66.26 | 66.90 | 66.58 (5408/8122) | 64.72 (7885/12183) | 31.74 | 0 |

### 2.1 Primary endpoints (사전 지정: Ours − VZ, T1 EM과 all EM)

| Contrast | Scope | Δ pp [95% CI] | n | CI excludes 0 | |Δ| ≥ 1 pp | 판정 |
|---|---|---|---|---|---|---|
| Ours - VZ | T1 | +2.19 [+1.10, +3.26] | 4061 | True | True | 차이 있음 |
| Ours - VZ | all | +0.46 [-0.23, +1.15] | 12183 | False | False | 미입증 |

### 2.2 Secondary contrasts (탐색적)

| Contrast | Scope | Δ pp [95% CI] | n | CI excludes 0 | 판정 |
|---|---|---|---|---|---|
| Ours - VZ | T2 | -0.57 [-1.52, +0.44] | 4061 | False | 미입증 |
| Ours - VZ | T3 | -0.25 [-1.32, +0.81] | 4061 | False | 미입증 |
| Ours - VZ | hit | -0.41 [-1.19, +0.37] | 8122 | False | 미입증 |
| Ours - ReComp | T1 | +0.00 [+0.00, +0.00] | 4061 | False | 미입증 |
| Ours - ReComp | T2 | -2.41 [-3.39, -1.45] | 4061 | True | 차이 있음 |
| Ours - ReComp | T3 | -2.19 [-3.16, -1.21] | 4061 | True | 차이 있음 |
| Ours - ReComp | hit | -2.30 [-3.05, -1.54] | 8122 | True | 차이 있음 |
| Ours - ReComp | all | -1.53 [-2.03, -1.03] | 12183 | True | 차이 있음 |
| VZ - ReComp | T1 | -2.19 [-3.26, -1.10] | 4061 | True | 차이 있음 |
| VZ - ReComp | T2 | -1.85 [-2.76, -0.98] | 4061 | True | 차이 있음 |
| VZ - ReComp | T3 | -1.95 [-3.02, -0.89] | 4061 | True | 차이 있음 |
| VZ - ReComp | hit | -1.90 [-2.66, -1.16] | 8122 | True | 차이 있음 |
| VZ - ReComp | all | -1.99 [-2.69, -1.31] | 12183 | True | 차이 있음 |

### 2.3 History-matched 보충 (report-only): 두 arm의 T1 정규화 답이 같은 대화의 hit EM

| Contrast | Dialogues (matched / all) | Δ pp [95% CI] |
|---|---|---|
| Ours - VZ | 3443 / 4061 | -0.58 [-1.38, +0.20] |
| Ours - ReComp | 4061 / 4061 | -2.30 [-3.05, -1.54] |
| VZ - ReComp | 3443 / 4061 | -1.48 [-2.23, -0.75] |

## 3. 지연 (ms, mean / p50 / p95)

### 3.1 첫 턴 (T1)

| Arm | TTFT | E2E | prefill→first token | decode | GPU peak GB |
|---|---|---|---|---|---|
| ReComp | 522.6 / 525.0 / 586.3 | 550.1 / 549.8 / 635.4 | 505.1 / 507.5 / 567.2 | 27.4 / 19.9 / 57.5 | 6.62 / 6.73 / 7.84 |
| Ours-KV25 | 522.7 / 525.4 / 586.4 | 550.3 / 550.0 / 635.9 | 505.2 / 507.5 / 567.4 | 27.4 / 19.9 / 57.3 | 6.57 / 6.73 / 7.15 |
| VZ-Pruned-E2E | 159.6 / 172.7 / 180.2 | 186.4 / 192.9 / 229.6 | 101.8 / 114.6 / 120.3 | 26.7 / 19.2 / 56.4 | 4.36 / 4.31 / 5.35 |

VZ T1 TTFT 분해 (mean ms): image_decode 1.8, processor_and_h2d 15.3, vision_projector_pack_saliency 39.7, saliency_order_plan 0.9, embed_assembly 0.1, prefill_to_first_token 101.8. LM prefill 토큰 수 627 / 647 / 708 (full-token 기준 2242 / 2339 / 2532). VZ T1은 실제 serving 경로(vision tower 1회 → image-only saliency → [sys|keep|question] 조립 → LM prefill 1회)에서 측정한 값이며 ReComp/Ours T1 TTFT와 직접 비교 가능하다.

### 3.2 Cache hit (T2–T3)

| Arm | TTFT | E2E | SSD MB | preads | chunk I/O | scatter |
|---|---|---|---|---|---|---|
| ReComp | 529.3 / 532.5 / 593.1 | 557.7 / 559.4 / 642.4 | 0.0 / 0.0 / 0.0 | 0 / 0 / 0 | N/A / N/A / N/A | N/A / N/A / N/A |
| Ours-KV25 | 246.4 / 247.3 / 277.7 | 275.3 / 273.3 / 324.0 | 308.0 / 320.9 / 352.3 | 65 / 65 / 65 | 116.7 / 116.7 / 135.3 | 82.4 / 83.5 / 93.9 |
| VZ-Pruned-E2E | 243.7 / 245.0 / 273.3 | 272.8 / 270.4 / 320.4 | 308.0 / 320.9 / 352.3 | 65 / 65 / 65 | 114.0 / 114.3 / 132.1 | 82.4 / 83.5 / 93.4 |

### 3.3 Paired Δ (ms, request 단위, 공동 bootstrap)

| Contrast | Scope | Metric | Δ [95% CI] |
|---|---|---|---|
| Ours - VZ | T1 | ttft_ms | +363.1 [+357.3, +369.0] |
| Ours - VZ | T1 | request_e2e_ms | +363.9 [+358.1, +369.8] |
| Ours - VZ | hit | ttft_ms | +2.7 [+1.9, +3.5] |
| Ours - VZ | hit | request_e2e_ms | +2.5 [+1.7, +3.3] |
| Ours - ReComp | T1 | ttft_ms | +0.2 [+0.1, +0.2] |
| Ours - ReComp | T1 | request_e2e_ms | +0.1 [+0.1, +0.2] |
| Ours - ReComp | hit | ttft_ms | -282.9 [-288.4, -277.5] |
| Ours - ReComp | hit | request_e2e_ms | -282.5 [-287.9, -277.0] |
| VZ - ReComp | T1 | ttft_ms | -362.9 [-368.9, -357.1] |
| VZ - ReComp | T1 | request_e2e_ms | -363.7 [-369.6, -357.9] |
| VZ - ReComp | hit | ttft_ms | -285.6 [-291.1, -280.0] |
| VZ - ReComp | hit | request_e2e_ms | -284.9 [-290.4, -279.4] |

## 4. 이미지별 persistence / activation (MEASURED; source dialogue T1에서 1회)

| Arm | Stored content | Persistence ms | native writer ms | hash ms | synthetic ms | Logical MB | OS write MB | Activation ms |
|---|---|---|---|---|---|---|---|---|
| Ours-KV25 | 25.0% | 565.8 / 572.5 / 671.0 | 399.5 / 402.8 / 476.0 | 156.9 / 158.2 / 172.8 | — | 309.7 / 320.5 / 338.0 | 309.7 / 320.5 / 338.0 | 161.6 / 163.2 / 178.2 |
| VZ-Pruned-E2E | 25.0% | 576.7 / 580.9 / 689.6 | 412.0 / 411.3 / 501.9 | 155.7 / 157.1 / 171.2 | 3.1 / 3.1 / 3.3 | 309.7 / 320.5 / 338.0 | 309.7 / 320.5 / 338.0 | 160.4 / 162.0 / 177.0 |

VZ − Ours per image: service_persistence_ms +10.9 [+8.0, +13.8]; file_logical_bytes +58.6 [+58.2, +58.9]; os_write_bytes +58.6 [+58.2, +58.9]; activation_ms -1.2 [-1.6, -0.8]

## 5. 대화 단위 session 시간 (113 정의)

Shared-stream = 3개 요청 E2E 합 + source dialogue에만 1회 귀속된 build/activation. Standalone-equivalent = 요청 합 + 이미지 setup을 모든 대화에 재귀속(DERIVED). 귀속 setup 합 582915 ms = 물리 event 합 582915 ms (True).

| Arm | 3-request E2E sum | Shared-stream session | Standalone-equivalent session |
|---|---|---|---|
| ReComp | 1665.5 / 1695.8 / 1894.6 | 1665.5 / 1695.8 / 1894.6 | 1665.5 / 1695.8 / 1894.6 |
| Ours-KV25 | 1100.8 / 1108.7 / 1250.5 | 1172.1 / 1118.6 / 1845.5 | 1830.4 / 1853.7 / 2061.6 |
| VZ-Pruned-E2E | 731.9 / 734.6 / 847.8 | 804.2 / 740.7 / 1485.4 | 1473.8 / 1484.9 / 1723.1 |

| Contrast | Metric | Δ ms [95% CI] |
|---|---|---|
| Ours - VZ | shared_stream_session_ms | +367.9 [+361.7, +374.1] |
| Ours - VZ | standalone_session_ms | +356.6 [+349.6, +363.2] |
| Ours - VZ | request_e2e_sum_ms | +368.8 [+362.6, +375.1] |
| Ours - ReComp | shared_stream_session_ms | -493.5 [-504.3, -482.5] |
| Ours - ReComp | standalone_session_ms | +164.8 [+161.4, +168.4] |
| Ours - ReComp | request_e2e_sum_ms | -564.8 [-575.6, -553.8] |
| VZ - ReComp | shared_stream_session_ms | -861.4 [-878.0, -844.8] |
| VZ - ReComp | standalone_session_ms | -191.8 [-200.2, -183.0] |
| VZ - ReComp | request_e2e_sum_ms | -933.6 [-950.5, -916.7] |

## 6. VZ store vs Ours store, layer별 (G2 기록; 398 main images 평균 / 최대)

S rows의 K는 position realign 후 비교. layer 0는 gate(위), deep layer의 S-row 차이는 문맥 효과(축소 문맥 T1 vs 전체 문맥 T1)이며 gate가 아니다. "control vs Ours"는 같은 길이(L+q)로 자른 원본 시퀀스의 구조 행과 Ours의 차이, 즉 shape rounding floor다.

| Layer | K_S realigned (mean / max) | V_S (mean / max) | K_sys (mean / max) | V_sys (mean / max) | K struct control vs Ours (max) | V struct control vs Ours (max) |
|---|---|---|---|---|---|---|
| 0 | 4.08e-03 / 4.48e-03 | 1.08e-03 / 2.51e-03 | 9.84e-04 / 2.21e-03 | 1.12e-03 / 2.51e-03 | 2.37e-03 | 2.60e-03 |
| 1 | 2.61e-02 / 3.84e-02 | 1.96e-02 / 3.38e-02 | 5.59e-03 / 6.28e-03 | 3.85e-03 / 4.99e-03 | 6.28e-03 | 4.99e-03 |
| 2 | 2.59e-02 / 3.54e-02 | 3.28e-02 / 4.71e-02 | 4.40e-03 / 5.90e-03 | 7.14e-03 / 9.02e-03 | 5.90e-03 | 9.02e-03 |
| 4 | 4.54e-02 / 6.32e-02 | 1.06e-01 / 1.42e-01 | 4.49e-03 / 5.21e-03 | 1.05e-02 / 1.25e-02 | 5.21e-03 | 1.25e-02 |
| 8 | 2.14e-01 / 2.48e-01 | 3.86e-01 / 4.32e-01 | 5.49e-03 / 6.82e-03 | 1.05e-02 / 1.27e-02 | 6.82e-03 | 1.27e-02 |
| 16 | 3.52e-01 / 3.96e-01 | 4.77e-01 / 5.51e-01 | 5.77e-03 / 6.46e-03 | 1.15e-02 / 1.26e-02 | 6.95e-03 | 1.32e-02 |
| 24 | 2.94e-01 / 3.37e-01 | 4.48e-01 / 5.21e-01 | 6.22e-03 / 7.00e-03 | 1.16e-02 / 1.22e-02 | 7.56e-03 | 1.52e-02 |
| 30 | 2.60e-01 / 3.08e-01 | 5.03e-01 / 5.88e-01 | 7.61e-03 / 8.56e-03 | 1.40e-02 / 1.52e-02 | 1.01e-02 | 1.94e-02 |
| 31 | 2.58e-01 / 3.02e-01 | 5.18e-01 / 6.05e-01 | 7.04e-03 / 8.07e-03 | 1.25e-02 / 1.36e-02 | 9.90e-03 | 1.79e-02 |

## 7. G6 — B main 대비 요청 단위 재현 (report-only)

| Arm | Turn | Requests | Tokens equal | % | Prediction equal | Prompt identical | Mismatch: prompt-identical | Mismatch: history-cascade |
|---|---|---|---|---|---|---|---|---|
| ReComp | 1 | 4061 | 4061 | 100.00 | 4061 | 4061 | 0 | 0 |
| ReComp | 2 | 4061 | 4061 | 100.00 | 4061 | 4061 | 0 | 0 |
| ReComp | 3 | 4061 | 4061 | 100.00 | 4061 | 4061 | 0 | 0 |
| Ours-KV25 | 1 | 4061 | 4061 | 100.00 | 4061 | 4061 | 0 | 0 |
| Ours-KV25 | 2 | 4061 | 4061 | 100.00 | 4061 | 4061 | 0 | 0 |
| Ours-KV25 | 3 | 4061 | 4061 | 100.00 | 4061 | 4061 | 0 | 0 |

불일치 없음: ReComp와 Ours의 12,183 요청 모두 B main과 토큰 단위로 동일하다.

## 8. G8 — VZ hit vs 같은 history의 cache-less 단일 prefill (report-only)

20 images × T2,T3 = 40 pairs: first token equal 40/40, all tokens equal 40/40, answer equal 40/40; 첫 토큰 logit max |Δ| 0.132 / 0.125 / 0.191 (shape rounding; gate 아님). G4-d(5 correctness images)는 gpu_correctness.json에 있다.

## 9. 한계

- 같은 재구성 MT-GQA를 B main과 다시 사용했다. Ours − ReComp는 복제, VZ 비교는 새로 사전 등록한 탐색적 비교다.
- Hit EM은 각 arm 자신의 생성 history 아래의 method-level 비교다(2.3의 matched 보충은 조건부 부분집합이며 판정에 쓰지 않는다).
- 세 arm 회전(ordinal % 3)은 B main의 5-arm 회전과 다르므로 열적·캐시 상태의 순서 효과는 B main과 동일하지 않다.
- Store는 이미지당 source dialogue의 T1에서 1회 만든다. VZ prefix KV는 인과 구조상 질문 독립이지만 T1 길이(L+q)에 따른 BF16 shape rounding을 가진다(G2-c 같은 길이 control, G8).
- Persistence/activation은 source dialogue에만 발생한 실측이며 standalone-equivalent는 DERIVED다. OS write는 NAND write가 아니다.
- VZ T1의 saliency·plan 비용은 TTFT 안에 있다(breakdown 참조). Non-source T1의 capture 증분은 측정하지 않았다(VZ는 자신의 saliency capture 외에 없음).
- 보고 스크립트(149, freeze 제외)는 main 진행 중 1개 commit 이미지(99 requests)로 렌더링 경로를 한 번 검증했다. 그 부분 출력은 scratchpad에만 있고 결과 디렉터리에 없으며, 판정 규칙은 freeze 전에 고정되어 변경되지 않았다.

