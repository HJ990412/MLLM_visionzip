## AllHead native online stage 계측의 정확한 범위

AllHead selector host 합에는 full-K pread/H2D, scoring, top-k, V read/H2D와 assembly가 포함된다. 이를 순수 scoring 시간으로 해석하지 않는다. 아래 값은 기존 production code가 저장한 host/CUDA interval의 요청별 layer 합이다. Host/CUDA와 상위 interval은 서로 중첩하므로 합산해서 TTFT를 만들지 않는다. CUDA event interval은 해당 stage의 계측이며 독립 kernel-only microbenchmark가 아니다.

| Native interval | Requests | Mean / p50 / p95 ms |
|---|---|---|
| selector_host_inclusive_ms | 8122 | 826.306 / 820.321 / 1008.071 |
| host_intervals_ms/projection_prefill | 8122 | 11.173 / 11.162 / 11.413 |
| host_intervals_ms/scoring_k_h2d | 8122 | 149.113 / 150.353 / 179.203 |
| host_intervals_ms/scoring | 8122 | 20.589 / 20.538 / 21.987 |
| host_intervals_ms/topk | 8122 | 6.174 / 6.171 / 6.353 |
| host_intervals_ms/selected_v_h2d | 8122 | 38.240 / 39.145 / 43.019 |
| host_intervals_ms/assembly | 8122 | 1.966 / 1.968 / 2.083 |
| host_intervals_ms/actual_attention_prefill | 8122 | 5.185 / 5.177 / 5.453 |
| host_intervals_ms/projection_decode | 8122 | 10.802 / 7.596 / 21.956 |
| host_intervals_ms/actual_attention_decode | 8122 | 3.353 / 2.360 / 6.819 |
| cuda_intervals_ms/projection_prefill | 8122 | 6.863 / 6.908 / 7.108 |
| cuda_intervals_ms/scoring_k_h2d | 8122 | 147.959 / 149.185 / 178.022 |
| cuda_intervals_ms/scoring | 8122 | 20.170 / 20.119 / 21.546 |
| cuda_intervals_ms/topk | 8122 | 6.187 / 6.195 / 6.371 |
| cuda_intervals_ms/selected_v_h2d | 8122 | 36.617 / 37.618 / 41.339 |
| cuda_intervals_ms/assembly | 8122 | 1.850 / 1.854 / 2.018 |
| cuda_intervals_ms/actual_attention_prefill | 8122 | 5.378 / 5.264 / 6.790 |
| cuda_intervals_ms/projection_decode | 8122 | 10.145 / 7.131 / 20.626 |
| cuda_intervals_ms/actual_attention_decode | 8122 | 1.951 / 1.496 / 4.231 |

B의 native fixed-prefix selector mean / p50 / p95: 0.029757 / 0.029703 / 0.035117 ms. 모든 8,122개 followup의 query/static score call은 0이다. 이 작은 prefix 선택 interval은 AllHead의 read/H2D/assembly 포함 host 총합과 같은 범위가 아니다.

## 모델별 최종 상태 (요청된 14항목)

| 항목 | LLaVA | Qwen |
|---|---|---|
| BASELINE REUSE CONTRACT | PASS | PASS |
| FINAL SOURCE FREEZE | PASS | PASS |
| CPU REGRESSION | PASS | PASS |
| GPU CORRECTNESS | PASS | PASS |
| MAIN INTEGRATION | PASS | FAIL |
| MAIN-RUNNER SMOKE | PASS | NOT RUN |
| FULL MAIN COVERAGE | PASS | NOT RUN |
| GENERATED-HISTORY LINEAGE | PASS | NOT RUN |
| SELECTION/BUDGET VALIDATION | PASS | NOT RUN |
| I/O AND TIMING AUDIT | PASS | NOT RUN |
| QUALITY RESCORING | PASS | NOT RUN |
| SHARED/DERIVED SESSION ACCOUNTING | PASS | NOT RUN |
| ARTIFACT PROTECTION | PASS | PASS |
| RAW REPRODUCIBILITY | PASS | PASS |

LLaVA 전체 main은 60,915/60,915개, Qwen main은 G7 실패로 0/36,549개다. 두 모델 합계 97,464개를 모두 실행한 것으로 해석하지 않는다. Qwen GPU CORRECTNESS PASS는 고정 10개 B core replay 범위이며 main integration의 FP32 실패를 면제하지 않는다. Qwen RAW REPRODUCIBILITY PASS는 보존된 검증/실패 evidence 범위이고 main 결과는 없다.
