# MT-GQA-reconstructed generated-history: LLaVA 5-arm / Qwen 3-arm B main

Run: `/home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z`. 본 보고서의 main 수치는 이 run에서 실제 채택한 요청만 사용한다. MT-GQA-reconstructed이며 공식 MetaCompress 원본 artifact/evaluator 재현은 아니다.

## 1. 실행 범위

| Model | Main status | 방법 | Images / dialogues | Final / expected requests | Physical started / failed image attempts | Smoke | History |
|---|---|---|---|---|---|---|---|
| llava | PASS | 5 | 398 / 4061 | 60,915 / 60,915 | 60915 / 0 | PASS | PASS |
| qwen | BLOCKED | 3 | main 0 / 0 | 0 / 36,549 | 0 / 0 | NOT RUN | NOT RUN |

CPU 456개 회귀검사, B 고정 LLaVA 5개/Qwen 10개 검증, AllHead 검증, 통합·smoke·reference·예외 주입은 main 분모와 별도다. 실행된 세부 개수와 실패 attempt는 physical_attempts.csv 및 독립 감사 JSON에 있다. ReComp followup은 비교 cohort이며 SSD hit가 아니다.

Qwen은 새 B 경로에서도 `n313060 / mtgqa_000285 / T2`의 기존 FP32 dense/compact criterion(atol=rtol=1e-5)을 실패했다. 215,040개 원소 중 16개 위반, 최대 절대차 2.16215848923e-5. 같은 capture의 native BF16 KV bits, memory/SSD first logits·generated IDs 비교는 이 assertion 전에 통과했다. 이는 G7 면제가 아니며 Qwen main·smoke는 BLOCKED / NOT RUN이다. 기존 v1 실패·이전 main 실패·이번 B core PASS·이번 main integration FAIL을 서로 대체하지 않는다. FP64 대조 또는 허용치 완화로 승인하지 않았다.

## llava: 2. 품질

| Method | T1 % | T2 % | T3 % | All % (correct/N) | Followup % (correct/N) |
|---|---|---|---|---|---|
| ReComp | 63.186 | 68.111 | 68.850 | 66.716 (8128/12183) | 68.481 (5562/8122) |
| FullLoad | 63.186 | 68.062 | 68.702 | 66.650 (8120/12183) | 68.382 (5554/8122) |
| MPIC-32 | 63.186 | 68.013 | 69.096 | 66.765 (8134/12183) | 68.555 (5568/8122) |
| AllHead | 63.186 | 67.865 | 68.702 | 66.585 (8112/12183) | 68.284 (5546/8122) |
| B | 63.186 | 65.698 | 66.658 | 65.181 (7941/12183) | 66.178 (5375/8122) |

| B − baseline | Followup accuracy Δpp [95% CI] | All accuracy Δpp [95% CI] |
|---|---|---|
| ReComp | -2.302 [-3.052, -1.540] | -1.535 [-2.035, -1.027] |
| FullLoad | -2.204 [-2.951, -1.451] | -1.469 [-1.968, -0.967] |
| MPIC-32 | -2.376 [-3.135, -1.607] | -1.584 [-2.090, -1.072] |
| AllHead | -2.105 [-2.868, -1.352] | -1.404 [-1.912, -0.901] |

T1/T2/T3 모든 paired CI와 first-token/full-sequence/normalized-prediction agreement 및 both-correct/reference-only/B-only/both-wrong은 paired_analysis.csv, prediction_agreement.csv에 있다. Strict normalized exact match를 원문 prediction/gold[0]에서 독립 재계산했다. History는 방법 자체의 실제 생성 답변이므로 차이는 history divergence를 포함한다. B/A 일치는 baseline 품질 동등성 근거가 아니다.

## llava: 3. Followup latency / 실제 SSD I/O

| Method | TTFT mean / p50 / p95 ms | E2E mean / p50 / p95 ms | Full-K scoring / other K / V MB | Struct / aux / total MB | Preads |
|---|---|---|---|---|---|
| ReComp | 533.24 / 535.27 / 599.76 | 561.71 / 565.43 / 648.90 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 |
| FullLoad | 781.71 / 771.47 / 957.39 | 810.78 / 803.17 / 990.62 | 0.00 / 586.96 / 586.96 | 0.00 / 0.00 / 1173.92 | 64.00 |
| MPIC-32 | 793.12 / 800.64 / 942.74 | 821.73 / 828.60 / 974.70 | 0.00 / 586.96 / 586.96 | 0.00 / 0.26 / 1174.18 | 65.00 |
| AllHead | 877.51 / 871.98 / 1062.36 | 911.15 / 907.52 / 1100.97 | 586.96 / 0.00 / 577.73 | 19.43 / 0.00 / 1184.12 | 78.31 |
| B | 262.58 / 260.52 / 310.24 | 291.42 / 287.34 / 344.66 | 0.00 / 144.31 / 144.31 | 19.43 / 0.00 / 308.05 | 65.00 |

AllHead additional selected-K 및 probe read는 실제 trace에서 0이다. Full K 한 번을 scoring과 answer에서 재사용하며 total bytes에 이중 합산하지 않는다. Other K는 FullLoad/MPIC/B의 실제 K payload read다. Structural sidecar 중복은 실제 traffic이다. ReComp KV read 0은 image-file I/O 0을 뜻하지 않는다. H2D·selector·assembly·GPU peak의 측정 필드/미측정 수는 latency.csv에 명시했다. H2D는 tensor bytes이며 버스 계측이 아니다. GPU peak는 coexisting contexts 범위이며 독립 배포 footprint가 아니다.

## llava: 4. 이미지별 storage / setup

| Method | Stored content % (image / pooled) | Logical / allocated MB | OS write MB / calls | Persistence / activation ms | Unused probe MB | Provenance |
|---|---|---|---|---|---|---|
| FullLoad | 100.0000 / 100.0000 | 1264.667 / 1264.699 | 1264.667 / 108.04 | 1879.086 / 651.845 | 54.819 | MEASURED |
| MPIC-32 | 100.0000 / 100.0000 | 1187.802 / 1187.808 | 1187.802 / 69.00 | 1832.132 / 588.829 | 0.000 | MEASURED |
| AllHead | 100.0000 / 100.0000 | 1264.667 / 1264.699 | 1264.667 / 108.04 | 1879.086 / 689.584 | 54.819 | MEASURED_SHARED/DERIVED_FROM_SHARED |
| B | 25.0000 / 25.0000 | 309.676 / 309.682 | 309.676 / 81.95 | 684.747 / 158.378 | 0.000 | MEASURED |

B는 원래 N 기준 ceil(N/4) content와 모든 structural rows를 저장하고 마지막 파일은 정확히 k rows에서 끝난다. Full CPU materialization/repack 비용은 유지한다. FullLoad/AllHead canonical bundle은 동일 물리 build이며 표의 두 deployment footprint를 물리 총량에 중복 합산하지 않는다. Canonical bundle의 미사용 probe sidecar까지 쓰기·fsync·완료 hash 비용에 포함한다. AllHead base-only persistence는 NOT_MEASURED. OS write는 /proc/self/io successful write bytes/calls이며 NAND write가 아니다. Native writer stage와 common seal stage는 storage_metrics.csv에 있으며 중첩 component를 request critical path에 합산하지 않는다.

## llava: 5. 실제 shared-image stream 회계

| Method | 3-request E2E mean ms | Physical attributed session mean / p50 / p95 ms |
|---|---|---|
| ReComp | 1675.61 | 1675.61 / 1705.11 / 1903.65 |
| FullLoad | 2174.39 | 2422.44 / 2197.28 / 4668.85 |
| MPIC-32 | 2195.97 | 2433.24 / 2252.51 / 4557.08 |
| AllHead | 2374.65 | 2442.24 / 2408.34 / 3153.86 |
| B | 1134.18 | 1216.81 / 1148.19 / 1974.52 |

B − ReComp 실제 귀속 session Δms: **-458.800 [-470.517, -446.719]**. 원본 physical build/activation event 합과 dialogue 귀속 setup 합의 일치를 session_accounting_audit.json에서 확인했다. Canonical build는 FullLoad에 한 번만 귀속되며 AllHead의 build 귀속 0은 독립 배포 무료를 의미하지 않는다.

## llava: 6. Standalone-equivalent 환산

| Method | Derived session mean / p50 / p95 ms | 범위 |
|---|---|---|
| ReComp | 1675.61 / 1705.11 / 1903.65 | actual requests + per-image setup once/dialogue; DERIVED |
| FullLoad | 4692.57 / 4647.55 / 5891.21 | actual requests + per-image setup once/dialogue; DERIVED |
| MPIC-32 | 4599.54 / 4516.87 / 6118.47 | actual requests + per-image setup once/dialogue; DERIVED |
| AllHead | 4931.85 / 4889.16 / 6196.67 | actual requests + per-image setup once/dialogue; DERIVED |
| B | 1976.32 / 1968.42 / 2414.37 | actual requests + per-image setup once/dialogue; DERIVED |

B − ReComp 환산 session Δms: **+300.712 [+288.335, +314.206]**. Store를 모든 dialogue마다 실제 새로 생성한 측정값이 아니다. Source T1만 capture하므로 hypothetical non-source capture 증분은 NOT_MEASURED이며 완전한 cold-new-image session으로 부르지 않는다. Qwen score/clone tail은 source T1 E2E에 한 번 들어가며 setup에 재청구하지 않는다. Conditioning/logging/audit/cleanup은 request/setup 합과 분리한다.

## 7. LLaVA AllHead와 B의 실제 tradeoff

| Method | Attention content | Stored content % | SSD MB / V MB | TTFT mean / p95 ms | Followup accuracy % | Selector host / prefix selector ms | Selection |
|---|---|---|---|---|---|---|---|
| AllHead | ceil(N/4) every layer | 100.0000 | 1184.120 / 577.734 | 877.505 / 1062.361 | 68.284 | 826.31 | actual suffix/layer dependent |
| B | ceil(N/4) every layer | 25.0000 | 308.050 / 144.312 | 262.583 / 310.240 | 66.178 | 0.03 | same-image invariant |

AllHead T2/T3 layer-level Jaccard 평균 0.866747, 선택 집합이 달라진 layer-dialogue 비율 1.000000. 129,952 layer-dialogue pairs이며 image-macro 집계는 selection_image_summary.csv에 별도 보존했다. 모든 질문의 선택이 반드시 달라야 한다는 조건은 적용하지 않았다.

Selector 열은 AllHead의 전체 layer selector host 합과 B의 native prefix-selector wall로 scope가 다르며 중첩 stage를 TTFT에 더하지 않는다. AllHead는 layer별 live Q와 전체 attention heads의 causal score, full K read, covering V chunks를 사용한다. KV attention 예산 25%가 SSD read 25%를 뜻하지 않는다. MPIC는 첫 32개 이미지 rows를 재계산하면서 전체 context를 유지한다. 기존 MPIC/AllHead 알고리즘은 변경하지 않았다.

## 추정과 재현의 한계

Paired CI는 seed1234, image-cluster bootstrap 10,000회이며 같은 이미지의 모든 대화·turn·방법을 함께 resample한다. Point estimate와 bootstrap 모두 pooled sum/count로 요청 가중치를 보존한다. Setup image-macro는 별도 표기했다. Bootstrap은 독립 반복 실험이 아니며 CI가 0을 포함해도 동등성을 증명하지 않는다. 평균 개선을 p95 개선으로 일반화하지 않는다. B는 중요도 상위 25% content만 저장하고 attention에 사용한다. LLaVA의 기존 dense GPU buffer를 유지하므로 GPU 메모리 75% 감소를 뜻하지 않는다. POSIX_FADV_DONTNEED 성공을 기록했지만 cold NAND를 보장하지 않는다. 단일 GPU·batch1·한 이미지씩 제한된 저장 정책이며 cache saturation/hit-rate 또는 병렬 throughput의 검증이 아니다.

기존 artifact 보호: PASS, 108,088개. 기존 source 변경 0건. 8 MiB 초과 기존 파일은 9×64 KiB fingerprint + inode/mtime/ctime이며 full-byte SHA 증명이 아니다. 작은 기존 artifact/source와 새 payload/raw/manifest에는 full SHA를 사용했다. 완료 이미지 payload는 독립 audit·hash·atomic commit 후 소유 allowlist로 정리했으며 재실행에는 재생성이 필요하다.

새 실행 명령, gate 조건과 payload 재생성 절차는 REPRODUCE.md, 실제 source/config/모델/환경 freeze는 run의 해당 JSON 및 source_frozen/을 참조한다. 실행 시작 또는 partial 상태를 full main 완료로 해석하지 않는다.

## 필수 세부 계측 및 session 비교 보충

단위: MB=10^6 bytes, ms, 정확도 차이=percentage points. 모든 표는 이번 실행의 독립 감사 CSV와 commit된 raw/event에서 생성했다. 평균·분위수는 별도이며 중첩 component interval을 TTFT/E2E에 더하지 않는다.

### llava: 실제 공유-stream과 standalone 환산

실제 physical shared-image stream 비용 귀속

| Method | Request E2E sum mean | Persistence attribution mean | Activation attribution mean | Total mean / p50 / p95 | Δ vs ReComp [95% CI] | Setup provenance |
|---|---|---|---|---|---|---|
| ReComp | 1675.612 | 0.000 | 0.000 | 1675.612 / 1705.106 / 1903.654 | +0.000 [+0.000, +0.000] | MEASURED_NO_KV_SETUP |
| FullLoad | 2174.394 | 184.161 | 63.884 | 2422.438 / 2197.276 / 4668.847 | +746.826 [+730.550, +764.358] | MEASURED |
| MPIC-32 | 2195.974 | 179.559 | 57.708 | 2433.241 / 2252.506 / 4557.078 | +757.629 [+742.777, +773.641] | MEASURED |
| AllHead | 2374.652 | 0.000 | 67.583 | 2442.235 / 2408.341 / 3153.862 | +766.623 [+753.893, +779.223] | MEASURED_SHARED/DERIVED_FROM_SHARED |
| B | 1134.181 | 67.109 | 15.522 | 1216.812 / 1148.192 / 1974.521 | -458.800 [-470.517, -446.719] | MEASURED |

Derived standalone-equivalent: dialogue마다 measured image setup 재귀속

| Method | Request E2E sum mean | Persistence attribution mean | Activation attribution mean | Total mean / p50 / p95 | Δ vs ReComp [95% CI] | Setup provenance |
|---|---|---|---|---|---|---|
| ReComp | 1675.612 | 0.000 | 0.000 | 1675.612 / 1705.106 / 1903.654 | +0.000 [+0.000, +0.000] | MEASURED_NO_KV_SETUP |
| FullLoad | 2174.394 | 1864.666 | 653.509 | 4692.568 / 4647.548 / 5891.214 | +3016.956 [+2961.363, +3073.857] | MEASURED |
| MPIC-32 | 2195.974 | 1812.999 | 590.565 | 4599.538 / 4516.865 / 6118.470 | +2923.926 [+2858.813, +2992.708] | MEASURED |
| AllHead | 2374.652 | 1864.666 | 692.528 | 4931.846 / 4889.160 / 6196.668 | +3256.234 [+3198.296, +3315.031] | MEASURED_SHARED/DERIVED_FROM_SHARED |
| B | 1134.181 | 683.286 | 158.858 | 1976.325 / 1968.424 / 2414.367 | +300.712 [+288.335, +314.206] | MEASURED |

Standalone 표는 shared-store에서 측정한 request에 setup을 재귀속한 파생값이다. AllHead persistence는 같은 canonical bundle의 DERIVED_FROM_SHARED 값이다. Non-source T1의 가상 capture 증분은 NOT_MEASURED이므로 완전한 cold-session 실측이 아니다.

### llava: 저장/setup 분포와 paired 차이

| Method | Valid content rows mean / p50 / p95 | Valid content MB mean / p50 / p95 | Total store MB mean / p50 / p95 | OS write MB mean / p50 / p95 | Persistence ms mean / p50 / p95 | Activation ms mean / p50 / p95 |
|---|---|---|---|---|---|---|
| FullLoad | 2193.528 / 2256.000 / 2400.000 | 1150.040 / 1182.794 / 1258.291 | 1264.667 / 1304.799 / 1382.313 | 1264.667 / 1304.799 / 1382.313 | 1879.086 / 1729.867 / 2992.428 | 651.845 / 651.665 / 791.649 |
| MPIC-32 | 2193.528 / 2256.000 / 2400.000 | 1150.040 / 1182.794 / 1258.291 | 1187.802 / 1223.693 / 1299.042 | 1187.802 / 1223.693 / 1299.042 | 1832.132 / 1625.216 / 3231.694 | 588.829 / 596.684 / 652.437 |
| AllHead | 2193.528 / 2256.000 / 2400.000 | 1150.040 / 1182.794 / 1258.291 | 1264.667 / 1304.799 / 1382.313 | 1264.667 / 1304.799 / 1382.313 | 1879.086 / 1729.867 / 2992.428 | 689.584 / 698.362 / 795.630 |
| B | 548.382 / 564.000 / 600.000 | 287.510 / 295.698 / 314.573 | 309.676 / 320.452 / 338.021 | 309.676 / 320.452 / 338.021 | 684.747 / 651.360 / 983.746 | 158.378 / 160.104 / 178.521 |

| B − baseline | Scope | Metric / unit | Mean difference [95% CI] |
|---|---|---|---|
| ReComp | followup | ttft_ms / ms | -270.660 [-276.325, -264.802] |
| ReComp | followup | request_e2e_ms / ms | -270.292 [-275.967, -264.430] |
| ReComp | followup | ssd_read_bytes / MB | +308.050 [+304.174, +311.985] |
| ReComp | image_macro_setup | file_logical_bytes / MB | +309.676 [+306.355, +312.910] |
| ReComp | image_macro_setup | allocated_bytes / MB | +309.682 [+306.361, +312.916] |
| ReComp | image_macro_setup | os_write_bytes / MB | +309.676 [+306.355, +312.910] |
| ReComp | image_macro_setup | service_persistence_ms / ms | +684.747 [+671.258, +698.939] |
| ReComp | image_macro_setup | activation_ms / ms | +158.378 [+156.740, +160.007] |
| FullLoad | followup | ttft_ms / ms | -519.129 [-528.140, -510.270] |
| FullLoad | followup | request_e2e_ms / ms | -519.361 [-528.339, -510.520] |
| FullLoad | followup | ssd_read_bytes / MB | -865.869 [-876.789, -854.999] |
| FullLoad | image_macro_setup | file_logical_bytes / MB | -954.991 [-965.064, -944.808] |
| FullLoad | image_macro_setup | allocated_bytes / MB | -955.017 [-965.091, -944.835] |
| FullLoad | image_macro_setup | os_write_bytes / MB | -954.991 [-965.064, -944.808] |
| FullLoad | image_macro_setup | service_persistence_ms / ms | -1194.339 [-1243.857, -1148.198] |
| FullLoad | image_macro_setup | activation_ms / ms | -493.467 [-500.117, -486.811] |
| MPIC-32 | followup | ttft_ms / ms | -530.540 [-538.700, -522.310] |
| MPIC-32 | followup | request_e2e_ms / ms | -530.311 [-538.432, -522.107] |
| MPIC-32 | followup | ssd_read_bytes / MB | -866.131 [-877.051, -855.261] |
| MPIC-32 | image_macro_setup | file_logical_bytes / MB | -878.126 [-887.373, -868.733] |
| MPIC-32 | image_macro_setup | allocated_bytes / MB | -878.126 [-887.373, -868.733] |
| MPIC-32 | image_macro_setup | os_write_bytes / MB | -878.126 [-887.373, -868.733] |
| MPIC-32 | image_macro_setup | service_persistence_ms / ms | -1147.385 [-1204.289, -1093.935] |
| MPIC-32 | image_macro_setup | activation_ms / ms | -430.451 [-435.132, -425.661] |
| AllHead | followup | ttft_ms / ms | -614.922 [-625.320, -604.567] |
| AllHead | followup | request_e2e_ms / ms | -619.729 [-630.148, -609.407] |
| AllHead | followup | ssd_read_bytes / MB | -876.070 [-887.379, -865.030] |
| AllHead | image_macro_setup | file_logical_bytes / MB | -954.991 [-965.064, -944.808] |
| AllHead | image_macro_setup | allocated_bytes / MB | -955.017 [-965.091, -944.835] |
| AllHead | image_macro_setup | os_write_bytes / MB | -954.991 [-965.064, -944.808] |
| AllHead | image_macro_setup | service_persistence_ms / ms | -1194.339 [-1243.857, -1148.198] |
| AllHead | image_macro_setup | activation_ms / ms | -531.206 [-538.518, -523.639] |

### llava: Followup H2D / online component 계측

| Method | KV H2D MB | Payload compute H2D MB | Rater H2D MB | H2D ms | Layer selector host ms | Prefix selector ms | Cache assembly ms | Scatter ms |
|---|---|---|---|---|---|---|---|---|
| ReComp | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| FullLoad | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| MPIC-32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 296.822 | NOT_MEASURED | NOT_MEASURED | 14.610 | NOT_MEASURED |
| AllHead | NOT_MEASURED | 740.984 | 18.342 | NOT_MEASURED | 826.306 | NOT_MEASURED | 1.966 | NOT_MEASURED |
| B | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 0.030 | NOT_MEASURED | 82.832 |

H2D 열은 각 native 계측의 범위이며 서로 합산하지 않는다. FullLoad/B/MPIC에서 tensor bytes가 기록되지 않은 항목은 NOT_MEASURED다. MPIC H2D 시간은 동기 전송 실측이며 tensor bytes 또는 bus counter와 구분한다.

### llava: Activation auxiliary residency와 service 무결성 검사

| Method | Metric | Measured images | Mean / p50 / p95 (bytes or ms as named) |
|---|---|---|---|
| FullLoad | host_tensor_bytes | 398 | 2621440.000 / 2621440.000 / 2621440.000 |
| FullLoad | gpu_tensor_bytes | 398 | 0.000 / 0.000 / 0.000 |
| FullLoad | visual_payload_resident_bytes | 398 | 0.000 / 0.000 / 0.000 |
| FullLoad | metadata_ready_file_bytes | 398 | 2645826.264 / 2646471.000 / 2647726.900 |
| FullLoad | service_integrity_read_bytes | 398 | 1264650545.736 / 1304782440.000 / 1382296399.900 |
| FullLoad | unused_bundle_integrity_read_bytes | 0 | NOT_MEASURED |
| FullLoad | service_integrity_ms | 398 | 650.655 / 650.453 / 790.406 |
| FullLoad | unused_bundle_integrity_ms | 0 | NOT_MEASURED |
| FullLoad | activation_outer_wall_ms | 398 | 651.845 / 651.665 / 791.649 |
| MPIC-32 | host_tensor_bytes | 0 | NOT_MEASURED |
| MPIC-32 | gpu_tensor_bytes | 398 | 0.000 / 0.000 / 0.000 |
| MPIC-32 | visual_payload_resident_bytes | 398 | 0.000 / 0.000 / 0.000 |
| MPIC-32 | metadata_ready_file_bytes | 398 | 42103.995 / 43288.000 / 45676.050 |
| MPIC-32 | service_integrity_read_bytes | 398 | 1187790921.683 / 1223682328.000 / 1299030636.050 |
| MPIC-32 | unused_bundle_integrity_read_bytes | 0 | NOT_MEASURED |
| MPIC-32 | service_integrity_ms | 398 | 583.015 / 591.414 / 646.066 |
| MPIC-32 | unused_bundle_integrity_ms | 0 | NOT_MEASURED |
| MPIC-32 | activation_outer_wall_ms | 398 | 588.829 / 596.684 / 652.437 |
| AllHead | host_tensor_bytes | 398 | 20894498.734 / 21446656.000 / 22605824.000 |
| AllHead | gpu_tensor_bytes | 398 | 0.000 / 0.000 / 0.000 |
| AllHead | visual_payload_resident_bytes | 398 | 0.000 / 0.000 / 0.000 |
| AllHead | metadata_ready_file_bytes | 398 | 20920069.997 / 21472872.000 / 22633295.900 |
| AllHead | service_integrity_read_bytes | 398 | 1209831369.535 / 1248306792.000 / 1322343247.900 |
| AllHead | unused_bundle_integrity_read_bytes | 398 | 54819176.201 / 56475648.000 / 59953152.000 |
| AllHead | service_integrity_ms | 398 | 626.191 / 625.881 / 711.730 |
| AllHead | unused_bundle_integrity_ms | 398 | 50.793 / 57.080 / 80.289 |
| AllHead | activation_outer_wall_ms | 398 | 689.584 / 698.362 / 795.630 |
| B | host_tensor_bytes | 398 | 2621440.000 / 2621440.000 / 2621440.000 |
| B | gpu_tensor_bytes | 398 | 0.000 / 0.000 / 0.000 |
| B | visual_payload_resident_bytes | 398 | 0.000 / 0.000 / 0.000 |
| B | metadata_ready_file_bytes | 398 | 2685199.397 / 2687023.500 / 2690942.500 |
| B | service_integrity_read_bytes | 398 | 309660197.789 / 320435783.500 / 338004988.900 |
| B | unused_bundle_integrity_read_bytes | 0 | NOT_MEASURED |
| B | service_integrity_ms | 398 | 152.026 / 153.469 / 171.651 |
| B | unused_bundle_integrity_ms | 0 | NOT_MEASURED |
| B | activation_outer_wall_ms | 398 | 158.378 / 160.104 / 178.521 |

Host tensor bytes는 structural/prefix와 AllHead의 v_hidden 등 실제 상주 텐서이며 단순 metadata bytes가 아니다. MPIC metadata-ready file bytes는 읽은 metadata 파일 크기이며 Python 객체 heap 크기가 아니다. Visual KV payload residency는 별도 열로 기록한다. AllHead의 미사용 probe sidecar는 activation integrity scan에서 검증되지만 followup serving read는 0이다. Service integrity interval은 activation outer wall에 이미 포함되며 중복 가산하지 않는다. 종료 후 독립 audit의 rehash는 이 service 비용 밖이다.

| MPIC measured component | N | Mean / p50 / p95 |
|---|---|---|
| n_image_tokens | 8122 | 2239.073 / 2340.000 / 2544.000 |
| n_recomputed_image_tokens | 8122 | 32.000 / 32.000 / 32.000 |
| n_reused_image_tokens | 8122 | 2207.073 / 2308.000 / 2512.000 |
| n_recomputed_text_tokens | 8122 | 70.789 / 70.000 / 90.000 |
| embedding_read_ms | 8122 | 2.330 / 0.262 / 32.769 |
| kv_read_ms | 8122 | 377.159 / 378.010 / 457.782 |
| h2d_ms | 8122 | 296.822 / 304.422 / 341.517 |
| cache_assembly_ms | 8122 | 14.610 / 14.992 / 16.366 |
| decode_cache_assembly_ms | 8122 | 3.642 / 3.714 / 4.070 |
| selective_prefill_interval_ms | 8122 | 788.140 / 795.501 / 937.097 |
| ssd_embedding_bytes | 8122 | 262144.000 / 262144.000 / 262144.000 |
| metadata_resident_bytes | 8122 | 42247.995 / 43955.000 / 47517.000 |

MPIC의 32는 이미지 rows 수이며 32%가 아니다. Selective-prefill interval은 read/H2D/assembly와 active-row decoder를 포함한 중첩 interval이다. 순수 recompute GPU 시간은 NOT_MEASURED이며 위 component를 합해 TTFT를 만들지 않는다.

### llava: AllHead / B V chunk coverage와 read amplification

| Method | Attended content fraction mean / p50 / p95 | V chunk coverage mean / p50 / p95 | V read amplification request mean / p50 / p95 | Pooled byte ratio | Image-macro mean ratio |
|---|---|---|---|---|---|
| AllHead | 0.250 / 0.250 / 0.250 | 0.984 / 0.988 / 0.997 | 4.003 / 4.017 / 4.062 | 4.003380 | 4.002800 |
| B | 0.250 / 0.250 / 0.250 | NOT_APPLICABLE: sequential full B payload | 1.000 / 1.000 / 1.000 | 1.000000 | 1.000000 |

V amplification 분모는 original N에 대해 선택한 정확한 k content rows의 native V bytes다. 분자는 실제 v.bin read이며 AllHead에서는 covering chunks의 미선택·구조·padding rows도 포함할 수 있다. 별도 structural sidecar read는 이 V 비율 밖에 있고 total SSD read에는 포함된다. Coverage는 layer별 선택 V chunk 수 / ceil(full native visual rows/64)를 먼저 계산한 뒤 요청·layer를 동일 가중한다. B payload는 정확한 k content rows를 전부 읽으므로 이 content-V 비율은 1이다.

### 실행 wall time의 범위

모델 로딩·warmup 이후 smoke와 main을 포함한 launcher wall time: 50184.428 s (13.940 h). Conditioning/logging/이미지 감사/cleanup을 포함하며 request E2E와 setup 귀속 합과 다르다. GPU 사전검증 및 최종 독립 통계/보존 검사는 이 wall time 밖이다.

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

## 측정된 store bytes에 근거한 용량 추정

종료 시점 여유 175.146 GiB에서 30 GiB reserve를 남긴 가정상 KV 전용 여유는 145.146 GiB다. 기존 raw·검증 payload·다른 파일은 그대로 두며 추가 쓰기가 없다는 가정이다. 이미지별 실제 allocated bytes를 사용했다.

| Method | 398-image stores GiB | Mean-size capacity estimate (images) | Fits KV-only headroom |
|---|---:|---:|---|
| llava / fullload | 468.781 | 123 | False |
| llava / mpic32_ssd | 440.280 | 131 | False |
| llava / sparsevlm_ssd_kv25_allhead | 468.781 | 123 | False |
| llava / ours_prefixstore_kv25 | 114.789 | 503 | True |

FullLoad/AllHead 행은 같은 canonical bundle을 각 단독 배포의 footprint로 표시한 것이며 실제 물리 총량에 두 번 더하지 않는다. AllHead 미사용 probe sidecar도 포함한다. Mean-size capacity는 이미지 크기 분포가 같다는 환산이며, 실제 workload 전체의 sum과 fit 여부를 별도 표시했다. 현재 실험은 한 이미지씩 생성·감사·정리했으므로 이것은 실제 동시 적재·eviction·hit-rate 실험 결과가 아니다.
