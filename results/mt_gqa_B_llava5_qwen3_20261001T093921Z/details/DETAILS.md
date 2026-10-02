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
