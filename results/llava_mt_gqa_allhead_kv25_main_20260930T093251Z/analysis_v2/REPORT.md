# LLaVA MT-GQA AllHead / KV25 본실험

MAIN 5-ARM: **VALID**, 실제 final 요청 **60,915/60,915**. READY FOR PAPER MAIN TABLE: **YES**.

기존 frozen MT-GQA-reconstructed의 method-local generated history이며 strict normalized exact match를 쓴다. 공식 MetaCompress artifact/evaluator 재현은 아니다.

| Method | Acc T1/T2/T3 (%) | All Acc (%) | Hit Acc (%) | T1 TTFT (ms) | Hit TTFT mean/p50/p95 (ms) | SSD MB/hit |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 63.19/68.11/68.85 | 66.72 | 68.48 | 525.14 | 533.56/535.41/607.21 | 0.00 |
| FullLoad | 63.19/68.06/68.70 | 66.65 | 68.38 | 524.07 | 749.21/740.26/911.56 | 1173.92 |
| MPIC-32 | 63.19/68.01/69.10 | 66.77 | 68.55 | 523.83 | 766.71/779.27/901.90 | 1174.18 |
| SparseVLM-SSD-KV25-AllHead | 63.19/67.87/68.70 | 66.58 | 68.28 | 523.28 | 852.15/847.47/1010.99 | 1184.12 |
| Ours-KV25 | 63.19/65.70/66.66 | 65.18 | 66.18 | 523.70 | 269.52/266.55/317.09 | 320.45 |

| Method | Logical retention | Full-K/scoring MB | Selected-K MB | Selected-V MB | Structural/other MB | V chunk coverage | Preads/hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 1.000000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.00 |
| FullLoad | 1.000000 | 0.000 | 586.959 | 586.959 | 0.000 | 1.0000 | 64.00 |
| MPIC-32 | 1.000000 | 0.000 | 586.959 | 586.959 | 0.262 | 1.0000 | 65.00 |
| SparseVLM-SSD-KV25-AllHead | 0.250000 | 586.959 | 0.000 | 577.734 | 19.427 | 0.9844 | 78.31 |
| Ours-KV25 | 0.250000 | 0.000 | 150.512 | 150.512 | 19.427 | 0.2538 | 65.00 |

AllHead는 layer별 full K를 한 번 읽고 scoring과 answer attention에 재사용한다. selected-K 추가 읽기와 probe 읽기는 0이며 재사용 K를 bytes 합계에 다시 더하지 않는다. Structural sidecar의 실제 중복 K는 별도 기록한다. MPIC-32는 full context에서 32개 이미지 token을 재계산한다.

| Ours − baseline | Δ Hit Acc (%p), 95% CI | Δ Hit TTFT (ms), 95% CI | TTFT 감소율 (%) | SSD 감소율 (%) |
|---|---:|---:|---:|---:|
| ReComp | -2.302 [-3.052, -1.540] | -264.038 [-269.784, -258.262] | 49.49 | N/A (ReComp SSD=0) |
| FullLoad | -2.204 [-2.951, -1.451] | -479.687 [-488.239, -471.266] | 64.03 | 72.70 |
| MPIC-32 | -2.376 [-3.135, -1.607] | -497.185 [-505.273, -489.057] | 64.85 | 72.71 |
| SparseVLM-SSD-KV25-AllHead | -2.105 [-2.868, -1.352] | -582.630 [-592.711, -572.448] | 68.37 | 72.94 |

CI는 seed=1234, image-cluster bootstrap 10,000회이고 image의 모든 대화·turn·방법을 함께 resample한다. Point estimate와 동일한 요청 가중 평균이다. CI의 0 포함은 동등성 증거가 아니다. History 차이를 포함한 method-level 비교다.

| Method | Provisioning | Persistence | Actual-stream session E2E (ms) | Derived standalone E2E (ms) |
|---|---|---|---:|---:|
| ReComp | FRESH_IMAGE_STREAMING | MEASURED (0.00 ms/image) | 1676.81 | 1676.81 |
| FullLoad | FRESH_IMAGE_STREAMING | MEASURED (1195.77 ms/image) | 2226.64 | 3310.46 |
| MPIC-32 | FRESH_IMAGE_STREAMING | MEASURED (1110.09 ms/image) | 2252.50 | 3268.35 |
| SparseVLM-SSD-KV25-AllHead | FRESH_IMAGE_STREAMING | MEASURED_SHARED_WITH_FULLLOAD (1195.77 ms/image) | 2385.28 | 4159.56 |
| Ours-KV25 | FRESH_IMAGE_STREAMING | MEASURED (1136.48 ms/image) | 1260.08 | 2293.38 |

FullLoad와 AllHead는 실제 각자의 source T1에서 FP16 K/V bits·prefix·v_hidden의 동일성을 검증한 canonical store를 공유한다. 기존 검증된 raster serializer가 함께 만드는 미사용 probe sidecar의 쓰기 비용도 canonical persistence에 포함되며, base-only 비용은 별도 측정하지 않았다. 실제 build는 이미지당 한 번 FullLoad source dialogue에 귀속된다. AllHead의 독립 배포와 모든 dialogue의 standalone 비용은 DERIVED이다. T1 capture/exit materialization은 request E2E에 이미 들어 있으므로 다시 더하지 않는다.

True TTFT는 prompt 작성 전부터 JPEG 읽기/디코딩(픽셀 요청), tokenizer/processor, SSD read/H2D/assembly/prefill, 첫 token materialization과 CUDA sync까지다. 응답 decode 완료 E2E와 구분한다. DONTNEED는 타이머 밖에서 파일별 성공을 기록하며 NAND/controller cold를 보장하지 않는다. Metadata activation 및 hash validation은 별도 setup이다. AllHead host/CUDA stage 구간은 중첩되므로 합산하지 않는다.

두 KV25는 ceil(N/4)의 실제 content attention 예산이며 full original payload를 SSD에 유지한다. Dense GPU cache를 사용하므로 75% GPU 메모리 절감 주장을 하지 않는다. 이미지별 streaming은 동시성/throughput/cache saturation 검증이 아니다. 효율 결론은 이 fixed-budget SSD adaptation과 layout/cache 조건에 한정한다.

독립 감사: PASS; AllHead K-read 재사용: PASS; Qwen GPU / MT-VQA: NOT RUN.
보호 검사: PASS; 기존 source 변경 0건, 기존 artifact 변경 0건. 큰 기존 파일은 inode/mtime/ctime + 9-window fingerprint 정책이며 전체-byte SHA256 검증으로 과장하지 않는다.

이미지별 N/k/read amplification, selection 변화, cap/실패/retry/중복, metadata와 GPU memory, 중첩 timing은 함께 저장된 CSV/JSON 및 image raw를 참조한다. Validation/smoke/retry는 main final counts와 별도다.

추가 측정 범위는 extra_rows_bytes.csv와 persistence_stage_scope.csv에 있다. Ours extra-row 진단은 실제 unused_loaded_real_rows 필드로 재계산했으며 원 분석의 누락 필드 fallback을 정정했다. 원 분석 산출물은 보존했다. 추론/config/주요 A–D 지표는 바뀌지 않았다. Capture materialization은 source T1 E2E에 포함되고, 이미지 경계 독립 감사·hash·결과 logging은 service-session E2E 밖이다. Metadata activation의 outer wall에는 activation 자체의 hash 검증이 포함된다.
