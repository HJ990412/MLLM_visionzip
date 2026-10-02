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

Paired CI는 seed1234, image-cluster bootstrap 10,000회이며 같은 이미지의 모든 대화·turn·방법을 함께 resample한다. Point estimate와 bootstrap 모두 pooled sum/count로 요청 가중치를 보존한다. Setup image-macro는 별도 표기했다. Bootstrap은 독립 반복 실험이 아니며 CI가 0을 포함해도 동등성을 증명하지 않는다. 평균 개선을 p95 개선으로 일반화하지 않는다. 상위 25% pruning이나 GPU 메모리 75% 감소를 구현한 실험이 아니다. POSIX_FADV_DONTNEED 성공을 기록했지만 cold NAND를 보장하지 않는다. 단일 GPU·batch1·한 이미지씩 제한된 저장 정책이며 cache saturation/hit-rate 또는 병렬 throughput의 검증이 아니다.

기존 artifact 보호: PASS, 108,088개. 기존 source 변경 0건. 8 MiB 초과 기존 파일은 9×64 KiB fingerprint + inode/mtime/ctime이며 full-byte SHA 증명이 아니다. 작은 기존 artifact/source와 새 payload/raw/manifest에는 full SHA를 사용했다. 완료 이미지 payload는 독립 audit·hash·atomic commit 후 소유 allowlist로 정리했으며 재실행에는 재생성이 필요하다.

새 실행 명령, gate 조건과 payload 재생성 절차는 REPRODUCE.md, 실제 source/config/모델/환경 freeze는 run의 해당 JSON 및 source_frozen/을 참조한다. 실행 시작 또는 partial 상태를 full main 완료로 해석하지 않는다.
