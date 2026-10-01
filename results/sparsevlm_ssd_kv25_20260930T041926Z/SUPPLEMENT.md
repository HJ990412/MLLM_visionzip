# SparseVLM-SSD-KV25 보충 분석

이 문서는 측정 완료 후 기존 raw/독립 audit/activation receipt를 읽어 만든 supplemental analysis다. 동결된 production code, 계약, raw, 선택/측정 조건은 변경하지 않았다. 기존 REPORT의 판정은 그대로이며, 아래 보충 검산이 실패하면 해당 사실을 별도로 표시한다.

99 independent audit: **PASS**; supplemental input/adoption checks: **PASS**.

## Turn별 정확도

GQA/smoke는 기존 normalized equality 또는 gold-token-prefix scorer, MT는 기존 strict normalized equality scorer를 사용한다. 아래 값은 99가 raw에서 독립 재계산한 summary.csv를 그대로 표시한다.

| Dataset / 방법 | N | Hit N | 전체 % | Hit % | T1 % | T2 % | T3 % | T4 % | T5 % | T6 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke / ReComp | 12 | 8 | 83.33 | 75.00 | 100.00 | 100.00 | 50.00 | NOT RUN | NOT RUN | NOT RUN |
| smoke / FullLoad | 12 | 8 | 83.33 | 75.00 | 100.00 | 100.00 | 50.00 | NOT RUN | NOT RUN | NOT RUN |
| smoke / SparseVLM-SSD-KV25-Probe3 | 12 | 8 | 83.33 | 75.00 | 100.00 | 100.00 | 50.00 | NOT RUN | NOT RUN | NOT RUN |
| smoke / SparseVLM-SSD-KV25-AllHead | 12 | 8 | 83.33 | 75.00 | 100.00 | 100.00 | 50.00 | NOT RUN | NOT RUN | NOT RUN |
| smoke / Ours-KV25 | 12 | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | NOT RUN | NOT RUN | NOT RUN |
| gqa / ReComp | 240 | 200 | 62.50 | 63.00 | 60.00 | 62.50 | 65.00 | 70.00 | 47.50 | 70.00 |
| gqa / FullLoad | 240 | 200 | 62.50 | 63.00 | 60.00 | 57.50 | 65.00 | 72.50 | 47.50 | 72.50 |
| gqa / SparseVLM-SSD-KV25-Probe3 | 240 | 200 | 60.00 | 60.00 | 60.00 | 57.50 | 60.00 | 67.50 | 47.50 | 67.50 |
| gqa / SparseVLM-SSD-KV25-AllHead | 240 | 200 | 61.67 | 62.00 | 60.00 | 57.50 | 62.50 | 70.00 | 47.50 | 72.50 |
| gqa / Ours-KV25 | 240 | 200 | 57.08 | 56.50 | 60.00 | 62.50 | 60.00 | 47.50 | 42.50 | 70.00 |
| mt / ReComp | 120 | 80 | 75.83 | 78.75 | 70.00 | 77.50 | 80.00 | NOT RUN | NOT RUN | NOT RUN |
| mt / FullLoad | 120 | 80 | 75.83 | 78.75 | 70.00 | 77.50 | 80.00 | NOT RUN | NOT RUN | NOT RUN |
| mt / SparseVLM-SSD-KV25-Probe3 | 120 | 80 | 76.67 | 80.00 | 70.00 | 80.00 | 80.00 | NOT RUN | NOT RUN | NOT RUN |
| mt / SparseVLM-SSD-KV25-AllHead | 120 | 80 | 75.83 | 78.75 | 70.00 | 77.50 | 80.00 | NOT RUN | NOT RUN | NOT RUN |
| mt / Ours-KV25 | 120 | 80 | 74.17 | 76.25 | 70.00 | 75.00 | 77.50 | NOT RUN | NOT RUN | NOT RUN |

## Selector와 실제 attention 측정 범위

아래 ms는 hit 요청당 평균이다. Selector host 범위는 layer별 selector interval의 합으로서 score/K acquisition/selection/assembly를 포함할 수 있다. Actual attention은 해당 eager call 범위이며 host는 launch/호출 구간, CUDA는 event 구간이다. 서로 겹치는 항목을 더해 TTFT를 만들지 않는다. 전체 LLM prefill·MLP·projection·decode E2E와 동일한 범위도 아니다. 기존 control arms의 분리된 actual-attention 계측이 없으면 NOT_INSTRUMENTED로 표시한다.

| Dataset / 방법 | Selector host ms | Prefill attention host/CUDA ms | Decode attention host/CUDA ms | 계측 |
|---|---:|---:|---:|---|
| smoke / ReComp | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| smoke / FullLoad | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| smoke / SparseVLM-SSD-KV25-Probe3 | 905.499 | 5.269 / 3.783 | 2.910 / 1.578 | MEASURED |
| smoke / SparseVLM-SSD-KV25-AllHead | 900.473 | 5.185 / 3.579 | 2.911 / 1.595 | MEASURED |
| smoke / Ours-KV25 | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| gqa / ReComp | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| gqa / FullLoad | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| gqa / SparseVLM-SSD-KV25-Probe3 | 836.321 | 5.309 / 4.010 | 3.114 / 1.778 | MEASURED |
| gqa / SparseVLM-SSD-KV25-AllHead | 849.228 | 5.112 / 3.657 | 3.119 / 1.802 | MEASURED |
| gqa / Ours-KV25 | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| mt / ReComp | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| mt / FullLoad | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |
| mt / SparseVLM-SSD-KV25-Probe3 | 777.557 | 5.311 / 5.422 | 3.360 / 1.927 | MEASURED |
| mt / SparseVLM-SSD-KV25-AllHead | 813.401 | 5.160 / 5.437 | 3.316 / 1.928 | MEASURED |
| mt / Ours-KV25 | NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT MEASURED / NOT MEASURED | NOT_INSTRUMENTED |

Projection별 실제 prefill/decode 호출 수는 `attention_interval_summary.csv`에 요청당 전체 layer 합으로 제공한다. GPU gate의 layer별 prefill은 각 q/k/v 1회이며, 측정 raw의 실제 counter도 독립 audit 대상이다. H2D visual payload subtotal과 rater visual bytes는 별도이며 system/suffix/processor를 포함한 전체 H2D로 해석하지 않는다.

## Metadata activation

같은 metadata_activation 값이 hit마다 raw에 반복되어도 한 번의 context activation으로 센다. 집계 단위는 adopted phase/dialogue-or-image/method context다. 같은 image의 별도 MT dialogue는 별도 context/provisioning일 수 있으므로 합치지 않는다. File bytes는 활성화 대상 metadata 파일 크기이며 실제 OS read 반환량으로 바꾸어 해석하지 않는다. Hash verified bytes 역시 metadata resident bytes와 다르다.

| Dataset / 방법 | Context N | Activation ms | Metadata file MB | Host tensor MB | GPU tensor MB | Hash verification ms | 상태 |
|---|---:|---:|---:|---:|---:|---:|---|
| smoke / ReComp | 4 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT_APPLICABLE |
| smoke / FullLoad | 4 | 0.929 | 2.645 | 2.621 | 0.000 | NOT MEASURED | MEASURED |
| smoke / SparseVLM-SSD-KV25-Probe3 | 4 | 678.943 | 20.866 | 20.840 | 0.000 | 664.332 | MEASURED |
| smoke / SparseVLM-SSD-KV25-AllHead | 4 | 626.058 | 20.866 | 20.840 | 0.000 | 612.284 | MEASURED |
| smoke / Ours-KV25 | 4 | 1.172 | 2.663 | 2.621 | 0.000 | NOT MEASURED | MEASURED |
| gqa / ReComp | 40 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT_APPLICABLE |
| gqa / FullLoad | 40 | 0.957 | 2.645 | 2.621 | 0.000 | NOT MEASURED | MEASURED |
| gqa / SparseVLM-SSD-KV25-Probe3 | 40 | 664.441 | 20.851 | 20.826 | 0.000 | 651.709 | MEASURED |
| gqa / SparseVLM-SSD-KV25-AllHead | 40 | 629.494 | 20.851 | 20.826 | 0.000 | 616.604 | MEASURED |
| gqa / Ours-KV25 | 40 | 1.155 | 2.663 | 2.621 | 0.000 | NOT MEASURED | MEASURED |
| mt / ReComp | 40 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT_APPLICABLE |
| mt / FullLoad | 40 | 1.011 | 2.646 | 2.621 | 0.000 | NOT MEASURED | MEASURED |
| mt / SparseVLM-SSD-KV25-Probe3 | 40 | 669.309 | 21.319 | 21.293 | 0.000 | 655.799 | MEASURED |
| mt / SparseVLM-SSD-KV25-AllHead | 40 | 678.211 | 21.319 | 21.293 | 0.000 | 664.331 | MEASURED |
| mt / Ours-KV25 | 40 | 1.244 | 2.664 | 2.621 | 0.000 | NOT MEASURED | MEASURED |

## 공유/독립 배포 저장공간

| 구분 | Apparent GB | 실제 allocated disk GB | 파일 수 |
|---|---:|---:|---:|
| Probe3_standalone | 50.422571 | 50.423718 | 4000 |
| AllHead_standalone | 48.238059 | 48.238354 | 2720 |
| shared_two_variants_actual | 50.422571 | 50.423718 | 4000 |
| probe_sidecar_only | 2.184511 | 2.185363 | 1280 |
| Ours_existing_store | 47.511720 | 47.511974 | 4000 |
| metadata_files | 0.834036 | 0.834331 | 120 |

두 새 방법이 공유한 실제 canonical 파일은 한 번만 계상한다. Probe3 독립 배포는 sidecar를 포함하고 AllHead 독립 배포는 probe sidecar를 제외한다. Read-only GQA serving은 CACHE-HIT REEVALUATION이며 전체 persistence/session은 NOT_REMEASURED다. MT fresh T1의 shared bundle persistence 측정은 존재하더라도 AllHead base와 Probe3-only 생성 비용이 자동 분리되지는 않는다.

별도 setup split receipt: [receipt.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/setup_split_01/receipt.json). 별도 진단을 수행했다면 원 5-arm timing에 소급 합산하지 않는다.

### 별도 fresh setup 측정

아래 값은 한 이미지의 새로운 정상 T1에서 얻은 canonical tensor를 사용하는 별도 serializer 진단이다. Production pilot의 shared-bundle persistence, cache-hit TTFT, session 비용을 대체하지 않는다. Base를 먼저 쓰고 동일 T1의 CPU K에서 Probe3 raw sidecar를 이어서 생성했으며 이 임시 setup residency를 serving residency로 해석하지 않는다.

| 단계 | Wall ms | Fsync inclusive | T1 포함 | 범위 |
|---|---:|---|---|---|
| AllHead_base_setup | 1498.431 | True | False | canonical full K/V + system K/V + visual embeddings + structural KV + base metadata; no probe files |
| Probe3_incremental_sidecar_setup | 75.222 | True | False | same_actual_T1_canonical_FP16_CPU_K_retained_from_immediately_preceding_base_phase |
| fresh_Probe3_setup_total_excluding_T1 | 1573.661 | N/A | False | actual sequential base+sidecar setup wall; one fresh T1 excluded |
| fresh_Probe3_setup_phase_sum | 1573.653 | N/A | False | sum of two non-overlapping persistence phases; one fresh T1 excluded |
| source_T1_TTFT | 824.277 | N/A | True | actual normal full-image T1; request time separate from setup |
| source_T1_request_E2E | 865.995 | N/A | True | actual normal full-image T1; TTFT is contained within E2E |
| independent_readback_excluded | 1307.926 | N/A | False | excluded from persistence phases |
| hash_verification_excluded | 577.709 | N/A | False | excluded from persistence phases |

추가 범위 제한: One fixed image diagnostic, not a pilot-mean or deployment benchmark. Separate run-owned serializer; production pilot persistence remains its measured shared bundle. Canonical CPU K is retained temporarily across the two setup phases; this is not serving residency. Base first then sidecar order is fixed; no timing comparison between repeated independent builds. Hash/readback/activation audit time is excluded from persistence phase wall times and reported separately. Neither OS write completion nor fsync establishes NAND/internal controller traffic or erase cost.

## Cap / rater fallback / attempt 채택

| Dataset / 방법 | N / Hit N | Explicit cap true / unrecorded | Generated length=16 | Rater fallback / eligible hits | Nonfinite fields |
|---|---:|---:|---:|---:|---:|
| gqa / FullLoad | 240 / 200 | 0 / 0 | 0 | 0 / 0 | 0 |
| gqa / Ours-KV25 | 240 / 200 | 0 / 0 | 0 | 0 / 0 | 0 |
| gqa / ReComp | 240 / 200 | 0 / 0 | 0 | 0 / 0 | 0 |
| gqa / SparseVLM-SSD-KV25-AllHead | 240 / 200 | 0 / 0 | 0 | 0 / 200 | 0 |
| gqa / SparseVLM-SSD-KV25-Probe3 | 240 / 200 | 0 / 0 | 0 | 0 / 200 | 0 |
| mt / FullLoad | 120 / 80 | 0 / 0 | 0 | 0 / 0 | 0 |
| mt / Ours-KV25 | 120 / 80 | 0 / 0 | 0 | 0 / 0 | 0 |
| mt / ReComp | 120 / 80 | 0 / 0 | 0 | 0 / 0 | 0 |
| mt / SparseVLM-SSD-KV25-AllHead | 120 / 80 | 0 / 0 | 0 | 0 / 80 | 0 |
| mt / SparseVLM-SSD-KV25-Probe3 | 120 / 80 | 0 / 0 | 0 | 0 / 80 | 0 |
| smoke / FullLoad | 12 / 8 | 0 / 0 | 0 | 0 / 0 | 0 |
| smoke / Ours-KV25 | 12 / 8 | 0 / 0 | 0 | 0 / 0 | 0 |
| smoke / ReComp | 12 / 8 | 0 / 0 | 0 | 0 / 0 | 0 |
| smoke / SparseVLM-SSD-KV25-AllHead | 12 / 8 | 0 / 0 | 0 | 0 / 8 | 0 |
| smoke / SparseVLM-SSD-KV25-Probe3 | 12 / 8 | 0 / 0 | 0 | 0 / 8 | 0 |

요청 전 startup failure 1건을 별도로 보존했다. `smoke_execution.log`의 frozen config equality 오류는 model load/요청 실행 전에 발생하여 실행 요청은 0개다. Root 실행 검토상 GPU gate relative/absolute 경로 문자열 차이였으며 source/config를 바꾸지 않고 정확한 고정 경로로 `smoke_execution_02.log` 실행을 이어갔다. 이는 이미지별 요청 retry가 아니다. 최종 smoke 성공 여부는 phase validation receipt로 확인한다.

Cap flag 미기록은 false로 간주하지 않는다. 길이 16 관측은 EOS가 마지막인 경우를 포함할 수 있어 명시적 cap 판정과 분리한다.

Attempt directories: 84; PASS receipt가 없는 failed/partial 보존 attempt: 0; 최종 raw에 채택된 attempt: 84. `attempt_adoption.csv`와 `supplement_validation.json`은 cohort별 첫 PASS receipt의 exact raw 행과 최종 adopted 행을 비교한다. 좋은 점수/시간을 기준으로 재선택하지 않는다.

| 보존 failure/log | 상태 | 범위 |
|---|---|---|
| [smoke_execution.log](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/smoke_execution.log) | PRE_REQUEST_STARTUP_FAILURE | config equality guard before model load and any request; excluded from request retry counts |
| [gpu_validation.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_diagnostic_01/gpu_validation.json) | PARTIAL | pre-pilot correctness diagnostic, excluded from measured request counts |
| [final_audit_execution.log](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/final_audit_execution.log) | PRESERVED_LOG | execution log; no failed-request count inferred from a log filename |
| [pilot_execution.log](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/pilot_execution.log) | PRESERVED_LOG | execution log; no failed-request count inferred from a log filename |
| [setup_split_execution.log](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/setup_split_execution.log) | PRESERVED_LOG | execution log; no failed-request count inferred from a log filename |
| [smoke_execution.log](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/smoke_execution.log) | PRESERVED_LOG | execution log; no failed-request count inferred from a log filename |
| [smoke_execution_02.log](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/smoke_execution_02.log) | PRESERVED_LOG | execution log; no failed-request count inferred from a log filename |

## Controlled projection/position 및 legacy QA 진단

원본: [gpu_validation.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_full_01/gpu_validation.json). 세부값은 새 `controlled_validation_summary.json`에 원 receipt에서 복사했다. 동일 선택 주입, one-token suffix, 예외 해제 후 FullLoad 복원, legacy QA projection 차이는 timing-excluded correctness/diagnostic이며 5-arm latency와 섞지 않는다.

## 재현 및 보호 근거

| Artifact | 상태 | 시간 |
|---|---|---|
| [gpu_independent_audit.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_independent_audit.json) | PASS | NOT RECORDED |
| [protection_rehash.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/protection_rehash.json) | PASS | 2026-09-30T04:45:53.078242+00:00 |
| [protection_final.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/protection_final.json) | PASS | 2026-09-30T05:36:44.102784+00:00 |
| [validation.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/validation.json) | PASS | NOT RECORDED |
| [reproduction_commands.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/reproduction_commands.json) | RECORDED | NOT RECORDED |
| [source_freeze.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/source_freeze.json) | RECORDED | NOT RECORDED |
| [source.diff](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/source.diff) | RECORDED | NOT RECORDED |
| [contract_freeze_v2.json](/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/contract_freeze_v2.json) | RECORDED | NOT RECORDED |

기존 protection_rehash가 pilot 전에 완료된 경우 그것만으로 post-pilot 보호 완료를 주장하지 않는다. 최종 보호 receipt의 범위(stat/hash/새 파일 allowlist)를 함께 확인한다. Qwen GPU 및 전체 MT-GQA/MT-VQA는 이 pilot에 포함되지 않는다.

고정 실행/재개 명령(원 `reproduction_commands.json`):

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# cpu_new
/home/dblab/anaconda3/envs/mllm_ft/bin/python -m unittest discover -s tests -p 'test_sparsevlm_ssd*.py' -v
# gpu_new_receipt
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/97_validate_sparsevlm_ssd_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_revalidation_NEW_TIMESTAMP --limit 10 --regression-receipt /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/legacy_cpu_tests.json
# resume_exact
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/98_eval_sparsevlm_ssd_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z --results-dir /home/dblab/hj/mllm_v2/results/sparsevlm_ssd_kv25_20260930T041926Z --gpu-gate runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_full_01/gpu_validation.json --phase all --resume
# audit_new_output_only
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/99_audit_sparsevlm_ssd_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z --output-dir /home/dblab/hj/mllm_v2/results/sparsevlm_ssd_kv25_20260930T041926Z
```

새 GPU receipt/output 경로를 사용하고 기존 artifact overwrite 거부를 유지한다.

이 보충 문서는 최종 REPORT의 dataset/quality/TTFT 판정을 변경하지 않는다. Supplemental validation이 FAIL이면 해당 불일치를 먼저 해결·보고해야 하며 READY를 자동 승인하지 않는다.
