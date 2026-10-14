# GLM-4.6V-Flash mt_pilot 독립 결과

상태: **PASS**. 채택 raw 요청 360 / 계약 360.

정확도는 raw prediction과 원래 gold[0]의 strict normalized exact match를 독립 계산했다.
GPU 정합성 gate는 별도 검증 receipt를 요구한다. 이 보고서가 native correctness를 승인하지 않는다.

| 방법 | 요청 | 정확도 % | Followup TTFT mean/p50/p95 ms | 실제 KV read MB/request |
|---|---:|---:|---|---:|
| recompute | 120 | 63.333 | 129.681/128.987/150.127 | 0.000000 |
| fullload | 120 | 63.333 | 74.301/73.567/81.803 | 15.901696 |
| ours_prefixstore_kv25 | 120 | 60.000 | 61.286/60.913/65.530 | 4.579328 |

방법 비교는 각 방법이 생성한 자기 history의 차이를 포함한다. MB=10^6 bytes, GiB=2^30 bytes, ms, 정확도 차이는 percentage points다.

actual shared-image session은 이미지별 실제 build와 activation을 한 번만 귀속한다.
standalone-equivalent는 이미지 setup을 대화마다 다시 부과한 DERIVED 값이다. 비-source capture 추가 비용은 NOT_MEASURED다.

paired_analysis.csv는 seed1234, 이미지 단위 paired cluster bootstrap 10,000회다.
같은 이미지의 방법/대화/turn을 함께 resample하고 pooled sum/count로 요청 가중치를 보존한다.
CI가 0을 포함해도 동등성을 주장하지 않는다. OS read/write를 물리 NAND I/O로 부르지 않는다.

전체 source·membership·history·selection·I/O 감사와 count는 analysis.json, raw hash는 raw_evidence_index.json에 있다.
