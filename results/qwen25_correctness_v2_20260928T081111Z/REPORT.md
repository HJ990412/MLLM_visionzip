# Qwen2.5-VL-7B-Instruct correctness v2 및 pilot 재검증

## 범위와 판정

이 결과는 [원래 v2 계약](../../docs/qwen25_correctness_contract_v2.md)을 첫 GPU 실행 전에 고정한 뒤, 첫 실행에서 새 Q projection 분기를 발견하여 중단하고, [두 번째 실행의 사전 부록](../../runs/qwen25_correctness_v2_20260928T081111Z/contract_addendum.md)을 다시 고정한 결과다. 첫 [v2 validation](../../runs/qwen25_correctness_v2_20260928T073413Z/gpu_validation/validation.json)은 G1–G15 PASS였으나 전체 판정 UNRESOLVED, pilot 불가로 원본 그대로 보존했다. 첫 실행 후의 [Q-shape 대조](../../runs/qwen25_correctness_v2_20260928T073413Z/diagnostic_qproj_shape_201535625/shape_control.json)는 질문 201535625의 505행 prefix + 20행 suffix에서 동일 RMSNorm 입력을 525행 shape로 투영하면 Q/K/V 모두 bitwise로 복원됨을 확인했다. 이것은 그 사례의 NF4/BF16 행렬 shape 민감성에 대한 근거이며 다른 분기에 대한 면제가 아니다.

두 번째 [GPU validation](../../runs/qwen25_correctness_v2_20260928T081111Z/gpu_validation/validation.json)은 고정 10쌍에서 G1–G15 모두 PASS, numerical branch attribution PASS, pilot_eligible=true다. [Smoke](../../runs/qwen25_correctness_v2_20260928T081111Z/smoke/summary.json), 새 [GQA raw](../../runs/qwen25_correctness_v2_20260928T081111Z/gqa_pilot/raw.jsonl)와 [MT raw](../../runs/qwen25_correctness_v2_20260928T081111Z/mt_pilot/raw.jsonl)도 완료했다. 이전 v1의 13 PASS / 2 FAIL과 debugging 재실행 13 PASS / 2 FAIL은 그대로 유지한다. 두 번째 v2 PASS는 v1 PASS나 ReComp와의 수치 동등성을 뜻하지 않는다.

입력 40개 파일의 사전 [freeze](../../runs/qwen25_correctness_v2_20260928T081111Z/frozen_inputs.json) SHA256은 642570323383bc6ba78382b87b2ff26e560b85b2ed6f0a1403a0a7d89ec34617이다. 계약 문서 SHA256 498589b23b43948df28bcc6bcd1e9638f03e57f1b4531029401c6ece86e5f8fb, 두 번째 validator SHA256 3c9672cc89fc449eceebae1fee70e3b64fc509fcd30061b492ad71be6a710e84, validation JSON SHA256 a3b851f6fc291711ef53452b6eab164ae84e6d3801ce717b19b8386a120ad860이다. [보호 감사](../../runs/qwen25_correctness_v2_20260928T081111Z/protected_after.json)는 이전 legacy/Qwen/debug/첫 v2의 20,129개 파일 변경·추가 0건이었다. 원래 [포팅 보고서](../qwen25_port_20260928T054537Z/PORT_REPORT.md), [debug 보고서](../qwen25_correctness_debug_20260928T070425Z/REPORT.md), 원래 v1 [validation](../../runs/qwen25_port_validate_gpu_20260928_0625_sdpa_frozen/validation.json)과 [변경 없는 debug validation](../../runs/qwen25_correctness_debug_20260928T070425Z/full_validation_sdpa_unchanged/validation.json)은 해시가 포함된 [evidence manifest](../../runs/qwen25_correctness_v2_20260928T081111Z/evidence_manifest.json)로 연결했다.

모델/processor/checkpoint revision cc594898137f460bfe9f0759e9844b3ce807cfb5, NF4, BF16 compute 및 native BF16 KV, SDPA, min_pixels=200704/max_pixels=802816, batch 1·단일 이미지, seed 1234·greedy·최대 16토큰, chunk 64·nominal 25% 및 기존 rounding/clamping은 유지했다. SSD에는 28개 layer의 native 4 KV heads와 전체 원본 visual KV를 보존한다. Ours는 image-only VisionZip 순서로 SSD를 first-k 순차 읽은 뒤 GPU에서 logical order로 복원하는 P2 경로다. 온라인 query scoring 호출은 0이며 P1 physical order는 진단 근거로만 남았다.

## Correctness

첫 세 질문 ID 201751701, 201751740, 201751873은 이미 관찰한 재현 집합이다. 추가 고정 일곱 ID 20929611, 201861403, 202108008, 201535625, 202101069, 2093976, 202144724는 deterministic 검증 집합이며 unseen holdout으로 부르지 않는다. [고정 validation manifest](../../runs/qwen25_correctness_v2_20260928T081111Z/validation_manifest.json)의 content SHA256은 6458545772438ef1b41cd35bbe7ec17c319be6f779d1b939fc9a94e0ada928a8이다.

| 항목 | 두 번째 실행 판정 | 검사 근거 |
|---|---|---|
| G1 BF16 round-trip | PASS 10/10 | 모든 28 layer K/V, structural/visual raw bits·dtype·shape·valid rows exact |
| G2 FullLoad SSD 대 matched memory | PASS 10/10 | 직접 captured prefix와 SSD prefix의 KV·first logits·generated IDs bitwise exact, 반복 실행 exact |
| G3 RepackedFull100 | PASS 10/10 | 전량 read와 inverse mapping으로 canonical KV exact 복원 |
| G4 Ours25 대 독립 compact memory | PASS 10/10 | SSD loader/compact builder를 쓰지 않은 독립 selected-ID gather reference와 KV·logits·sequence exact |
| G5–G7 | PASS 10/10 | T1 vision=1, hit vision=0, online score=0, first-k ID·정렬·inverse exact |
| G8 MRoPE/structural | PASS 10/10 | stock full logical prompt의 3축 좌표, token identity, structural prefix 및 corrupted fixture 검출 |
| G9 mask/FP32 oracle | PASS 10/10 | 독립 boolean causal visibility와 FP32 GQA oracle elementwise allclose(1e-5,1e-5), 잘못된 mask fixture 검출 |
| G10 state isolation | PASS 10/10 | 잘못된 image identity 거절, poisoned rope_deltas 제거, image/method 순서 변경 후 재현 |
| G11–G12 I/O | PASS 10/10 | hit에서는 미선택 visual payload read 0; 실제 반환 bytes/ranges 및 각 57 preads·57 spans 계획 일치 |
| G13 frozen policy | PASS 10/10 | 64-token chunk, nominal 25%, 해상도·backend·revision·seed 정책 보존 |
| G14 LLaVA regression | PASS, CPU 범위 | 기존 CPU suite 351 tests PASS; 실제 LLaVA GPU regression은 NOT RUN |
| G15 보호 hash | PASS | 이전 20,129개 파일의 byte hash·파일 집합 무변경 |

G9의 가장 큰 샘플별 FP32 절대 차이는 1.2547e-5다. 판정은 고정된 atol=1e-5 **및** rtol=1e-5의 원소별 allclose이므로 PASS이며, 절대 차이가 무조건 1e-5 미만이라는 주장은 하지 않는다. 모든 selected token의 original/stored/compact ID·원래 sequence position·MRoPE t/h/w 연결과 suffix visibility는 [validation 원본](../../runs/qwen25_correctness_v2_20260928T081111Z/gpu_validation/validation.json)에 있다. SSD prefix 재사용 경로와 reference의 동일 계산 결과를 검증한 범위 밖으로 일반화하지 않는다.

## Numerical diagnostics

v1의 원래 elementwise atol=0.125, rtol=0.02를 변경하지 않았다. 아래 평균은 각 쌍 logits 전체 152,064원소의 평균 절대 차이를 다시 10쌍 평균한 값이며, p99 열은 **샘플별 p99의 최대값**이다. 각 쌍의 정확한 max/mean/p99·위반 수·first/gen/pred agreement는 [numerical_diagnostics.csv](numerical_diagnostics.csv)와 validation JSON에 보존했다.

| 진단 | 최대 abs | 평균 abs | 최대 샘플 p99 abs | v1 위반 원소 / 총원소 | v1 PASS 쌍 | 첫 token / 전체 생성열 / 예측 일치 |
|---|---:|---:|---:|---:|---:|---:|
| D1 ReComp full-prefill 대 FullLoad split-prefill | 0.437500 | 0.046141 | 0.265625 | 35,318 / 1,520,640 | 5/10 | 9/10 · 9/10 · 9/10 |
| D2 Dense-selected 대 logical compact | 0.312500 | 0.040437 | 0.203125 | 14,911 / 1,520,640 | 3/10 | 10/10 · 10/10 · 10/10 |

D1의 질문 202101069은 first token·전체 sequence·예측이 달랐다. 다른 9쌍은 같았지만 수치 동등성은 아니다. 첫 v2와 두 번째 v2의 10쌍 로그잇 통계는 같으며, 두 번째 PASS는 오차 감소가 아니라 사전에 고정한 한 Q-shape 분기 근거를 반영한 결과다. D2는 BF16 attention output에서 처음 갈라졌고 독립 FP32 semantic oracle은 통과했다. D3 P1 physical-order 대 P2 logical-order는 새 재실행이 아닌 해시 고정된 [기존 trace](../../runs/qwen25_correctness_debug_20260928T070425Z/sdpa_q2/trace_201751740.json)를 참조했다: P0/P1 최대 abs 0.25, P0/P2 0.15625, P1/P2 0.25, 세 생성열은 동일했다.

## Smoke

고정 [smoke manifest](../../runs/qwen25_correctness_v2_20260928T081111Z/smoke_manifest.json)로 20 images × 3 questions × 3 methods = 180 requests를 실행했다. T1은 모든 방법에서 full-image였고, 각 방법의 40개 후속 질문은 독립 cache hit였다. first logits NaN/Inf·요청 실패·중복 ID·truncation·state isolation 오류가 각각 0이다. 기존 scorer의 전체 정확도는 ReComp/FullLoad 48.33%, Ours 45.00%, hit 정확도는 55.00%/55.00%/50.00%였다. 낮아진 Ours 품질을 PASS 조건에서 제외하거나 숨기지 않았다. Ours와 FullLoad의 hit 예측 일치는 35/40이었다.

## Pilot: 품질·지연·SSD

GQA [원래 frozen workload](../../runs/qwen25_port_20260928T054537Z/gqa_manifest/manifest.json)은 40 images × 6 independent questions × 3 methods = 720 requests, 방법당 200 hits다. MT [원래 frozen workload](../../runs/qwen25_port_20260928T054537Z/mt_manifest/manifest.json)은 40 dialogues × 3 turns × 3 methods = 360 requests, 방법당 80 hits다. MT history는 각 방법이 실제 생성한 이전 답변만 사용했고 이미지는 T1에 한 번만 있다. 모든 arm의 T1은 unpruned full-image, vision forward=1이며 capture는 그 forward에 piggyback했다. T2/T3 cache hits의 vision forward=0, Ours online score=0이다. 두 실행 모두 80 stores를 만들었고 frozen workload 일치, method/turn/history coverage, GPU inventory 및 raw audit가 PASS다. MT의 모든 40개 이미지에서 Ours T2/T3의 selected chunks와 57개 read spans가 동일했고, method별 생성 history와의 불일치는 0건이다.

실행별 독립 산출물은 GQA [config](gqa/config.json) · [new manifest](../../runs/qwen25_correctness_v2_20260928T081111Z/gqa_pilot/manifest.json) · [validation evidence](gqa/validation_evidence.json) · [summary](gqa/summary.json) · [analysis](gqa/ANALYSIS.md), MT [config](mt/config.json) · [new manifest](../../runs/qwen25_correctness_v2_20260928T081111Z/mt_pilot/manifest.json) · [validation evidence](mt/validation_evidence.json) · [summary](mt/summary.json) · [analysis](mt/ANALYSIS.md)에 있다. Config 파일은 실행 전에 고정된 manifest와 validation 설정을 실행 후 읽기 전용으로 정리한 provenance 사본이다.

아래 SSD MiB는 1 MiB=1,048,576 B인 **hit당 실제 반환 bytes**다. 비율은 이미지별 평균 / 전체 원본 token 합계 기준을 순서대로 적었다. GQA 마지막 열은 세션이 아닌 **이미지당 6개 독립 요청 + 한 번의 persistence/activation**이며, MT는 실제 3-turn session E2E다.

| Pilot | Method | All quality | Hit quality | Hit TTFT mean/p50/p95 ms | Hit request E2E ms | Hit visual / total SSD MiB | 실제 보존율 평균/합계 | Persistence ms | Session E2E 또는 GQA 이미지 총 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA | ReComp | 56.67% | 59.00% | 110.889 / 110.519 / 132.287 | 135.762 | 0 / 0 | — | — | 818.224 |
| GQA | FullLoad | 56.67% | 59.00% | 72.712 / 70.086 / 87.238 | 96.878 | 21.438 / 22.586 | 100% / 100% | 71.400 | 700.183 |
| GQA | Ours25 | 57.50% | 60.00% | 59.943 / 56.893 / 81.335 | 85.269 | 6.125 / 7.273 | 31.730% / 32.092% | 71.838 | 651.642 |
| MT | ReComp | 68.33% | 70.00% | 116.643 / 114.985 / 136.559 | 141.850 | 0 / 0 | — | — | 423.185 |
| MT | FullLoad | 68.33% | 70.00% | 81.114 / 76.590 / 120.670 | 107.300 | 22.750 / 23.898 | 100% / 100% | 84.831 | 448.857 |
| MT | Ours25 | 65.83% | 66.25% | 65.837 / 60.662 / 94.087 | 91.339 | 6.737 / 7.886 | 33.399% / 33.255% | 77.858 | 417.183 |

전체 요청 기준 request E2E 평균은 GQA ReComp/FullLoad/Ours 136.371/102.505/94.256 ms, MT 141.062/116.513/108.110 ms다. 각 scope의 TTFT·E2E p50/p95와 생성 길이 평균은 pilot별 [GQA summary](gqa/summary.json), [MT summary](mt/summary.json)에 보존했다.

Ours가 실제로 보존한 token은 GQA 4,480/13,960, MT 4,928/14,819이다. Nominal 25%는 chunk 예산이며 정확한 token 보존율이 아니다. GQA 이미지 28/40개, MT 35/40개에서 실제 보존율이 30%를 넘었다. FullLoad의 읽기에는 마지막 chunk padding이 포함된다: GQA valid visual 19.086 MiB 대 실제 visual read 21.438 MiB, MT 20.260 대 22.750 MiB. Ours의 선택된 visual valid/read는 GQA 6.125/6.125 MiB, MT 6.737/6.737 MiB로 해당 pilot에서 선택 chunk의 padding read가 0이었다. Structural read는 두 저장 방법 모두 hit당 1.148 MiB, metadata read는 0이다. Hit에서 실제 57 preads·57 spans였다. 각 이미지별 원본 token 수, 전체/선택 chunk 수, kept tokens, read bytes, preads/spans, metadata bytes·residency는 [GQA per-image CSV](gqa/per_image_geometry.csv)와 [MT per-image CSV](mt/per_image_geometry.csv)에 모두 있다.

FullLoad와 Ours의 **저장 footprint**는 본질적으로 같다. 전체 visual KV를 보존하므로 GQA 평균 22.6224/22.6225 MiB, MT 23.9364/23.9365 MiB(Full/Ours)이고 Ours/Full ratio는 약 1.000006이다. Hit당 total read ratio는 GQA 0.3220, MT 0.3300이다. 첫 hit 전 무결성 activation은 저장소 **전체**를 검증하므로 unselected visual도 한 번 읽는다: 58 preads, GQA 평균 22.623 MiB와 약 14.27 ms(Ours), MT 23.937 MiB와 약 14.99 ms. 이는 hit TTFT 밖이지만 위 이미지/세션 총 시간에는 한 번 포함된다. Activation과 모든 hits를 합친 전체 SSD 반환 bytes 비율은 GQA 이미지당 0.4352, MT 3-turn session당 0.5536이다. G11의 selected-only 주장은 hit 요청 범위에 한정된다. Metadata는 Ours activation 후 GQA 평균 95,572 B, MT 99,430 B resident이며 hit 중 visual payload를 CPU/GPU에 숨겨 보유하지 않는다.

T1 후처리 비용을 따로 기록했다. Ours의 score 계산은 GQA 1.723 ms, MT 1.760 ms, permutation 0.938/0.949 ms, repack 0.873/0.850 ms, write 9.692/10.135 ms, fsync 32.780/38.258 ms, capture clone 0.415/0.378 ms였다(모두 이미지당 평균). Persistence 합계 71.838/77.858 ms, activation 14.269/14.995 ms를 위 총 시간에 한 번 포함했다. FullLoad의 persistence 71.400/84.831 ms와 activation 13.752/14.487 ms도 포함했다. Score/capture clone은 T1 request E2E 측정 후 실행되어 그 시간과 중복 합산되지 않는다. 실제 background overlap은 가정하지 않았다.

TTFT는 request 시작부터 prompt/input 준비, 필요한 SSD read/H2D/cache assembly, prefill, 첫 토큰 materialization 및 CUDA sync까지다. Request E2E는 decoding 종료까지다. ReComp의 processor/vision/full visual prefill은 timer 안에 있다. 이미지 파일 read와 RGB/JPEG decode, 모델 로딩/warmup, 결과 logging, OS page-cache DONTNEED conditioning은 timer 밖이다. Conditioning은 GQA 400/400 hit에서 57/57 hint 성공, MT 160/160에서도 57/57 성공 및 실패 0건이었다. 이는 OS page-cache hint일 뿐 SSD controller/NAND cold 보장이 아니다. GPU inventory상 다른 compute process 0, 실행 순서는 이미지별 deterministic rotation이었다.

생성 token 수는 GQA ReComp/FullLoad/Ours 541/541/542개(요청당 2–9), MT 276/276/272개(2–4)였다. 두 pilot 모두 NaN/Inf first logits, truncation, request failure, duplicate ID가 각각 0이다. 품질 평가는 기존 repository exact scorer이며 공식 외부 evaluator가 아니다. Turn별 정확도는 다음과 같다.

| Pilot/turn | ReComp | FullLoad | Ours25 |
|---|---:|---:|---:|
| GQA Q1 | 45.0% | 45.0% | 45.0% |
| GQA Q2 | 52.5% | 52.5% | 50.0% |
| GQA Q3 | 65.0% | 65.0% | 65.0% |
| GQA Q4 | 65.0% | 65.0% | 67.5% |
| GQA Q5 | 42.5% | 42.5% | 45.0% |
| GQA Q6 | 70.0% | 70.0% | 72.5% |
| MT T1 | 65.0% | 65.0% | 65.0% |
| MT T2 | 65.0% | 65.0% | 62.5% |
| MT T3 | 75.0% | 75.0% | 70.0% |

Paired CI는 [GQA supplement](gqa/PILOT_SUPPLEMENT.md)와 [MT supplement](mt/PILOT_SUPPLEMENT.md)의 image-cluster bootstrap, 4,000 resamples, seed 1234를 일관되게 사용한다. 아래 차이는 왼쪽 방법 − 오른쪽 방법이다. 두 supplement와 primary analysis의 일부 CI 끝점은 독립 난수 draw 순서가 달라 약간 다르지만 점 추정치는 같다. CI에 0이 들어도 equivalence가 아니다.

| Pilot/scope | Paired contrast | Quality 차이, 95% CI (percentage points) | TTFT 차이 ms, 95% CI |
|---|---|---:|---:|
| GQA hit | FullLoad − ReComp | 0.0 [0.0, 0.0] | -38.176 [-41.218, -34.834] |
| GQA hit | Ours − ReComp | +1.0 [-3.5, +5.5] | -50.946 [-54.408, -47.249] |
| GQA hit | Ours − FullLoad | +1.0 [-3.5, +5.5] | -12.770 [-15.029, -10.044] |
| MT hit | FullLoad − ReComp | 0.0 [0.0, 0.0] | -35.529 [-40.269, -30.130] |
| MT hit | Ours − ReComp | -3.75 [-11.25, +3.75] | -50.805 [-55.102, -45.842] |
| MT hit | Ours − FullLoad | -3.75 [-11.25, +3.75] | -15.277 [-19.753, -11.150] |

All-turn 품질 차이는 GQA Full−ReComp 0, Ours−ReComp/Full +0.833 percentage points(95% CI -2.917,+4.583), MT Full−ReComp 0, Ours−ReComp/Full -2.5 points(CI -7.5,+2.5)다. FullLoad와 ReComp는 이 pilot에서 생성열이 모두 같았으나 GPU validation D1 logits는 v1 기준 FAIL 사례가 있고, GQA 200 hit의 첫 logits SHA는 모두 달랐다. 같은 생성열은 수치 동등성의 증거가 아니다.

## 부록 A: validation 쌍별 numerical diagnostics

표의 p99은 각 질문의 logits 원소 기준이다. 판정은 기존 v1 elementwise 기준이며, first token/sequence/prediction 일치 여부는 [CSV](numerical_diagnostics.csv)에 있다.

| QID | D1 max/mean/p99 abs | D1 위반 | D1 v1 | D2 max/mean/p99 abs | D2 위반 | D2 v1 |
|---|---:|---:|---|---:|---:|---|
| 201751701 | 0.132812 / 0.019921 / 0.062500 | 0 | PASS | 0.125000 / 0.018063 / 0.062500 | 0 | PASS |
| 201751740 | 0.437500 / 0.031024 / 0.093750 | 51 | FAIL | 0.156250 / 0.031929 / 0.093750 | 5 | FAIL |
| 201751873 | 0.437500 / 0.139939 / 0.265625 | 34,348 | FAIL | 0.312500 / 0.094779 / 0.203125 | 7,406 | FAIL |
| 20929611 | 0.125000 / 0.020019 / 0.068359 | 0 | PASS | 0.250000 / 0.038515 / 0.109375 | 10 | FAIL |
| 201861403 | 0.250000 / 0.072993 / 0.187500 | 206 | FAIL | 0.312500 / 0.088011 / 0.195312 | 7,360 | FAIL |
| 202108008 | 0.125000 / 0.021079 / 0.066406 | 0 | PASS | 0.187500 / 0.019663 / 0.062500 | 0 | PASS |
| 201535625 | 0.125000 / 0.023617 / 0.078125 | 0 | PASS | 0.156250 / 0.024670 / 0.085938 | 2 | FAIL |
| 202101069 | 0.250000 / 0.081482 / 0.156250 | 711 | FAIL | 0.187500 / 0.026836 / 0.093750 | 0 | PASS |
| 2093976 | 0.156250 / 0.022637 / 0.078125 | 0 | PASS | 0.250000 / 0.034781 / 0.123047 | 125 | FAIL |
| 202144724 | 0.187500 / 0.028700 / 0.093750 | 2 | FAIL | 0.171875 / 0.027118 / 0.093750 | 3 | FAIL |

## 부록 B: 이미지별 visual geometry와 실제 hit read

FullLoad는 각 이미지의 모든 chunk와 원본 visual token을 사용한다. 다음 표의 선택 chunk·kept token은 Ours25이며 실제 read MiB는 padding 포함 visual/structural/metadata 반환 bytes의 합계다. Hit마다 57 preads·57 spans이고 image별 상세 bytes·activation은 [GQA CSV](gqa/per_image_geometry.csv), [MT CSV](mt/per_image_geometry.csv)에 있다.

### GQA 40 images

| Image | Visual tokens | Full chunks | Ours chunks | Kept tokens | Retention | Ours visual / total read MiB | Full visual / total read MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| n355567 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n9181 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n390187 | 368 | 6 | 2 | 128 | 34.78% | 7.000 / 8.148 | 21.000 / 22.148 |
| n133585 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n272098 | 484 | 8 | 2 | 128 | 26.45% | 7.000 / 8.148 | 28.000 / 29.148 |
| n472825 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n450919 | 266 | 5 | 1 | 64 | 24.06% | 3.500 / 4.648 | 17.500 / 18.648 |
| n37274 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n293477 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n44249 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n331357 | 322 | 6 | 2 | 128 | 39.75% | 7.000 / 8.148 | 21.000 / 22.148 |
| n51002 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n244826 | 299 | 5 | 1 | 64 | 21.40% | 3.500 / 4.648 | 17.500 / 18.648 |
| n470131 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n195249 | 299 | 5 | 1 | 64 | 21.40% | 3.500 / 4.648 | 17.500 / 18.648 |
| n281241 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n90294 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n16425 | 270 | 5 | 1 | 64 | 23.70% | 3.500 / 4.648 | 17.500 / 18.648 |
| n527589 | 266 | 5 | 1 | 64 | 24.06% | 3.500 / 4.648 | 17.500 / 18.648 |
| n262929 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n48494 | 280 | 5 | 1 | 64 | 22.86% | 3.500 / 4.648 | 17.500 / 18.648 |
| n58220 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n522733 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n199286 | 368 | 6 | 2 | 128 | 34.78% | 7.000 / 8.148 | 21.000 / 22.148 |
| n298104 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n166008 | 266 | 5 | 1 | 64 | 24.06% | 3.500 / 4.648 | 17.500 / 18.648 |
| n67005 | 266 | 5 | 1 | 64 | 24.06% | 3.500 / 4.648 | 17.500 / 18.648 |
| n154856 | 280 | 5 | 1 | 64 | 22.86% | 3.500 / 4.648 | 17.500 / 18.648 |
| n567860 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n314171 | 368 | 6 | 2 | 128 | 34.78% | 7.000 / 8.148 | 21.000 / 22.148 |
| n437064 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n498140 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n16656 | 414 | 7 | 2 | 128 | 30.92% | 7.000 / 8.148 | 24.500 / 25.648 |
| n329479 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n406334 | 368 | 6 | 2 | 128 | 34.78% | 7.000 / 8.148 | 21.000 / 22.148 |
| n302387 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n52544 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n540852 | 460 | 8 | 2 | 128 | 27.83% | 7.000 / 8.148 | 28.000 / 29.148 |
| n234722 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n494918 | 266 | 5 | 1 | 64 | 24.06% | 3.500 / 4.648 | 17.500 / 18.648 |

### MT 40 images

| Image | Visual tokens | Full chunks | Ours chunks | Kept tokens | Retention | Ours visual / total read MiB | Full visual / total read MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| n130464 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n9856 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n470131 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n228268 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n313060 | 529 | 9 | 2 | 128 | 24.20% | 7.000 / 8.148 | 31.500 / 32.648 |
| n433692 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n483840 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n115614 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n59627 | 414 | 7 | 2 | 128 | 30.92% | 7.000 / 8.148 | 24.500 / 25.648 |
| n162108 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n469525 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n460556 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n477702 | 368 | 6 | 2 | 128 | 34.78% | 7.000 / 8.148 | 21.000 / 22.148 |
| n302387 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n493357 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n329514 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n275148 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n257997 | 270 | 5 | 1 | 64 | 23.70% | 3.500 / 4.648 | 17.500 / 18.648 |
| n234683 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n410476 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n305495 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n437192 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n481655 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n450919 | 266 | 5 | 1 | 64 | 24.06% | 3.500 / 4.648 | 17.500 / 18.648 |
| n557666 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n256120 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n187961 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n531731 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n211324 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n429961 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n88366 | 299 | 5 | 1 | 64 | 21.40% | 3.500 / 4.648 | 17.500 / 18.648 |
| n522733 | 391 | 7 | 2 | 128 | 32.74% | 7.000 / 8.148 | 24.500 / 25.648 |
| n288870 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n541688 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n309148 | 529 | 9 | 2 | 128 | 24.20% | 7.000 / 8.148 | 31.500 / 32.648 |
| n449058 | 414 | 7 | 2 | 128 | 30.92% | 7.000 / 8.148 | 24.500 / 25.648 |
| n489190 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n95369 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n195925 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |
| n275857 | 345 | 6 | 2 | 128 | 37.10% | 7.000 / 8.148 | 21.000 / 22.148 |

## 최종 상태

| 항목 | 판정 |
|---|---|
| LEGACY STRICT LOGITS CONTRACT v1 | **FAIL**: 원래 13 PASS/2 FAIL, 변경 없는 debugging 재실행도 13 PASS/2 FAIL. v2 D1/D2의 원래 기준 PASS는 5/10 및 3/10. |
| GPU SYSTEM CORRECTNESS v2 | **PASS**: 두 번째 prospective 실행 G1–G15 전부 PASS; 첫 v2 실행 UNRESOLVED는 별도 보존. |
| INDEPENDENT POSITION/MASK/ATTENTION CHECKS | **PASS**: MRoPE, structural, causal mask, negative fixtures 및 FP32 oracle 10/10. |
| GQA PILOT | **VALID UNDER v2**: 720 requests, frozen manifest·audit PASS. |
| MT PILOT | **VALID UNDER v2**: 360 requests, method-specific generated history·audit PASS. |
| READY FOR LIMITED CROSS-MODEL PILOT TABLE | **YES, 제한적**: 이 Qwen 구현과 고정 workload의 v2 system/semantic gate 및 새 pilot 결과를 표에 올릴 수 있다. LLaVA 실제 GPU regression은 이번 실행에서 하지 않았고, 다른 모델·데이터셋의 품질 동등성은 주장하지 않는다. |
