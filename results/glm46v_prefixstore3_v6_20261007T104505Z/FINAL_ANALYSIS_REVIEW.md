# GLM-4.6V-Flash v6 최종 분석 독립 검토

**독립 raw·native protocol·strict EM·I/O·caller outer setup·54개 paired 통계 검산 PASS.** 398 images/4,061 dialogues/36,549 final requests, 방법당12,183 및8,122 followups를 완료했다. V6 실패 image attempt·재시도는0이고 첫 완전한 성공만 채택했다. GQA720/MT360은 별도 완료된 pilot이며 validation/이전 v3–v5 요청은 이 main 분모에 없다.

## 독립 재구성과 evidence

- GLM production/store/analysis helper를 import하지 않은 derived script가 adopted raw398파일과 setup398파일의 full SHA256·소유 attempt·count를 검산했다. Pinned tokenizer의 independent decode와 공식 chat template만 protocol 참조로 사용했다.
- Original question/gold[0]/membership/ordinal rotation이 맞았다. Generated IDs에서 scoring answer와 EOS 한 개만 제외한 native body를 별도로 재구성했고, 모든 요청의 자기 method/dialogue body·lineage·prompt·expanded input IDs가 맞았다. 원래 geometry에서 직접 만든3축 positions/offset도 모든 요청과 일치했다.
- T1 generated IDs는4,061/4,061 dialogues에서 세 방법 사이에 exact 같았다. 모든 T1은 full pixels이며 ReComp도 매 요청 full pixels/vision1회다. Ours T1의 raw k는 저장 예산을 나타내므로 T1 attended content를 k로 잘못 해석하지 않았다.
- Cached81개 실제 OS ranges(40layers×K/V+structure), returned bytes/calls·H2D tensor bytes·original-N budget·source T1 IDs·all same-image frozen B selection·compact cache/decode 증가를 검산했다. FullLoad와 B는 원래19개 구조 행을 모두 보존하고 content padding0이다.
- Seed1234,10,000회,398 image-cluster paired resampling에서 모든 방법/대화/turn을 같이 유지하고 pooled sum/count를 계산했다. Published54 point/CI의 numerical residual은 정확0이다. 모든 품질·latency mean/p50/p95 및3-turn session summary도 raw 재계산과 일치했다.

검산 attempt: `independent_review/attempt_20261007T130724092714Z`. Receipt SHA256 `6b602d22033bf3ad83cca4f7a009da7677e8faec054797cc2138aa98fbc9a83b`. 프로그램과 독립 quality/latency/session/paired CSV 및 raw inventory가 같은 폴더에 있다. 검산 프로그램은 첫 attempt에서 PASS했고 실패 audit attempt는0이다.

## 실제 품질·latency·읽기

| 방법 | 전체 EM % | Followup EM % | Followup TTFT mean/p50/p95 ms | Followup KV MB/request |
|---|---:|---:|---:|---:|
| recompute | 61.216 | 61.635 | 126.523/127.200/149.011 | 0.000000 |
| fullload | 61.159 | 61.549 | 73.477/73.457/76.761 | 15.179750 |
| ours_prefixstore_kv25 | 58.852 | 58.089 | 60.578/60.448/63.024 | 4.396972 |

Followup strict correct counts는 R5006/F4999/B4718, 각 분모8122다. 기존 소문자화·punctuation 제거·whitespace collapse·a/an/the 제거 규칙과 unchanged gold[0] exact match를 적용했다. Native control body를 scoring prediction과 혼동하지 않았다.

FullLoad와 B의 실제 OS returned KV bytes는 각각123,289,927,680 및35,712,204,800이다. 두 방법의 pread는 각각657,882 calls다. ReComp KV read0은 image-file I/O가0이라는 뜻이 아니며 image-file bytes는 NOT_MEASURED다.

| B−baseline | Followup EM Δpp [95% CI] | TTFT Δms [95% CI] | E2E Δms [95% CI] | KV ΔMB/request [95% CI] |
|---|---:|---:|---:|---:|
| recompute | -3.546 [-4.416, -2.692] | -65.945 [-67.250, -64.669] | -67.192 [-68.742, -65.638] | +4.397 [+4.319, +4.475] |
| fullload | -3.460 [-4.328, -2.597] | -12.899 [-13.052, -12.735] | -14.423 [-15.239, -13.577] | -10.783 [-11.018, -10.547] |

작은 native40KiB/token KV에서도 B의 요청 TTFT·I/O 개선과 정확도 손실이 함께 관찰됐다. Accuracy CI는 모두 음수다. 이 비교에는 method-local generated history 차이가 포함된다. Mean CI를 p95 CI로 바꾸어 읽지 않으며 CI가0을 포함해도 동등성으로 해석하지 않는다.

## 실제 outer setup와 session 회계

V6의796 build/796 activation events는 caller 진입 전부터 runner.persist/runner.activate의 완전한 반환까지 측정했다. Physical setup 합과 source dialogue attribution 합은 모두 **55670.386902522296 ms**다. Source clone/importance는 source request E2E에 이미1회 포함돼 setup에 다시 더하지 않았다.

| 측정 범위 | outer 총합 ms | inner 진단 총합 ms | outer−inner 포함 비용 ms |
|---|---:|---:|---:|
| Writer | 47936.413759 | 45846.786648 | 2089.627111 |
| Activation | 7733.973144 | 7692.572124 | 41.401019 |

Writer identity preparation/expected-identity 등록/return tail 및 activation의 caller 준비/IO summary/return tail이 새 outer timer에 실제 포함됐다. 이는 v5의 누락분을 추정한 보정이 아니라 v6의 새 물리 호출에서 측정한 값이다. Inner component들을 outer 비용에 다시 가산하지 않았다.

| 방법 | A 실제 shared-stream session mean ms | B derived standalone-equivalent mean ms |
|---|---:|---:|
| recompute | 641.398 | 641.398 |
| fullload | 543.477 | 614.079 |
| ours_prefixstore_kv25 | 513.202 | 569.204 |

| B−baseline | A actual session Δms [95% CI] | B derived Δms [95% CI] |
|---|---:|---:|
| recompute | -128.196 [-131.311, -125.063] | -72.194 [-75.030, -69.327] |
| fullload | -30.275 [-31.919, -28.600] | -44.875 [-46.658, -43.031] |

A는 각 image/method의 실제 setup을 source dialogue에 한 번만 귀속한다. B는 같은 측정 setup을 모든 dialogue에 재부과한 DERIVED 값이고 독립 cold-new-image session 실측이 아니다. Hypothetical non-source T1 capture 증분은 NOT_MEASURED다. Sessions는3개 actual request E2E의 합을 사용하며 TTFT를 E2E 대신 쓰지 않았다.

## 출력·보호·해석의 한계

- Empty/thinking open/thinking close는 세 방법 모두0. Cap/malformed는 R3/F3/B2이며 모두 final 분모와 실제 native history에 포함했다. 출력 때문에 재시도·필터를 하지 않았다.
- 첫 token은 흔히 native answer-begin control151361이다. TTFT는 native 첫 출력 token materialization이며 first lexical answer latency는 NOT_MEASURED다. Control boundary를 reasoning 또는 lexical answer로 혼동하지 않는다.
- Request/dialogue weighted estimates와 image-macro storage를 구분한다. 동일 image 안의 모든 대화·turn·방법을 함께 bootstrap했다. Bootstrap은 독립 반복 GPU 실행이나 데이터셋 밖 일반화의 보장이 아니다.
- Logical/allocated store는 file bytes/st_blocks scope다. 구조19행과 metadata 때문에 content25.1267% 저장이 total bytes75% 감소를 뜻하지 않는다. Directory/inode/kernel FD memory, NAND physical traffic, cold cache 보장, saturation/hit-rate/eviction/parallel throughput은 별도 검증하지 않았다.
- ReComp/FullLoad의 서로 다른 native projection/attention shape에 따른 수치·예측 차이는 restore bits 오류 및 B compression loss와 구분한다. V5와 core5개의 SHA가 같아도 각 run의 실제 timing을 합치거나 동일하다고 가정하지 않았다. V6의 새 raw/setups만 primary estimand다.

기존 필수 source/report/details **25파일**의 before/after full-byte SHA256은 모두 같았다. `independent_review/attempt_20261007T130724092714Z/baseline_protection_after.json` SHA256 `ce1b2956b354c1b78a3fc667c3cd8e77831814375c78c227fed0d874d3180b20`. 이 제한된 보호 범위는 전체 기존 대용량 artifact의 global full-byte 보호를 뜻하지 않는다.

Payload 전체64,476 hashes와 선택 native K/V byte offsets 비교는 별도 store agent의796-store audit에서 판단한다. 이 검토는 중복 payload scan 없이 adopted raw/setup·native protocol·통계·baseline 보호를 독립 확인했다. Source·raw·gate·실패 기록을 수정하지 않았다.
