# GLM-4.6V-Flash PrefixStore-KV25 [B] 완료 결과 v6

최종 계약 실행은 `glm46v_prefixstore3_v6_20261007T104505Z`다. RTX4090에서 CPU86/86,
GPU admission CPU75/75, native GPU6사례와 FP64 attention480비교를 통과한 뒤
GQA720/MT360 파일럿과 MT-GQA398이미지·4,061dialogues·36,549요청을 완료했다.
최종 실패 attempt/재시도0, dialogue T1 generated IDs4,061/4,061 일치다.

| 방법 | 전체 EM % | Followup EM % | Followup TTFT ms | KV read MB/followup | Logical store MB/image | Setup 포함 A session ms | Derived B session ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 61.216 | 61.635 | 126.523 | 0.000 | KV store 없음 | 641.398 | 641.398 |
| FullLoad | 61.159 | 61.549 | 73.477 | 15.180 | 15.170 | 543.477 | 614.079 |
| Ours-PrefixStore-KV25 [B] | 58.852 | 58.089 | 60.578 | 4.397 | 4.420 | 513.202 | 569.204 |

Request latency/EM/session은 요청·dialogue 가중 평균, store는 image-macro 평균이다.
MB=10^6 bytes, ms. 모든 dialogue의 T1은 full pixels다. ReComp KV read0은 이미지 I/O0을 뜻하지 않는다.
A는 이미지별 실제 caller outer build/activation을 한 번, B는 같은 setup을 dialogue마다 재부과한
DERIVED 값이다. 비-source T1 capture 증분과 독립 cold session은 NOT_MEASURED다.

Seed1234,398paired image clusters,10,000 bootstrap에서 followup EM의 B−ReComp는
−3.546pp[−4.416,−2.692], B−FullLoad는−3.460pp[−4.328,−2.597]였다.
TTFT 차이는−65.945ms[−67.250,−64.669]와−12.899ms[−13.052,−12.735]다.
작은 native KV(40KiB/token)에서도 비용 개선이 유지되지만 정확도 손실을 동반한다.
Native generated history 차이가 포함된 방법 수준 비교다.

V6는 identity 준비·wrapper return을 포함한 caller outer timer를 사용한다. 실제 setup
55,670.386903ms와 source-session 귀속합이 같다. 독립 audit는796stores/64,476payloadhashes/
2,803,840selected K/V rows를 확인했고, 별도 raw 검산의54paired CI residual은0이다.
Cap32는 GLM의 native boundary 때문에 main 전에 고정했고 기존 실험 cap16과 다르다.
Thinking/empty0, cap/malformedR3/F3/B2를 모두 분모에 포함했다.

최종 source는 [mmimpress/glm46v](../mmimpress/glm46v), driver는119–123,
CPU tests는6개`tests/test_glm46v_*.py`다. 기존 LLaVA/Qwen source·결과25파일을 hash로 보호했고
main을 재실행하지 않았다. Remote push나 기존 artifact 삭제는 없다.

실제 고정값·전체scope/p50/p95·paired CI·실패 이력은
[최종 REPORT](../results/glm46v_prefixstore3_v6_20261007T104505Z/REPORT.md),
[CONTRACT](../results/glm46v_prefixstore3_v6_20261007T104505Z/CONTRACT.md),
[PORT_REPORT](../results/glm46v_prefixstore3_v6_20261007T104505Z/PORT_REPORT.md),
[VALIDATION](../results/glm46v_prefixstore3_v6_20261007T104505Z/VALIDATION.md),
[독립 분석](../results/glm46v_prefixstore3_v6_20261007T104505Z/FINAL_ANALYSIS_REVIEW.md),
[재현 명령](../results/glm46v_prefixstore3_v6_20261007T104505Z/REPRODUCE.md)에 있다.
V3/V4 실패·V5 qualified setup 결과는 별도 보존하고 V6 분모에 합치지 않았다.
큰 raw/store/weights는 로컬에 보존돼 있으며 local_artifact_index.json과 fullhash receipt로 찾는다.
OS I/O는 NAND 물리 I/O가 아니며 cache saturation/hit-rate/eviction을 검증한 실행은 아니다.
