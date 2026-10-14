# GLM-4.6V-Flash 연구 추가 결과

최종 v5 MT-GQA-reconstructed 3-arm main을 36,549/36,549 final requests로 완료했다.
GQA720/MT360 파일럿, CPU82, native GPU gate, 전체 raw 및 796 store 독립 감사가
통과했다. 이전 v3/v4의 실패·파일럿·partial main은 별도 artifact로 보존했다.

후속 ReComp/FullLoad/B의 strict EM은 61.635/61.549/58.089%,
TTFT는 126.512/73.468/60.588 ms, actual KV read는 0/15.180/4.397 MB다.
B의 후속 정확도 손실은 ReComp 대비 -3.546 pp (95% CI -4.416~-2.692),
FullLoad 대비 -3.460 pp (-4.328~-2.597)다. 효율 개선과 정확도 손실을 함께 보고한다.

[최종 REPORT](../results/glm46v_prefixstore3_v5_20261007T075730Z/REPORT.md),
[포팅](../results/glm46v_prefixstore3_v5_20261007T075730Z/PORT_REPORT.md),
[계약](glm46v_prefixstore3_contract.md),
[재현](../results/glm46v_prefixstore3_v5_20261007T075730Z/REPRODUCE.md)를 참조한다.
실험은 로컬에서 수행했으며 기존 LLaVA/Qwen main을 재실행하거나 remote push하지 않았다.
