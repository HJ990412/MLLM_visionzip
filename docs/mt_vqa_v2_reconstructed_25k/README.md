# MT-VQA-v2-reconstructed-25k

새 공식 VQAv2 validation 기반 25k three-turn workload. 저자 subset의 정확한 재현이 아니다.

- 계약: ../mt_vqa_v2_reconstructed_25k_contract.md
- 최종 완료 보고서: [MAIN_REPORT.md](../../results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/MAIN_REPORT.md)
- 모델별 main 결과: [LLaVA](../../results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/llava/main/REPORT.md), [Qwen](../../results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/qwen/main/REPORT.md), [GLM](../../results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/glm/main/REPORT.md)
- 원본/선택 data manifest: ../../data/mt_vqa_v2_reconstructed_25k/

원본 annotations, native generated IDs/raw, per-image atomic commits, full hashes 및 독립 공식 VQA 재채점을 사용한다. Smoke와 pilot을 main 결과로 해석하지 않는다.

최종 v3 main은 LLaVA 375,000, Qwen 225,000, GLM 225,000으로 총 825,000요청을 완료했다.
RUNNING.md는 초기 smoke 단계의 스냅샷이므로 최종 완료 여부를 판단하는 자료로 사용하지 않는다.
원본 이미지·전체 manifest·raw journal·요청별 대용량 CSV는 로컬에 보존하고, 레포에는 코드와
공식 scorer·요약표·완료/검증 receipt·해시/provenance를 게시한다.
