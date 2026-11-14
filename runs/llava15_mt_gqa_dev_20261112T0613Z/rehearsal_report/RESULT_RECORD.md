실행일 2026-11-12. Run `runs/llava15_mt_gqa_dev_20261112T0613Z`, 결과 `runs/llava15_mt_gqa_dev_20261112T0613Z/rehearsal_report` (REPORT.md, summary.json, decision.json). 상태 **REHEARSAL_NOT_A_RESULT**.

- 요청: main 60 / 60 (5 arm × 12,183), 실패·스킵 0, 독립 감사 PASS, main 0.01 h. Gate LG1–LG10 모두 PASS. T1 arm 간 토큰 일치 4 / 4.
- 모델: llava-hf/llava-1.5-7b-hf @ `b234b804b114`, 576 토큰 고정·separator 없음, KV 301.99 MB/이미지, HF 기본 전처리(중앙 자르기).

| Arm | Followup % | vs ReComp Δpp [95% CI] | TTFT mean ms | TTFT / ReComp | SSD MB |
|---|---|---|---|---|---|
| ReComp | 50.000 | – | 125.70 | 1.000× | 0.00 |
| U100_packed | 62.500 | +12.500 [+0.000, +37.500] | 208.29 | 1.657× | 301.99 |
| U50_packed | 62.500 | +12.500 [+0.000, +37.500] | 121.06 | 0.963× | 150.99 |
| U25_packed | 37.500 | -12.500 [-37.500, +0.000] | 84.77 | 0.674× | 75.50 |
| PYR-b20_packed | 37.500 | -12.500 [-37.500, +0.000] | 83.54 | 0.665× | 75.50 |

- (A) 주 운영점 U25 vs ReComp: Δ -12.500 [-37.500, +0.000] pp (차이 미입증), 원래 성능 대비 75.00%, TTFT 84.77 vs 125.70 ms = 0.674×, 75.50 MB/followup.
- (B) R100 = 1.657, S25 = 0.407, S50 = 0.581. 가설(R100 0.462–1.39, S25 0.34–0.892): R100 틀림, S25 맞음. 참고: Qwen R100 0.462·S25 0.892(같은 packed 경로), LLaVA-NeXT R100 1.39·S25 0.34(v1, 파일 분리 경로, 직접 비교 아님).
- (C) PYR-b20 − U25: **NO-GO**, Δ +0.000 [+0.000, +0.000] pp, SSD 75.50 / 75.50 MB, TTFT 0.986×.
- (F) T1 정확도 50.000% (모든 arm full pixel).
- 프롬프트와 다르게 한 점: LG3 참조를 같은 길이의 질문 제거 forward로 정의(문자 그대로의 짧은 forward는 BF16 shape 의존 반올림으로 bitwise 불가, 진단 기록), LG6 참조는 저장 순서 배치, H2D 단독은 fixture 진단. 자세한 내용은 REPORT.md.
- NOT_MEASURED: HOLD 경로 H2D 단독, NAND cold / O_DIRECT, GPU footprint, LLaVA-NeXT 100% packed, FullLoad arm, MT-VQA·Qwen·GLM·LLaVA-NeXT 재실행.
