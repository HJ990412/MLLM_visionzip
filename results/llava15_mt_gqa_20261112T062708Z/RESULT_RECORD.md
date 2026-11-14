실행일 2026-11-12. Run `runs/llava15_mt_gqa_20261112T062708Z`, 결과 `results/llava15_mt_gqa_20261112T062708Z` (REPORT.md, summary.json, decision.json). 상태 **FINAL**.

- 요청: main 60,915 / 60,915 (5 arm × 12,183), 실패·스킵 0, 독립 감사 PASS, main 3.20 h. Gate LG1–LG10 모두 PASS. T1 arm 간 토큰 일치 4,061 / 4,061.
- 모델: llava-hf/llava-1.5-7b-hf @ `b234b804b114`, 576 토큰 고정·separator 없음, KV 301.99 MB/이미지, HF 기본 전처리(중앙 자르기).

| Arm | Followup % | vs ReComp Δpp [95% CI] | TTFT mean ms | TTFT / ReComp | SSD MB |
|---|---|---|---|---|---|
| ReComp | 65.279 | – | 126.86 | 1.000× | 0.00 |
| U100_packed | 65.304 | +0.025 [-0.235, +0.295] | 209.32 | 1.650× | 301.99 |
| U50_packed | 64.959 | -0.320 [-0.817, +0.171] | 126.77 | 0.999× | 150.99 |
| U25_packed | 63.322 | -1.958 [-2.693, -1.239] | 86.38 | 0.681× | 75.50 |
| PYR-b20_packed | 63.371 | -1.908 [-2.626, -1.204] | 86.36 | 0.681× | 75.50 |

- (A) 주 운영점 U25 vs ReComp: Δ -1.958 [-2.693, -1.239] pp (ReComp보다 낮음 (CI 0 제외)), 원래 성능 대비 97.00%, TTFT 86.38 vs 126.86 ms = 0.681×, 75.50 MB/followup.
- (B) R100 = 1.650, S25 = 0.413, S50 = 0.606. 가설(R100 0.462–1.39, S25 0.34–0.892): R100 틀림, S25 맞음. 참고: Qwen R100 0.462·S25 0.892(같은 packed 경로), LLaVA-NeXT R100 1.39·S25 0.34(v1, 파일 분리 경로, 직접 비교 아님).
- (C) PYR-b20 − U25: **NO-GO**, Δ +0.049 [-0.472, +0.570] pp, SSD 75.50 / 75.50 MB, TTFT 1.000×.
- (F) T1 정확도 59.394% (모든 arm full pixel).
- 프롬프트와 다르게 한 점: LG3 참조를 같은 길이의 질문 제거 forward로 정의(문자 그대로의 짧은 forward는 BF16 shape 의존 반올림으로 bitwise 불가, 진단 기록), LG6 참조는 저장 순서 배치, H2D 단독은 fixture 진단. 자세한 내용은 REPORT.md.
- NOT_MEASURED: HOLD 경로 H2D 단독, NAND cold / O_DIRECT, GPU footprint, LLaVA-NeXT 100% packed, FullLoad arm, MT-VQA·Qwen·GLM·LLaVA-NeXT 재실행.
