# 결과 기록 — Qwen2.5-VL 보존 비율 sweep (25 / 30 / 40 / 50 / 100%), MT-GQA HOLD

프롬프트 문서가 저장소 파일이 아니라 대화로 전달되어, 그 문서의 "결과 기록 (실행 후 기록)" 절에 들어갈 내용을 여기에 둔다. 전체 보고서는 [REPORT.md](REPORT.md), 수치 전체는 [summary.json](summary.json).

## 2026-11-11 실행 — 판정 없음 (곡선 보고), 상태 FINAL

- Run `runs/qwen_budget_sweep_20261111T152143Z`, 결과 `results/qwen_budget_sweep_20261111T152143Z/`.
- followup = T2–T3, 8,122 요청/arm. CI = 이미지 단위 paired bootstrap 95% (10,000회, seed 1234). MB = 10^6 B.

| Arm | layer당 content 행 (N=345) | Followup % (정답) | vs ReComp Δpp [95% CI] | TTFT mean ms | TTFT / ReComp | SSD MB / followup | 읽기 호출 |
|---|---|---|---|---|---|---|---|
| ReComp | – | 61.364 (4984) | – | 113.222 | 1.000× | 0 | 0 |
| FullLoad | 345 | 61.401 (4987) | +0.037 [−0.151, +0.221] | 70.770 | 0.625× | 24.407 | pread 57 |
| U25_packed | 87 | 57.868 (4700) | −3.497 [−4.260, −2.743] | 46.614 | 0.412× | 6.371 | preadv 1 |
| U30_packed | 104 | 58.963 (4789) | −2.401 [−3.098, −1.700] | 46.956 | 0.415× | 7.403 | preadv 1 |
| U40_packed | 138 | 60.133 (4884) | −1.231 [−1.855, −0.625] | 47.679 | 0.421× | 9.448 | preadv 1 |
| U50_packed | 173 | 60.859 (4943) | −0.505 [−1.050, +0.026] | 48.234 | 0.426× | 11.507 | preadv 1 |
| U100_packed | 345 | 61.401 (4987) | +0.037 [−0.151, +0.221] | 52.270 | 0.462× | 21.765 | preadv 1 |

- **사전 고정한 기준선 (a) 손실 미검출**: **U50_packed** — ReComp 대비 Δ의 CI가 0을 포함하는 가장 작은 비율. 경계선 결과다: CI 상한이 +0.026 pp이고, 같은 arm을 FullLoad와 비교하면 −0.542 pp [−1.079, −0.012]로 0을 포함하지 않는다. "차이 미입증"이지 "동등"이 아니다.
- **사전 고정한 기준선 (b) LLaVA 25% 수준**: **U40_packed** — ReComp 대비 손실 점추정이 2.30 pp 이하인 가장 작은 비율(손실 1.231 pp). U30은 2.401 pp로 기준을 넘는다.
- 두 기준선 모두 HOLD에서 관찰한 값이며 다른 데이터에서 확인되지 않았다.
- **단조성**: 비율을 올릴 때 followup 정확도가 떨어진 구간은 없다. 인접 구간 상승폭은 25→30 +1.096, 30→40 +1.170, 40→50 +0.726, 50→100 +0.542 pp이고 네 구간 모두 CI가 0을 제외한다.
- **TTFT**: 25% → 50%에서 +1.62 ms(46.614 → 48.234), 50% → 100%에서 +4.04 ms. suffix prefill은 비율과 무관하게 약 42 ms이고, 늘어나는 것은 주로 읽기 시간(raw read 2.4 → 3.7 → 6.8 ms)이다.
- **packed 경로 효과 (100%, 같은 KV)**: U100_packed − FullLoad의 TTFT = −18.500 ms [−18.665, −18.336] (0.739×). 파일 1개·preadv 1회·GPU 조립의 효과이며 토큰 선택과 무관하다.
- **저장·setup (이미지당)**: store 6.39 / 7.42 / 9.45 / 11.50 / 21.72 MB (U25 / U30 / U40 / U50 / U100), FullLoad 24.34 MB. persistence 18.8 / 19.9 / 22.2 / 24.3 / 35.4 ms, FullLoad 63.8 ms.
- **3-turn session (shared-stream)**: ReComp 399.4, FullLoad 322.2, U25 268.8, U30 269.7, U40 271.5, U50 272.9, U100 283.0 ms.
- **정합성**: U100_packed ≡ FullLoad 토큰 단위 12,183 / 12,183 일치. ReComp·FullLoad·U25_packed는 PyramidKV run의 같은 요청과도 토큰 100% 일치(12,183 / 12,183씩).
- **Gate**: SG1–SG6, 추가 SGR, 기존 CPU 테스트(non-GLM 556개, 새 테스트 14개 포함), smoke 84 요청 모두 첫 시도에 PASS. freeze amendment 없음.
- **실행**: main 85,281 / 85,281 요청, 실패·스킵 0, 중단 0, 2.80시간. 독립 감사 PASS(111,988 checks), protection PASS, 기존 산출물 삭제 없음.
- **새 파일**: `mmimpress/qwen25/ratio_packed.py`, `scripts/qwen_sweep_support.py`, `scripts/172–176_*qwen_budget_sweep*.py`, `tests/test_qwen_budget_sweep.py`, `docs/qwen_budget_sweep_contract.md`. 기존 소스 수정 없음.
- **프롬프트와 다르게 한 점**: (1) freeze 전에 별도 run root(`runs/qwen_budget_sweep_dev_20261111T1530Z`)에서 gate·integration·smoke를 한 번 리허설했다. (2) SG4(CPU)는 PyramidKV run이 capture한 saliency 점수를 썼고, 이 실행의 점수로는 main의 이미지별 독립 감사가 398장 전부에서 다시 확인했다. (3) SGR gate와 28 layer SDPA 입력 비교를 추가했다. (4) GLM-4.6V 테스트 3건은 이 변경 전부터 실패해 PASS 기준에서 뺐다. (5) 보고 스크립트 176은 freeze 대상에서 제외했고, 첫 보고서 생성 뒤 그림 글자 위치와 기준선 (a)의 경계선 설명 문장을 고쳐 다시 생성했다(수치·규칙·기준선 결과 동일). 그림 팔레트 자동 검증은 node가 없어 실행하지 못했다. (6) 결과 기록을 프롬프트 문서가 아니라 이 파일에 두었다.
- **NOT_MEASURED**: H2D 시간 단독, NAND cold·O_DIRECT, standalone session의 non-source T1 capture 증분, 동시 요청 throughput, 25–100% 사이의 다른 비율과 25% 미만, LLaVA 25% 값(−2.30 pp, 552개)의 재측정, MT-VQA·GLM·LLaVA.
