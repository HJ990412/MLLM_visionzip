# 결과 기록 — LLaVA-NeXT PYR-beta-raw packed MT-VQA 25k 단일 arm

- Run `runs/llava_next_mtvqa_pyr_20261114T182331Z`, 결과 `results/llava_next_mtvqa_pyr_20261114T182331Z` (REPORT.md, decision.json, summary.json). 작성 2026-11-16.
- 요청: main 75,000/75,000, 실패 0, 감사 PASS, main 23.55 h. G-det-T1 25,000/25,000, G-det-U 120/120.
- (A) **IMPROVED**: D = PYR − [B] = +0.7364 pp [+0.5916, +0.8824] (followup 50.8466 % vs 50.1102 %, cross-run, 결정성 확인됨). D ≥ +1.0 pp: 아니오, 손실 회복률 44.3 %.
- (B) PYR − ReComp = -0.9262 pp [-1.0846, -0.7638]; PYR − FullLoad = -0.9340 pp [-1.0954, -0.7708]; PYR − SparseVLM-SSD-KV25-AllHead = -0.7246 pp [-0.8830, -0.5618].
- (D) followup TTFT 228.34 ms (p50 228.67, p95 269.88), T1 TTFT 527.00 ms, R_in = 0.4333; SSD 307.75 MB/followup, preadv 1회. v3 TTFT는 cross-run 참고치.
- (E) 3-turn session 1981.98 ms, store 310.48 MB/이미지.
- (F) MT-GQA PYR − U = +1.244 pp(EM) vs MT-VQA D = +0.736 pp(VQA score).
- main arm은 PYR-beta-raw_packed 하나이며 ReComp·FullLoad·SparseVLM·[B]는 v3(2026-10) 기록이다. MPIC 제외.
