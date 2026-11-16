# INTERIM — LLaVA-NeXT PYR-beta-raw packed MT-VQA 25k, 처음 2,500장 (판정 아님)

Run `runs/llava_next_mtvqa_pyr_20261114T182331Z`. 작성 2026-11-15 06:18:44 KST. **이 문서는 사전 고정된 1회 중간 보고이며 판정이 아니다.** 실행은 멈추지 않고 계속되며, 이 결과로 arm·설정·규칙을 바꾸지 않는다. 최종 판정은 25,000장 완주 후 `REPORT.md`와 `decision.json`에만 있다.

- 대상: main manifest 순서의 처음 2,500장(무작위 표본 순서이므로 무작위 부분집합), 7,500 요청(followup 5,000).
- main에서 실행한 arm은 PYR-beta-raw_packed 하나다. ReComp·FullLoad·SparseVLM·[B] 수치는 v3 run(2026-10)의 같은 이미지 기록이며 이번에 다시 실행하지 않았다.
- 아래 "판정 규칙 결과" 칸은 같은 규칙을 2,500장에 기계적으로 적용한 값일 뿐이다(판정 아님).
- 독립 감사(2,500장): PASS.

정합성 G-det-T1: PYR arm T1 생성 token = v3 T1 **2,500/2,500** → PASS (cross-run 정확도 비교: ALLOWED).

### (A) PyramidKV 배분 효과 — D = PYR-beta-raw_packed − [B] (followup 공식 VQA score, cross-run, 결정성 확인됨)

| 항목 | 값 |
|---|---|
| PYR-beta-raw packed followup | 50.1640 % |
| [B] v3 followup (같은 이미지) | 49.4940 % |
| D, 95% CI | +0.6700 pp [+0.2060, +1.1380] |
| 판정 규칙 결과 | **IMPROVED** |
| 참고: D ≥ +1.0 pp (MT-GQA GO 기준) | 아니오 |
| 참고: 손실 회복률 D / (ReComp − [B]) | 36.0 % (D / (ReComp - [B]) on the same images; ReComp - [B] = 1.8600 pp) |
| 조건: SSD MB/followup ≤ [B] × 1.02 | 308.41 vs 308.41 MB → 충족 |

### (B) 원래 성능 대비 (followup, cross-run, 결정성 확인됨)

| 비교 | Δ, 95% CI | baseline % | baseline 대비 % | CI |
|---|---:|---:|---:|---|
| PYR − ReComp | -1.1900 pp [-1.7080, -0.6800] | 51.3540 | 97.68 | CI excludes 0 |
| PYR − FullLoad | -1.2300 pp [-1.7540, -0.7040] | 51.3940 | 97.61 | CI excludes 0 |
| PYR − SparseVLM-SSD-KV25-AllHead | -1.0760 pp [-1.5840, -0.5720] | 51.2400 | 97.90 | CI excludes 0 |

### (C) turn별 공식 VQA score (%)

| arm | T1 | T2 | T3 | All | Followup |
|---|---:|---:|---:|---:|---:|
| PYR-beta-raw packed (this run) | 53.0480 | 49.5040 | 50.8240 | 51.1253 | 50.1640 |
| ReComp (v3) | 53.0480 | 50.5240 | 52.1840 | 51.9187 | 51.3540 |
| FullLoad (v3) | 53.0480 | 50.5640 | 52.2240 | 51.9453 | 51.3940 |
| SparseVLM-SSD-KV25-AllHead (v3) | 53.0480 | 50.2040 | 52.2760 | 51.8427 | 51.2400 |
| PrefixStore-KV25 [B] (v3) | 53.0480 | 48.5760 | 50.4120 | 50.6787 | 49.4940 |

turn별 Δ (PYR − baseline, cross-run, 결정성 확인됨):

| baseline | T1 | T2 | T3 | All | Followup |
|---|---|---|---|---|---|
| ReComp | +0.000 pp [+0.000, +0.000] | -1.020 pp [-1.776, -0.268] | -1.360 pp [-2.036, -0.700] | -0.793 pp [-1.139, -0.453] | -1.190 pp [-1.708, -0.680] |
| FullLoad | +0.000 pp [+0.000, +0.000] | -1.060 pp [-1.832, -0.280] | -1.400 pp [-2.088, -0.724] | -0.820 pp [-1.169, -0.469] | -1.230 pp [-1.754, -0.704] |
| SparseVLM-SSD-KV25-AllHead | +0.000 pp [+0.000, +0.000] | -0.700 pp [-1.436, +0.036] | -1.452 pp [-2.148, -0.760] | -0.717 pp [-1.056, -0.381] | -1.076 pp [-1.584, -0.572] |
| PrefixStore-KV25 [B] | +0.000 pp [+0.000, +0.000] | +0.928 pp [+0.240, +1.640] | +0.412 pp [-0.216, +1.056] | +0.447 pp [+0.137, +0.759] | +0.670 pp [+0.206, +1.138] |

answer_type별 followup Δ (탐색적, 다중비교 보정 없음):

| answer_type | 요청 수 | PYR % | [B] % | PYR − [B] | ReComp % | PYR − ReComp |
|---|---:|---:|---:|---|---:|---|
| yes/no | 1,888 | 44.6928 | 44.4386 | +0.254 pp [-0.215, +0.738] | 45.0477 | -0.355 pp [-0.817, +0.097] |
| number | 618 | 64.2233 | 62.0388 | +2.184 pp [+0.189, +4.238] | 65.2427 | -1.019 pp [-3.410, +1.290] |
| other | 2,494 | 50.8220 | 50.2125 | +0.609 pp [-0.103, +1.341] | 52.6864 | -1.864 pp [-2.615, -1.126] |

PYR과 [B]의 예측 비교 (정답 = 그 요청의 공식 VQA score > 0):

| 범위 | 요청 수 | 예측 문자열 일치 | 일치율 | PYR만 정답 | [B]만 정답 | PYR score 더 높음 | [B] score 더 높음 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2 | 2,500 | 2,238 | 89.52 % | 54 | 33 | 93 | 63 |
| T3 | 2,500 | 2,289 | 91.56 % | 46 | 37 | 77 | 59 |
| Followup | 5,000 | 4,527 | 90.54 % | 100 | 70 | 170 | 122 |

### (D) 지연·I/O

| 지표 (이번 실행, PYR-beta-raw packed) | mean | p50 | p95 |
|---|---:|---:|---:|
| followup TTFT ms | 223.67 | 225.10 | 268.29 |
| followup E2E ms | 255.99 | 252.67 | 308.08 |
| T2 TTFT ms | 222.88 | 224.23 | 267.67 |
| T3 TTFT ms | 224.46 | 226.07 | 269.26 |
| T1 TTFT ms (full pixel) | 527.81 | 529.62 | 670.06 |
| T1 E2E ms | 559.28 | 560.61 | 703.83 |
| packed read ms (preadv) | 95.64 | 95.83 | 119.35 |
| scatter ms | 82.22 | 82.17 | 100.66 |
| suffix prefill ms | 39.76 | 39.10 | 44.06 |
| prepare ms (read + scatter + mask) | 183.07 | 184.61 | 225.76 |

- SSD 308.41 MB/followup, preadv 1.00회·pread 0.00회/followup, T1의 store 읽기 0 bytes.
- **같은 실행 안 비율 R_in = followup TTFT / T1 TTFT = 223.67 / 527.81 = 0.4238** (T1은 full pixel 재계산). 참고로 v3 같은 이미지에서 ReComp followup / T1 = 1.0161, [B] followup / T1 = 0.5021.

| v3 arm (cross-run 참고치, 절대 TTFT는 실행 간 비교하지 않는다) | followup TTFT ms | T1 TTFT ms | SSD MB/followup | read 호출/followup |
|---|---:|---:|---:|---:|
| ReComp | 530.79 | 522.37 | 0.00 | 0.00 |
| FullLoad | 761.92 | 522.79 | 1174.92 | 64.00 |
| SparseVLM-SSD-KV25-AllHead | 842.20 | 529.09 | 1186.08 | 77.11 |
| PrefixStore-KV25 [B] | 262.23 | 522.31 | 308.41 | 65.00 |

- 환경 변동 지표(판정 아님): 이번 T1 TTFT 527.81 ms / v3 T1 525.18 ms(5 arm 평균, 같은 이미지) = 1.0050 ([B] arm T1 522.31 ms 대비 1.0105). packed read 처리율 3225 MB/s (MT-GQA PYR run 308.05 MB / 97.5 ms = 3159 MB/s).

### (E) 저장·session

| 지표 (이번 실행) | mean | p50 | p95 |
|---|---:|---:|---:|
| store MB/이미지 (logical) | 311.14 | 323.59 | 391.77 |
| persistence ms | 740.14 | 741.97 | 924.04 |
| activation ms | 181.86 | 185.14 | 225.09 |
| 3 turn 요청 E2E 합 ms | 1071.26 | 1079.77 | 1313.27 |
| 3-turn session ms = 3 E2E + persistence + activation | 1993.27 | 2007.58 | 2445.86 |

이미지당 dialogue 1개라 분할 상각이 없다. v3 session(cross-run 참고치, 같은 이미지): ReComp 1681.10 ms, FullLoad 4559.24 ms, SparseVLM-SSD-KV25-AllHead 4777.18 ms, PrefixStore-KV25 [B] 1958.34 ms.

### (F) MT-GQA와의 일관성

MT-GQA(EM)에서는 PYR − U = +1.244 pp [+0.624, +1.864]였다. MT-VQA(공식 VQA score)의 D = +0.670 pp [+0.206, +1.138]. 부호 같음. 지표가 EM과 VQA score로 다르므로 크기를 직접 비교하지 않는다.

한계는 최종 보고서와 같다(cross-run TTFT는 참고치, DONTNEED는 cold NAND 보장이 아님, dense GPU 버퍼 유지).
