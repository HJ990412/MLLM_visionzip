# PosShift 용량-반응 진단 — run `posshift_dose_20261103T060151Z`

계약 `docs/posshift_dose_contract.md`. 40 GQA images × T2–T6 = 200 hit / 조건, Ours store(전체 T1 문맥, 원래 위치)를 이미지마다 다시 만들고 text 시작 위치만 `(v0+V) − δ`로 옮겼다. 기준(base)은 `text_position_start=None`의 Ours hit. δmax = V − k − n_sep (이 workload에서 1080–2160) 는 061243Z PosShift와 같은 시작 위치다. 이 실험은 진단이며 arm 우열을 판정하지 않고 061243Z의 사전 판정도 바꾸지 않는다.

## 1. Gate

| Gate | 결과 |
|---|---:|
| Stages | {"SMOKE": "PASS", "GQA": "PASS"} |
| G-B 위치 검증 (prefill = t(δ)+arange, decode 연속) | 1800 / 1800 |
| G-C δ=0 logits bit-identical to base | 200 / 200 |
| G-D base hit tokens == 061243Z ours rows | 200 / 200 |
| 모든 조건 같은 bytes/preads | True |
| δmax > 1024 (모든 이미지) | True |
| G-E protection before/after | PASS |
| Report-only: shift_max tokens == 061243Z posshift rows | 200 / 200 |

## 2. δ별 hit EM 곡선

| 조건 | δ | Hit EM % | EM − base pp [95% CI] | 첫 토큰 = base % | 토큰열 = base % | 16-token cap % | 빈 출력 % | 서로 다른 출력 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours | base | 56.50 | — | 100.0 | 100.0 | 0.0 | 0.0 | 73 |
| shift_0 | 0 | 56.50 | +0.00 [+0.00, +0.00] | 100.0 | 100.0 | 0.0 | 0.0 | 73 |
| shift_1 | 1 | 58.00 | +1.50 [-0.50, +4.50] | 95.0 | 95.0 | 0.0 | 0.0 | 70 |
| shift_4 | 4 | 57.50 | +1.00 [-2.00, +4.00] | 89.0 | 88.5 | 0.0 | 0.0 | 72 |
| shift_16 | 16 | 58.50 | +2.00 [-1.50, +6.00] | 89.5 | 89.5 | 0.0 | 0.0 | 70 |
| shift_64 | 64 | 57.00 | +0.50 [-4.00, +5.00] | 88.5 | 87.5 | 0.0 | 0.0 | 69 |
| shift_256 | 256 | 15.50 | -41.00 [-50.50, -32.00] | 14.0 | 8.0 | 32.5 | 0.0 | 175 |
| shift_1024 | 1024 | 0.00 | -56.50 [-64.00, -48.50] | 0.0 | 0.0 | 100.0 | 73.5 | 41 |
| shift_max | max | 0.00 | -56.50 [-64.00, -48.50] | 0.0 | 0.0 | 100.0 | 48.0 | 90 |

CI는 image-cluster paired bootstrap 10,000회(seed 1234). 1 pp 미만 차이는 해석하지 않는다. 지연은 endpoint가 아니다.

## 3. 사전 고정 판독

- δ\* (EM ≤ 5%가 되는 가장 작은 δ): **1024** → **점진** (graded).
- 계약 문구: small shifts tolerated, collapse at a large shift: the 0% is read as a function of the size of the position mismatch
- 보조: 점추정 단조 비증가(1 pp 허용) = False.

## 4. 한계

- 같은 40-image 개발 세트, 200 hit/조건. 진단 목적의 탐색적 곡선이며 새로운 arm 비교가 아니다.
- base와 δ=0은 같은 store·같은 bytes를 읽고 position_ids만 다르게 들어간다(G-C). store는 다시 만들었고 G-D로 이전 run과의 재현을 확인했다.
- attention 분포는 측정하지 않았다. 붕괴 기전은 출력과 EM 곡선으로만 기술한다.
