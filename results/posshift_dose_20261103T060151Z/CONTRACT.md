# PosShift 용량-반응(dose-response) 진단 계약

**질문.** `recompute_kv25_ablation_20261102T061243Z`의 Ours-KV25-PosShift hit EM 0.00%(0/200)는
위치 불일치가 커질수록 점진적으로 나타나는 붕괴인가, 아니면 아주 작은 이동에서도 즉시 나타나는
(구현 결함을 의심해야 하는) 붕괴인가. 이 실험은 **진단**이며 어떤 arm의 우열도 판정하지 않는다.
061243Z의 사전 판정은 바꾸지 않는다.

## 고정 입력

- Workload: `runs/prefix25_persistence_20261001T050113Z/gqa_manifest.json`(sha `40e23fa7…`),
  40 GQA images × `questions[4:10]`, T1 = `questions[4]`, hit = T2–T6 (이미지당 5 hit, 총 200).
  061243Z와 같은 workload, 같은 prompt, greedy, max 16 tokens, seed 1234.
- Model/path: LLaVA-NeXT-Vicuna-7B NF4/BF16/eager, FP16 prefix25 store, 64-token chunk, separator sidecar,
  `request_cvpr25(budget=.25, mode='prefix', sep_policy='sidecar')`, DONTNEED는 timer 밖. 134의 `Ablation`
  클래스(T1, persistence, activation, `hit_arm(force_start=…)`)를 그대로 쓰고 새 serving 코드는 없다.
- Store: 이미지당 Ours store 1개를 T1 full forward에서 만든다(134 `persist_arm(arm='ours')`, 동일 writer).
  hit이 끝나면 `guarded_rmtree`로 지운다. 보존하는 store는 없다.

## 조건

Ours store의 K는 원래 위치로 회전돼 있다. text 시작 위치를 `t(δ) = (v0 + V) − δ`로 두고
`δ ∈ {0, 1, 4, 16, 64, 256, 1024, δmax}`, `δmax = (v0 + V) − (v0 + k + n_sep) = V − k − n_sep`
(= 061243Z PosShift와 같은 시작 위치)에서 같은 질문의 hit을 실행한다. `δmax > 1024`를 단언하고,
아니면 그 이미지는 중복 δ를 제외하고 기록한다. 각 (image, turn)마다 추가로 기준 경로
(`text_position_start=None`, 134의 Ours hit과 동일)를 1회 실행한다. 순서는 고정(기준 → δ 오름차순)이고
지연은 endpoint가 아니다. 모든 hit은 같은 store, 같은 bytes를 읽는다.

## Gate (fail-closed)

- **G-A freeze/sha.** 134와 같은 freeze; 151(보고)은 freeze 밖, 150·테스트·계약은 안.
- **G-B 위치 검증.** 모든 요청에서 `PositionRecorder.verify`: prefill `position_ids = t(δ) + arange`,
  decode 연속. 하나라도 실패하면 중단.
- **G-C δ=0 동일성.** δ=0 hit의 모든 step logits가 기준 경로와 `torch.equal`(bit-identical), 200/200.
  (134 G4-b와 같은 성질을 전체 workload에서 확인.)
- **G-D 기준 경로 재현.** 기준 경로의 토큰·예측이 061243Z의 같은 (image, turn) `ours` 행과 동일, 200/200.
  store는 다시 만들지만 T1과 writer는 결정적이다.
- **G-E 보호.** `results/`, `docs/`, `mmimpress/`, `scripts/`, `tests/`, `README.md`,
  `runs/prefix25_persistence_20261001T050113Z`, `runs/recompute_kv25_ablation_20261102T045423Z`,
  `runs/recompute_kv25_ablation_20261102T061243Z`의 sha256 manifest와 `kvstore*` inventory를
  before/after 비교. 새 run/results 디렉터리와 새 소스 5개만 제외. README 변경은 보고한다.

Report-only: δmax hit의 토큰이 061243Z `posshift` 행과 동일한 수(200 기대).

## 사전 고정 판독

δ별 hit EM은 200 hit의 strict normalized EM. `EM(δ) − EM(0)`은 image-cluster paired bootstrap
10,000회, seed 1234. **δ\* = EM(δ) ≤ 5%인 가장 작은 δ.**

| 결과 | 판독 |
|---|---|
| δ\* ≤ 4 | **급격**: 1~4 position 이동으로 붕괴. 구현을 먼저 점검해야 하며 PosShift 해석을 보류한다. |
| δ\* ≥ 16 | **점진**: 작은 이동은 견디고 큰 이동에서 붕괴. 061243Z 0%는 위치 불일치 크기에 따른 결과로 읽는다. |
| 어떤 δ(δmax 포함)에서도 EM > 5% | 061243Z PosShift가 재현되지 않음. 원인 조사 전에는 해석하지 않는다. |

보조(판정 아님): δ별 EM 곡선과 CI, δ=0 대비 첫 토큰 일치율, 16-token cap 도달률(EOS 없음),
단조성(점추정이 δ에 대해 1 pp 허용 범위에서 비증가인지). 1 pp 미만 차이는 해석하지 않는다.

## 실행

```
RUN=runs/posshift_dose_<UTC>
150 --freeze → unittest(test_recompute_kv25, test_posshift_dose) → 150 --protect before
→ 150 --gpu smoke,gqa (smoke: 2 images × 3 turns) → 150 --protect after → 151 --run $RUN
```

예상: 이미지당 ≈ 14 s(T1 0.6 s + persistence 0.6 s + activation 0.2 s + 46 hit × ≈ 0.28 s),
40 images ≈ 10 min, 모델 로드·smoke·protect 포함 ≈ 15 min. 디스크: store 1개(≈ 310 MB) 일시 점유,
run 디렉터리 raw ≈ 수 MB. 기존 run·store는 수정하지 않는다.

## 산출물

`results/posshift_dose_<UTC>/`: REPORT.md, summary.csv(δ별), curve.csv, REPRODUCE.md, analysis.json,
config.json, CONTRACT.md, protection_result.json. README에는 요약 섹션 하나만 추가. 커밋은 요청 시에만.
