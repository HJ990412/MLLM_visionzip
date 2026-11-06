# Reproduction

Environment: `conda activate mllm_ft` (torch 2.5.1, transformers 4.57.6), RTX 4090, no other GPU process.
Use a fresh timestamped run name; nothing is overwritten (no-clobber creation).

```bash
RUN=runs/recompute_kv25_ablation_$(date -u +%Y%m%dT%H%M%SZ)
python scripts/134_eval_recompute_kv25_ablation.py --run $RUN --freeze
PYTHONPATH=.:tests python -m unittest -v test_recompute_kv25 test_prefix25_persistence test_llava_kv25 \
    test_image_only_repack test_visdial_turn1_piggyback_core > $RUN/cpu_tests.log 2>&1
python scripts/134_eval_recompute_kv25_ablation.py --run $RUN --protect before
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/134_eval_recompute_kv25_ablation.py --run $RUN --gpu correctness,smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/134_eval_recompute_kv25_ablation.py --run $RUN --gpu gqa
python scripts/134_eval_recompute_kv25_ablation.py --run $RUN --protect after
python scripts/135_report_recompute_kv25_ablation.py --run $RUN
```

Inputs are copied from `runs/prefix25_persistence_20261001T050113Z` (GQA manifest `40e23fa7…`, correctness
manifest) and verified by hash at freeze. `gpu_freeze.json` binds every source file (except this report script)
and the GPU stages refuse to start if any changed. Raw rows: `runs/recompute_kv25_ablation_20261102T061243Z/llava/gqa/raw_requests.jsonl`;
persistence receipts `persistence.jsonl`; per-image G1/G2 receipts and per-layer KV differences `llava/gates.jsonl`;
G4 evidence `llava/gpu_correctness.json`. Stage-1 payload stores are deleted per image after their receipts
(guarded to `llava/stores/`; log `llava/cleanup.jsonl`) and must be regenerated; G4 stores and smoke Recomp stores
are retained in the run directory.

## Addendum: PosShift 출력 유형 (2026-11-03, REPORT의 "PosShift 출력 유형" 절)

REPORT.md의 이 절은 135가 생성하지 않는다. 135를 다시 실행하면 사라지므로 아래 명령으로 다시 만든다.
GPU는 필요 없고, 원본 raw는 읽기만 한다. 파일은 `results/recompute_kv25_ablation_20261102T061243Z/posshift_output_types/`에 있다.

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
D=results/recompute_kv25_ablation_20261102T061243Z/posshift_output_types
RAW=runs/recompute_kv25_ablation_20261102T061243Z/llava/gqa/raw_requests.jsonl
# 1) hit 200행 추출 + 표 1(첫 토큰, 길이, 정답 수). 추출본은 새 경로에 써서 기존 파일과 비교한다.
$PY -I $D/posshift_outputs.py $RAW /tmp/posshift_hits_check.json
sha256sum /tmp/posshift_hits_check.json   # d356ea92…94d3cd3 와 같아야 함
# 2) 표 2–4 (유형 분류, 토큰 디코딩 예시, 턴별 분포; tokenizer llava-hf/llava-v1.6-vicuna-7b-hf, 오프라인 캐시)
$PY -I $D/posshift_types.py $D/posshift_hits.json
# 3) 표 5 (이미지 단위 경향)
$PY -I $D/posshift_per_image.py $D/posshift_hits.json
```

입력:

| 파일 | sha256 |
|---|---|
| `runs/recompute_kv25_ablation_20261102T061243Z/llava/gqa/raw_requests.jsonl` | `0f6c839a8062c5a8d03ec48417263af626fe82481d5ca40b279deaf4506c34e9` |

`posshift_output_types/`:

| 파일 | 내용 | sha256 |
|---|---|---|
| `posshift_outputs.py` | raw → hit 추출, 표 1 | `4c3767ba89568e647ef74e1e83e46a6548d6c3544f6785a6f6f248668440d077` |
| `posshift_types.py` | 표 2–4 | `15a1ec671ff8b02cff68f7e1f45a805576fe8dd2ebd20c4398b74015d91e6dd8` |
| `posshift_per_image.py` | 표 5 | `8db5434d14aa23888b56c31afab05652cbe870b7431c155678fd83fa036e2f9e` |
| `posshift_hits.json` | PosShift hit 200행 추출본 | `d356ea92c26e9c0d2f864c85515c6f4422894d499bda09cfa36088b0a94d3cd3` |
| `posshift_outputs.log` | 1)의 출력 | `372c9c5243dcec601ecfeda1fd7517bcf52c3e732ac50c5cfa790b1f5cab2c0b` |
| `posshift_types.log` | 2)의 출력 | `d14df056578cd9f004df3e0b2d251ab5950b35e1b484911c5e753c34ca579f21` |
| `posshift_per_image.log` | 3)의 출력 | `bb09779acd687937249ee0775295f362d1635800c949d2f8b5168a8748d1f8a0` |

`posshift_hits.json`과 1)의 스크립트, 2)의 스크립트는 2026-11-03 분석 때 쓴 파일을 그대로 옮겼다. 3)은 그때 실행한 인라인 명령을
그대로 파일로 만든 것이다. 옮긴 뒤 다시 실행한 추출본은 `posshift_hits.json`과 바이트 단위로 같았고, raw sha256은 바뀌지 않았다.
