# 재현 절차

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 \
       PYTHONPATH=.:scripts:tests \
       OMP_NUM_THREADS=2 \
       MKL_NUM_THREADS=2
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/qwen_budget_sweep_20261111T152143Z
$PY scripts/172_freeze_qwen_budget_sweep.py --stage vectors                    # 새 run root, 벡터 5개, k 표
#   docs/qwen_budget_sweep_contract.md 를 확정한 뒤:
$PY scripts/172_freeze_qwen_budget_sweep.py --stage freeze --run $RUN           # storage plan, protection before, config/manifest
$PY scripts/173_validate_qwen_budget_sweep.py --run $RUN --stage cpu            # SG1, SG4 + 기존 CPU 테스트
$PY scripts/173_validate_qwen_budget_sweep.py --run $RUN --stage gpu            # SG2, SG3, SG5, SGR (core pair 10개)
$PY scripts/174_eval_qwen_budget_sweep.py --run $RUN --phase integration        # SG6
$PY scripts/174_eval_qwen_budget_sweep.py --run $RUN --phase all                # smoke 84 -> main 85,281 (중단 시 --resume)
$PY scripts/176_report_qwen_budget_sweep.py --run $RUN                          # 독립 감사(175), 곡선, protection after, REPORT
```

고정값: Qwen/Qwen2.5-VL-7B-Instruct `cc594898137f460bfe9f0759e9844b3ce807cfb5`, NF4 double / BF16 / SDPA, seed 1234, greedy max 16, batch 1, CPU threads 2,
transformers 4.57.6, torch 2.5.1+cu121. 벡터 SHA256은 `vectors/vectors_sha256.json`, 소스 SHA256은 `config.json`의 `source_sha256`.
packed hit 경로는 PyramidKV run의 `mmimpress/qwen25/layerwise.py::run_cache_layerwise`를 그대로 쓴다(`mmimpress/qwen25/ratio_packed.py`가 비율만 일반화).
절대 TTFT는 실행마다 달라지므로 같은 실행 안의 arm끼리만 비교한다.
