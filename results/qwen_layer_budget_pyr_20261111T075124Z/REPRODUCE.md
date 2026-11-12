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
RUN=runs/qwen_layer_budget_pyr_20261111T075124Z
$PY scripts/167_freeze_qwen_layer_budget_pyr.py --stage vectors                # 새 run root, 벡터 3개, 검산, k_l 표
#   design_ragged_cache.md 를 run root에 두고 docs/qwen_layer_budget_pyr_contract.md 를 확정한 뒤:
$PY scripts/167_freeze_qwen_layer_budget_pyr.py --stage freeze --run $RUN       # storage plan, protection before, config/manifest
$PY scripts/168_validate_qwen_layerwise_packed.py --run $RUN --stage cpu        # QG1 + 기존 CPU 테스트
$PY scripts/168_validate_qwen_layerwise_packed.py --run $RUN --stage gpu        # QG2-QG6 (core pair 10개)
$PY scripts/169_eval_qwen_layer_budget_pyr.py --run $RUN --phase integration    # QG7
$PY scripts/169_eval_qwen_layer_budget_pyr.py --run $RUN --phase all            # smoke 72 -> main 73,098 (중단 시 --resume)
$PY scripts/171_report_qwen_layer_budget_pyr.py --run $RUN                      # 독립 감사(170), 판정, protection after, REPORT
```

고정값: Qwen/Qwen2.5-VL-7B-Instruct `cc594898137f460bfe9f0759e9844b3ce807cfb5`, NF4 double / BF16 / SDPA, seed 1234, greedy max 16, batch 1, CPU threads 2,
transformers 4.57.6, torch 2.5.1+cu121. 벡터 SHA256은 `vectors/vectors_sha256.json`, 소스 SHA256은 `config.json`의 `source_sha256`.
절대 TTFT는 실행마다 달라지므로 같은 실행 안의 arm끼리만 비교한다.
