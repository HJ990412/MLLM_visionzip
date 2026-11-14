# Reproduce (LLaVA-1.5-7B MT-GQA HOLD)

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/llava15_mt_gqa_<new timestamp>
$PY scripts/177_freeze_llava15_mt_gqa.py --stage vectors --run $RUN      # 4 vectors (PYR-b20 copied), N = 576 rows / bytes, model + processor record
# write / check docs/llava15_mt_gqa_contract.md, then:
$PY scripts/177_freeze_llava15_mt_gqa.py --stage freeze --run $RUN --est-hours 3 6   # storage plan, protection before, manifest, config
$PY scripts/178_validate_llava15_mt_gqa.py --run $RUN --stage cpu        # LG8, LG9
$PY scripts/178_validate_llava15_mt_gqa.py --run $RUN --stage gpu        # LG1-LG7 on the 10 fixture images
$PY scripts/179_eval_llava15_mt_gqa_hold.py --run $RUN --stage integration   # LG10
$PY scripts/178_validate_llava15_mt_gqa.py --run $RUN --stage summary
$PY scripts/179_eval_llava15_mt_gqa_hold.py --run $RUN --stage smoke
$PY scripts/179_eval_llava15_mt_gqa_hold.py --run $RUN --stage main      # --resume after an interruption
$PY scripts/180_report_llava15_mt_gqa.py --run $RUN
```
The pinned model snapshot must be in the local HF cache (`llava-hf/llava-1.5-7b-hf` @ `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`). A reproduction uses a new
run root; existing runs / results / stores are never modified. HOLD payloads (about 604 MB per image) are deleted after each image's audit
and commit (run-local scratch only). After a recorded `freeze_amendment_<n>.json`, re-freeze with `177 --stage refreeze` and re-run every gate.
