# Reproduction and resume

Run directory: `/home/dblab/hj/mllm_v2/runs/llava_mt_gqa_allhead_kv25_main_20260930T093251Z`. Source/config/manifest changes require a new frozen run; do not edit them while main is active.

```bash
cd /home/dblab/hj/mllm_v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/102_validate_llava_mt_gqa_allhead_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/llava_mt_gqa_allhead_kv25_main_20260930T093251Z --validate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/100_eval_llava_mt_gqa_allhead_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/llava_mt_gqa_allhead_kv25_main_20260930T093251Z --phase all --resume
# After completion, independent re-audit to an unused directory:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/101_audit_llava_mt_gqa_allhead_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/llava_mt_gqa_allhead_kv25_main_20260930T093251Z --output-dir /home/dblab/hj/mllm_v2/results/llava_mt_gqa_allhead_kv25_main_20260930T093251Z_reaudit_01
```

Validation may be skipped only when integration_validation.json is PASS and its frozen source hashes match. Smoke/main resume skips independently audited atomic image commits only. Incomplete images restart from T1 with new physical IDs; failed attempt/raw/scratch remain. Main requires 60,915 final requests; starting a process is not completion.
