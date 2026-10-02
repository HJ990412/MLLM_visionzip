# Qwen MT-GQA 실행 상태

상태: RUNNING; phase: integration; main 채택 완료 0/36,549. 최종 독립 감사 전 VALID 아님.

재개:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/110_eval_qwen_mt_gqa_B.py --run-dir /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/qwen --phase all --resume
```
