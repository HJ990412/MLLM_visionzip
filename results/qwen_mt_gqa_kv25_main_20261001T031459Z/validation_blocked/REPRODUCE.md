# Qwen MT-GQA KV25 main 재현

Run: /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z
Core revision: 3355fb299f4d97098efd3ecafe42b29049762e4cda25febb83e3e2347f6b4c31
Contract SHA256: 24d91c1d4ef7f979287cf813eb00a6d4c4ae12ed377197f21b68477e0439cc4a
Config SHA256: 839033e919de9330aa2cf5231dc7282527f354d039dc94ebe3a40632c53582aa
Manifest SHA256: e538d0eaef705c8c16937953287781255270df9557258939a84435429f06a07f

새 GPU core 검증:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/91_validate_qwen25_kv25.py --out-dir NEW_VALIDATION_DIRECTORY --freeze /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z/gpu_freeze.json --llava-protection-receipt /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z/llava_protection_receipt.json
```

현재 run의 통합검증은 최초 1회 실행하며 raw/판정을 덮어쓰지 않는다:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/105_validate_qwen_mt_gqa_kv25_main.py --run-dir /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z
```

통합검증 PASS 뒤 smoke와 전체 main 실행/중단 재개:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/103_eval_qwen_mt_gqa_kv25_main.py --run-dir /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z --phase all --resume
```

동일 code/config/manifest의 완전한 committed image만 재감사 후 건너뛴다. Partial image는 전체 새 attempt로 재실행하고 기존 evidence를 보존한다.

최종 감사만 재실행할 때 새 output 디렉터리를 지정한다:
```bash
PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/104_audit_qwen_mt_gqa_kv25_main.py --run-dir /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z --output-dir NEW_AUDIT_DIRECTORY
```

Raw: main/images/IMAGE/attempt_NNNN/raw.jsonl; adoption: COMMITTED.json. Main 36,549와 integration36/smoke36 및 GPU reference/fault controls를 구분한다.
