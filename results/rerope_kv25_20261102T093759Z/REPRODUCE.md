# Reproduction

Environment: `conda activate mllm_ft` (torch 2.5.1, transformers 4.57.6), RTX 4090, no other GPU process.
Use a fresh timestamped run name; nothing is overwritten (no-clobber creation).

```bash
RUN=runs/rerope_kv25_$(date -u +%Y%m%dT%H%M%SZ)
python scripts/136_eval_rerope_kv25.py --run $RUN --freeze
PYTHONPATH=.:tests python -m unittest -v test_rerope_kv25 test_recompute_kv25 test_prefix25_persistence test_llava_kv25 \
    test_image_only_repack test_visdial_turn1_piggyback_core > $RUN/cpu_tests.log 2>&1
python scripts/136_eval_rerope_kv25.py --run $RUN --protect before
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/136_eval_rerope_kv25.py --run $RUN --gpu correctness,smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/136_eval_rerope_kv25.py --run $RUN --gpu gqa
python scripts/136_eval_rerope_kv25.py --run $RUN --protect after
python scripts/137_report_rerope_kv25.py --run $RUN
```

Inputs are copied from `runs/recompute_kv25_ablation_20261102T061243Z` (GQA/smoke/correctness manifests, GQA manifest
`40e23fa7…`) and verified by hash at freeze; the G-a/G-b evidence file is copied to `dev/` and its SHA256 is in the contract.
`gpu_freeze.json` binds every source file except the report scripts. Raw rows: `runs/rerope_kv25_20261102T093759Z/llava/gqa/raw_requests.jsonl`
(7 arms × 6 turns × 40 images); persistence receipts `persistence.jsonl`; rev.-3 G1/G2 receipts `llava/gates.jsonl`; ReRoPE
G-a/G-b receipts and per-layer differences `llava/gates_rerope.jsonl`; G4 evidence `llava/gpu_correctness.json` and
`llava/gpu_correctness_extension.json`. Stage-1 and smoke payload stores are deleted per image after their receipts
(guarded to `llava/stores/`; `llava/cleanup.jsonl` keeps each store's sealed per-file SHA256 list); G4 stores are retained.
