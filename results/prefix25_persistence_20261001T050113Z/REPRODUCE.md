# Reproduction

Environment: /home/dblab/anaconda3/envs/mllm_ft/bin/python (see run environment.json).
Input manifests, source before copies, source hashes, GPU freeze, failures and raw
receipts are retained under `/home/dblab/hj/mllm_v2/runs/prefix25_persistence_20261001T050113Z`. Offline cached revisions are fixed in config.
Create fresh timestamped runs/results paths; do not overwrite this run.

```bash
PYTHONPATH=.:tests /home/dblab/anaconda3/envs/mllm_ft/bin/python -m unittest -v test_prefix25_persistence test_llava_kv25 test_qwen25_kv25 test_qwen25_store test_image_only_repack test_visdial_turn1_piggyback_core test_qwen25_runner test_qwen25_vision
# Store CPU output as cpu_regression_v2.log in the new run.
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/106_prefix25_persistence.py --run NEW_RUN --freeze
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python -u scripts/106_prefix25_persistence.py --run NEW_RUN --model llava
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python -u scripts/106_prefix25_persistence.py --run NEW_RUN --model qwen
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/107_report_prefix25.py --run NEW_RUN
```

Run models sequentially on an idle GPU. A failed gate blocks that model's pilots.
CPU unittest discovery uses tests on PYTHONPATH; pytest is not installed.
Actual command logs are in llava_execution.log/qwen_execution.log and failures.jsonl.
Pilot payloads are not retained: regenerate them from each arm's own T1.
Existing artifact protection uses protected_before.jsonl and protection_after.json.
At least 30 GiB free is mandatory; the driver additionally budgets 16 GiB staging.
The large full MT experiment is deliberately excluded.

API: both writers default to storage_policy="full". B explicitly uses
storage_policy="prefix25"; serving uses budget_unit="visual_kv", ratio=.25
against original N. The LLaVA completion protocol requires seal_integrity(store)
after its capture writer; the experiment applies the same seal to A and B.
Unsealed B activation fails. Qwen keeps its native payload-hash checks as well.
LLaVA v2 keeps v_token_num/n_chunks_per_layer as the original virtual GPU geometry;
payload_rows/payload_chunks and stored_row_to_original describe actual SSD rows.
Qwen v2 keeps visual_count as N, full_importance_permutation as the complete rank,
and stored_to_original/stored_row_to_original as k actual stored rows. Neither
partial mapping is treated as a complete original permutation.

The original LLaVA/Qwen measurement freezes and source copies are retained.
The final default replay includes the prospective Qwen timing correction and
completion-envelope guard. final_gpu_regression.py replays frozen GPU samples
under final_gpu_freeze.json; original performance rows are not overwritten.

