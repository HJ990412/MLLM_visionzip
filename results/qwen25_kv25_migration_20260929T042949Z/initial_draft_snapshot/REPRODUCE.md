# Reproduce the Qwen KV25 pilot report

Run root: `/home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z`. GPU gate SHA256: `00f3135738bc3fd0831d05b4c576ca04bebc5c50a174bee2da3eab98d86b169d`.
Source pilot SHA256: `f776d199a6853e9412a23a7958202c5dec2d1f670330db2e59c21dd78830ba48`.
Smoke retry used `/home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/smoke_pilot_retry1`; the first zero-request startup failure remains in `/home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/smoke_pilot/startup_failure.json`.
Report source SHA256: `9f7cf98d837e131f1a91833fefada6990447186f2d08e6536ff51bf1a7b64e4f`.
Storage plan SHA256: `5e218a2744d1c3715042c4725e665b38064521f5ddb67eb5ffd8faa9795bca16`.
Timing addendum SHA256: `9793b19b2f1797cf1aff3154b0ebcd5df74b11a03bc9508e3cdf668f9c853e3b`.
Timing erratum SHA256: `1fafcd3e9b96844f3754afa06e9cd2a09f3272fa447d0a650497eac326e50a00`.

The pilot stores in `runs/qwen25_correctness_v2_20260928T081111Z` are protected read-only. Persistence was NOT_REMEASURED. Do not rerun into the existing output directories.

```bash
python scripts/94_eval_qwen_kv25_pilot.py --dataset smoke --run-dir NEW_SMOKE_DIR --validation /home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/gpu_validation/validation.json
python scripts/94_eval_qwen_kv25_pilot.py --dataset gqa --run-dir NEW_GQA_DIR --validation /home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/gpu_validation/validation.json
python scripts/94_eval_qwen_kv25_pilot.py --dataset mt --run-dir NEW_MT_DIR --validation /home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/gpu_validation/validation.json
python scripts/95_report_qwen_kv25_pilot.py --run-root NEW_RUN_ROOT --results-dir NEW_RESULTS_DIR
```

Exact workload, original manifest hashes, store meta hashes, source hashes, GPU gate and physical request IDs are in each pilot's `manifest.json`, `store_inventory.json`, `config.json` and `raw.jsonl`. Report-side recalculations and per-file hashes are in `report_audit.json`; the separate `scripts/96_audit_qwen_kv25_pilot.py` writes `independent_audit.json`.
