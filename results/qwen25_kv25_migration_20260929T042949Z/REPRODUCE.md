# Reproduce the Qwen KV25 pilot report

Validated interpreter: `/home/dblab/anaconda3/envs/mllm_ft/bin/python` (Torch 2.5.1+cu121, Transformers 4.57.6, bitsandbytes 0.49.2). The default `python` lacked Transformers and caused the preserved zero-request smoke startup failure.

Run root: `/home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z`. GPU gate SHA256: `00f3135738bc3fd0831d05b4c576ca04bebc5c50a174bee2da3eab98d86b169d`.
Source pilot SHA256: `f776d199a6853e9412a23a7958202c5dec2d1f670330db2e59c21dd78830ba48`.
Report source SHA256: `c39ba566db9c92b07abcd6e36c32f3e160ed5b793b082da669049655568f2a18`.
Storage plan SHA256: `5e218a2744d1c3715042c4725e665b38064521f5ddb67eb5ffd8faa9795bca16`.
Timing addendum SHA256: `9793b19b2f1797cf1aff3154b0ebcd5df74b11a03bc9508e3cdf668f9c853e3b`.
Timing erratum SHA256: `1fafcd3e9b96844f3754afa06e9cd2a09f3272fa447d0a650497eac326e50a00`.
LLaVA GPU-gate protection receipt SHA256: `b5d24355f6e98674bd836b051d3a008e014db7ee63f34f9b6f489e82dddba435`.
Final **post-pilot** protection receipt: `/home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/final_protection_receipt.json`; SHA256: `e096f5631991d744aba92c071f537cfd3e83e18a8bb23e051566310e211b069d`. A byte-identical copy is `final_protection_receipt.json` beside this file. It verifies 30,205 pre-task files, the 40 external MT image hashes, only the intended runner/store source changes, and 143,916,482,560 free bytes after pilots.

To regenerate this report from the **same raw requests**, choose a fresh results directory; the script will not overwrite an existing one:

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/95_report_qwen_kv25_pilot.py   --run-root /home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z   --results-dir NEW_EMPTY_RESULTS_DIR
```

The report binds the exact `smoke_pilot_retry1`, `gqa_pilot`, and `mt_pilot` directories plus the preserved `smoke_pilot/startup_failure.json`. It is not a generic reporter for another run root. A fresh benchmark requires a new run-local output directory for each arm/dataset, the same gate and protected-store checks, and a newly bound reporting manifest before comparisons. Do not point this report command at a new raw run without updating and freezing those bindings.

The pilot stores in `runs/qwen25_correctness_v2_20260928T081111Z` are protected read-only. Persistence was NOT_REMEASURED. Exact workload, source and store hashes, GPU gate, and physical request IDs are in each phase's `manifest.json`, `config.json`, `store_inventory.json` and `raw.jsonl`. Report-side recalculations and per-file hashes are in `report_audit.json`; the separately implemented `scripts/96_audit_qwen_kv25_pilot.py` writes `independent_audit.json`.


The final protection citation in REPORT.md and this REPRODUCE.md was added after report generation. When regenerating with script 95, append this verified post-pilot receipt citation to those documents; the generated numeric report and summary are otherwise reproduced by the command above.
