# Reproduce Qwen PrefixStore3

Python: /home/dblab/anaconda3/envs/mllm_ft/bin/python. No dependency upgrade.
Use a NEW timestamped run with source protection bootstrap and REQUEST.md.
The bootstrap record/source snapshots plus final_source_freeze.json bind this run.
Environment: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2.
Prepare additional manifest before outcomes, retain diagnostic attempts, freeze
the documented contract and all source. Use scripts/117_run_qwen_prefixstore3.py
--run NEW_RUN --stage prepare, diagnose, freeze, cpu, core, integration, review,
then main. Do not copy PASS receipts. Main runs actual smoke then all requests.
Resume this unchanged frozen run with --stage main --resume; completed image
evidence is independently checked; partial image is re-executed in a new attempt.
Never invoke legacy 110 directly for this v2 schema/contract.

Replay saved fixed inputs without GPU:
python scripts/116_qwen_prefixstore_semantic.py --fixture FIXED_INPUTS.pt --output NEW.json
All diagnostic tensors, including failure attempts, remain under /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_prefixstore3_20261002T074856Z.
Main uses unchanged native BF16/SDPA; FP64 only diagnostic references.
Analysis: python scripts/118_report_qwen_prefixstore3.py --run /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_prefixstore3_20261002T074856Z --output NEW_OUTPUT
Run final analysis/protection only after timed GPU main has completed. No prior
LLaVA GPU or artifact changes. Raw/payload full hashes, original model snapshot,
processor/tokenizer identity and native code revision are retained. Historical
large-file fingerprints are sampled, not full-byte SHA256 proof.

## Exact validation replay after all workload images have now been observed

For an exact replay, reuse the frozen additional_validation_manifest.json,
validation_selection.json and correctness_manifest.json as INPUTS, without copying
any PASS receipts. Mark them as previously used validation samples; after this
full Qwen run they are not new unseen samples. The prepare command is for a
prospective first run and may have no unobserved candidates now. The archived
protection_bootstrap_replay.py shows the inventory/snapshot policy. A fresh run
also needs REQUEST.md, CONTRACT.md, G7_REVIEW.md and a bootstrap.json recording
its own paths/source protection; all previous results stay read-only.

Run diagnose in a new diagnosis_vN. Run 116 fixed-input replay on its n313060/T2
fixture into a fresh G7_stage_attribution.json, then freeze/cpu/core/integration.
Run the archived repeat_validation.py with ROOT/RUN resolved for the new run,
then review and the archived launch_main.py (with its run-local bindings).
These launchers check actual new gate receipts. After launcher exit, execute
postanalysis_details_v2.py, then terminal_finalize_v3.py without redirecting output into an indexed file. The
terminal index binds all closed logs, raw, source, manifests and reports.

The report-only supplement reads existing raw/receipt files and executes no inference. Run the archived postanalysis_details_v2.py in its fresh run before terminal_finalize_v3.py. It preserves the 118 original durable-publication analysis and reports the already-measured outer writer-return boundary in the root report. See reporting_details/reporting_amendment.json for full input hashes. Do not execute terminal_finalize.py or terminal_finalize_v2.py for this terminal revision.
