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
