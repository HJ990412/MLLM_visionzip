# Reproduce this GQA pilot

Frozen run: `/home/dblab/hj/mllm_v2/runs/llava_spatial_kv25_20260929T091132Z`. Use the validated Conda interpreter `/home/dblab/anaconda3/envs/mllm_ft/bin/python` and the cached LLaVA-HF checkpoint at revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`. Run the CPU and GPU correctness gates against these exact source hashes before rerunning the smoke and pilot. After stores were cleaned, rebuilding requires each arm's normal image Turn 1 and fresh store persistence. Keep GQA questions[4:10], seed 1234, 64-token chunks, NF4, BF16 compute, FP16 SSD payload, eager attention, and greedy max_new_tokens=16. The raw rows and selection JSON are retained for analysis without rebuilding. The primary paired comparison is SpatialUniform minus IndexUniform; the 54:10 reference uses floor(10*k/64) auxiliary rows. The fixed independent coverage diagnostic is an 8 by 8 grid per branch.


Independent result verification (without rebuilding stores):
`/home/dblab/anaconda3/envs/mllm_ft/bin/python results/llava_spatial_kv25_20260929T091132Z/strict_external_audit.py runs/llava_spatial_kv25_20260929T091132Z --phase pilot`.
The first smoke failure and retry source freezes remain as separate immutable run artifacts.
