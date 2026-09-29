# Reproduce the gated LLaVA KV25 pilot

Run from `/home/dblab/hj/mllm_v2` in the unchanged `mllm_ft` environment after CPU/GPU correctness gates pass. Omit `--run-id` to let the runner create a new UTC run ID; it refuses existing paths.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/89_eval_llava_kv25.py \
  --phase both --mt-images 4 \
  --gpu-gate runs/llava_kv25_migration_20260929T030414Z/gpu_validation_v2.json
```

Completed run ID: `llava_kv25_migration_20260929T032642Z`. The frozen GQA index, MT index, model, source hashes, fixed sample gate, and schedule are recorded in `manifest.json`. Per-image persistence, activation, selection comparisons, disk headroom and retained full Visual KV store hashes are under `runs/llava_kv25_migration_20260929T032642Z/image_artifacts/`.
