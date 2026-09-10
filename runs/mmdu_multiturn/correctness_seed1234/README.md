# MMDU append-only DynamicCache validation

Result: **FAIL** for the append-only transformer-cache gate.

## Scope

This run compares full recomputation with one persistent DynamicCache for the first 3 user turns of each of 2 fixed MMDU dialogs. Both paths use the same Vicuna `USER:/ASSISTANT:` canonical prompt and gold teacher-forced prior answers.

Every source image is converted to RGB and resized to 336×336 with Pillow LANCZOS before the LLaVA-NeXT processor. This resize follows the official MMDU LLaVA-NeXT script.

This is not an official answer-quality reproduction: the official generation script wraps turns with `[INST]...[/INST]` and carries generated history, whereas this fixed Vicuna checkpoint uses the repository's model-native canonical wrapper and gold history. The same local prompt is used on both sides, so the result isolates transformer cache equivalence.

- MMDU repository: https://github.com/Liuziyu77/MMDU
- Official LLaVA-NeXT script: https://github.com/Liuziyu77/MMDU/blob/main/model_generation/LLaVa_next_gen_ans.py
- Index: `/home/dblab/hj/mllm_v2/data/mmdu/subsets/correctness_seed1234/index.json` (`5cb36a128954e9605f8a1147b4a02e9e4949b67be955d5d1aceaf211a9145a79`)
- Model: `llava-hf/llava-v1.6-vicuna-7b-hf`
- Quantization: `bitsandbytes NF4 double-quant, bfloat16 compute`
- Attention: `eager`
- Decode: greedy, at most 16 new tokens
- First-logit tolerance: max absolute <= 0.125 and mean absolute <= 0.01
- Context rule: prompt tokens + reserved decode tokens must be <= 4096

## Per-turn comparison

| dialog | turn | prompt tokens | new images | max abs logit diff | mean abs logit diff | first token | generated IDs |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| mmdu:84 | 1 | 2391 | 2 | 0 | 0 | yes | yes |
| mmdu:84 | 2 | 2582 | 0 | 0.125 | 0.015593749 | yes | yes |
| mmdu:84 | 3 | 2793 | 0 | 0.15625 | 0.022636043 | yes | yes |
| mmdu:69 | 1 | 2391 | 2 | 0 | 0 | yes | yes |
| mmdu:69 | 2 | 2594 | 0 | 0.125 | 0.018815337 | yes | yes |
| mmdu:69 | 3 | 2823 | 0 | 0.125 | 0.018876323 | yes | yes |

## Per-dialog gates

| dialog | turns | prefix | tokens | spans | cache length | first token | response |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mmdu:84 | 3 | yes | yes | yes | yes | yes | yes |
| mmdu:69 | 3 | yes | yes | yes | yes | yes | yes |

## Important boundary

`static_diverse_mmdu_gate_passed` is deliberately **false**. This run does not open existing SSD stores, does not concatenate independently computed image-prefix KV tensors, and does not validate Static+Diverse serving for MMDU. It only gates append-only `DynamicCache` correctness.

Detailed token IDs, token strings, visual spans, cache lengths, logits, and responses are in `raw.jsonl`; flattened results are in the CSV files.

## Reproduce

```bash
conda activate mllm_ft
python scripts/17_validate_mmdu_cache.py --index /home/dblab/hj/mllm_v2/data/mmdu/subsets/correctness_seed1234/index.json --run-dir /home/dblab/hj/mllm_v2/runs/mmdu_multiturn/correctness_seed1234 --max-new-tokens 16 --logit-max-atol 0.125 --logit-mean-atol 0.01 --attention eager --load-4bit
```
