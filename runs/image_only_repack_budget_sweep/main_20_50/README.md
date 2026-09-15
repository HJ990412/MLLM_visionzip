# GQA ImageOnly-Repack sequential-Prefix budget sweep run

Executed on 2026-09-14 with schema-v2 true-TTFT instrumentation.

This run contains exactly one same-run **ImageOnly-Repack FullLoad** arm and
seven **ImageOnly-Repack + sequential Prefix** arms (20/25/30/35/40/45/50%).
It does not contain ReComp, SparseVLM, or Static+Diverse. The evaluator's
generic `sanity.json` key named `max_abs_static_ttft_minus_prepare_prefill_ms`
is therefore non-applicable; the authoritative Prefix decomposition check is
in the sweep-level `validation.json` and has a maximum residual of
`0.081522041 ms`.

The complete, independently audited analysis is in
[`results/image_only_repack_budget_sweep`](../../../results/image_only_repack_budget_sweep/README.md).

## Measurement definitions

- **True TTFT:** request start through synchronized determination of the first
  output token. For Prefix, it includes first-k selection, SSD reads, cache
  reconstruction/scatter, prompt prefill, and first-token selection.
- **Decode:** immediately after that first-token boundary through synchronized
  completion of the final generated token.
- **E2E:** request start through final-token completion. It equals true TTFT
  plus decode within recorded precision.
- Cache eviction, store creation, ImageOnly VisionZip layout construction, and
  artifact writes are outside the online timer.

## Frozen workload and method

- GQA: 40 images, six questions per image, 240 requests (`[4:10]` slice)
- Index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`
- Ordered workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- Model: `llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4, eager attention
- Greedy decoding, `max_new_tokens=16`, chunk size 64, cold page cache
- One immutable ImageOnly VisionZip physical permutation for every budget
- Retrieval is sequential first-k only; no calibration, query score, static
  score, or diversity selector participates in serving
- Separator policy: sidecar

## Same-run results

| Budget | Correct | Accuracy | True TTFT mean | SSD MB/request | Actual preads/request |
|---:|---:|---:|---:|---:|---:|
| FullLoad | 147/240 | 61.25% | 727.368 ms | 1165.073 | 64 |
| 20% | 131/240 | 54.58% | 233.776 ms | 252.392 | 65 |
| 25% | 139/240 | 57.92% | 265.170 ms | 306.079 | 65 |
| 30% | 141/240 | 58.75% | 300.063 ms | 371.510 | 65 |
| 35% | 142/240 | 59.17% | 336.783 ms | 435.264 | 65 |
| 40% | 142/240 | 59.17% | 375.323 ms | 498.178 | 65 |
| 45% | 145/240 | 60.42% | 404.784 ms | 551.027 | 65 |
| 50% | 144/240 | 60.00% | 435.309 ms | 603.875 | 65 |

Adjacent selected chunks are coalesced into one contiguous normal-KV read per
layer/K-or-V file: 32 layers x K/V = 64 normal preads. Prefix adds one
separator-sidecar pread, for 65 actual calls per request regardless of budget.

## Raw artifacts

- `results.json`: 240 compound requests, each containing all eight arms
- `per_request.csv`: 1,920 flattened request-arm rows
- `results.jsonl`: 1,920 unique request-arm records
- `summary.csv`: evaluator-level descriptive statistics
- `../../../results/image_only_repack_budget_sweep/config.json`: command,
  environment, source/store/workload provenance for the published sweep
- `sanity.json`: generic evaluator checks; use the sweep-level validation for
  the Prefix-specific decomposition check noted above

## Reproduction

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/04_eval.py --index data/index.json --store kvstore_image_only_visionzip --limit 40 --questions 6 --skip 4 --ratio 0.25 --budgets 0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selectors visionzip_repack_prefix --prefix-layout visionzip_image_only --sep-policy sidecar --metric gqa --dataset gqa --expect-images 40 --expect-questions 240 --no-recompute --run-dir runs/image_only_repack_budget_sweep/main_20_50
```
