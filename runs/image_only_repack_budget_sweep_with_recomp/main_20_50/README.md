# GQA ImageOnly-Repack Prefix budget sweep with same-run ReComp

Executed on 2026-09-14 with schema-v2 true-TTFT instrumentation.

This run contains exactly one ReComp arm, one ImageOnly-Repack FullLoad arm,
and seven sequential-Prefix arms (20/25/30/35/40/45/50%). It contains no
SparseVLM or Static+Diverse serving arm. The authoritative audited analysis is
[`results/image_only_repack_budget_sweep_with_recomp`](../../../results/image_only_repack_budget_sweep_with_recomp/README.md).

## Measurement definitions

- **ReComp true TTFT:** after processor/tokenization and host-to-device input
  preparation, through per-question vision-tower execution, multimodal prompt
  prefill, first-token decision, and CUDA synchronization.
- **SSD-path true TTFT:** request start through selection, SSD reads, cache
  reconstruction/scatter, prompt prefill, first-token decision, and CUDA
  synchronization. Input preparation and cold-cache eviction are outside it.
- **Decode:** after the first-token boundary through final-token completion.
- **E2E:** true TTFT plus decode within the recorded timing precision.

ReComp never accesses the KV store. The raw ReComp `physical_layout` and
`calibration_questions` cells are run-global annotations emitted by the
generic evaluator and are not ReComp semantics; `retrieval=recompute` and
zero SSD bytes/preads are the validated provenance.

## Frozen workload and method

- GQA: 40 images, six questions per image, 240 requests (`[4:10]` slice)
- Index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`
- Ordered workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- Model: `llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4, eager attention
- Greedy decoding, `max_new_tokens=16`
- Cache arms: same ImageOnly VisionZip store, chunk size 64, cold page cache
- Prefix: same immutable permutation, exact first-k, separator sidecar
- No calibration question, online score, static score, or diversity selector

## Same-run results

| Method | Accuracy | True TTFT mean | TTFT p50 | TTFT p95 | SSD MB/request | Actual preads/request |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | 504.108 ms | 508.896 ms | 590.256 ms | 0.000 | 0 |
| FullLoad | 61.25% | 723.969 ms | 724.380 ms | 885.171 ms | 1165.073 | 64 |
| Prefix 20% | 54.58% | 235.445 ms | 234.921 ms | 282.315 ms | 252.392 | 65 |
| Prefix 25% | 57.92% | 262.623 ms | 264.232 ms | 309.973 ms | 306.079 | 65 |
| Prefix 30% | 58.75% | 299.282 ms | 300.531 ms | 353.638 ms | 371.510 | 65 |
| Prefix 35% | 59.17% | 337.508 ms | 339.933 ms | 399.088 ms | 435.264 | 65 |
| Prefix 40% | 59.17% | 375.624 ms | 379.729 ms | 437.434 ms | 498.178 | 65 |
| Prefix 45% | 60.42% | 406.280 ms | 411.303 ms | 474.478 ms | 551.027 | 65 |
| Prefix 50% | 60.00% | 440.426 ms | 443.558 ms | 522.726 ms | 603.875 | 65 |

Adjacent Prefix chunks are coalesced into one contiguous normal-KV read per
layer/K-or-V file: 32 layers x K/V = 64 normal preads. Prefix adds one
separator-sidecar pread, for 65 calls per request.

## Raw artifacts

- `results.json`: 240 compound requests, each containing all nine arms
- `per_request.csv`: 2,160 flattened request-arm rows
- `results.jsonl`: 2,160 unique request-arm records
- `summary.csv`: evaluator-level descriptive statistics
- `sanity.json`: generic evaluator checks; its Static+Diverse-only residual is
  non-applicable here. The Prefix and ReComp decomposition checks are in the
  published sweep-level `validation.json`.
- `../../../results/image_only_repack_budget_sweep_with_recomp/config.json`:
  command, environment, code, store, workload, and analysis provenance

## Reproduction

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/04_eval.py --index data/index.json --store kvstore_image_only_visionzip --limit 40 --questions 6 --skip 4 --ratio 0.25 --budgets 0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selectors visionzip_repack_prefix --prefix-layout visionzip_image_only --sep-policy sidecar --metric gqa --dataset gqa --expect-images 40 --expect-questions 240 --run-dir runs/image_only_repack_budget_sweep_with_recomp/main_20_50
```
