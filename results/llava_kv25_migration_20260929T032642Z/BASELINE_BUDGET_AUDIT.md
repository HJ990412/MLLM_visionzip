# Budget checks before a future main rerun

These are checks to perform before comparing QA-Chunk25 or ReKV-Chunk25 with LLaVA Ours-KV25. Their existing behavior and results were not changed in this migration.

## QA-Chunk25

- Freeze the query-dependent score source and the exact per-layer chunk-count rounding rule from `scripts/52_eval_query_aware_chunk_baseline.py` and `mmimpress/serve.py`.
- For every image, count non-structural content positions actually visible to attention after whole-chunk loading; do not equate a 25% chunk count with 25% content retention.
- Separate probe-key reads, normal K/V payload, separator sidecar, repeated file reads, and any structural rows read in normal chunks.
- Check selected chunk IDs and attended content IDs for each layer and question, including whether query changes selection.

## ReKV-Chunk25

- Freeze Stage A query/history input, layerwise scoring rule, candidate chunks and integer chunk-budget rounding from `mmimpress/rekv.py` and `scripts/70_eval_rekv_gqa.py`.
- Count per-layer attended non-structural content positions after whole-chunk loading and any compact-cache assembly; do not infer retention from the nominal chunk ratio.
- Separate Stage A probe/raw-key I/O, Stage B normal K/V payload, separator handling, duplicated reads and actual OS-returned bytes.
- Check history dependence, selected IDs, logical positions/RoPE and cache-state isolation at T2/T3.

For both baselines, record T1 persistence, metadata activation, normal payload and total physical reads, and the exact attended content denominator on the same frozen manifest before making any equal-budget claim.
