# QA-Select25 GQA pilot

SparseVLM-based query-aware SSD baseline versus fixed image-only repacked Prefix25.

## Main results

Cache-hit TTFT includes prompt construction, tokenization, initial H2D, all online selection/I/O/scatter, prefill, and synchronized first-token availability.

| Method | Accuracy (all) | Accuracy (hits) | Cache-hit TTFT | Nominal KV | Actual SSD MB | SSD ratio | Selector ms | Touched chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | 63.00% | 516.50 ms | — | 0.00 | 0.00% | 0.00 | — |
| FullLoad | 62.50% | 63.00% | 650.73 ms | 100.00% | 1165.07 | 100.00% | 0.00 | 100.00% |
| QA-Select25 | 60.00% | 60.00% | 903.46 ms | 25.00% | 1196.50 | 102.72% | 112.52 | 96.36% |
| Ours25 | 57.92% | 57.50% | 258.97 ms | 25.00% | 306.08 | 26.28% | 0.20 | 24.28% |

## I/O locality

| Method | Logical kept tokens | Probe MB | Preads | SSD read ms | Runs/layer | Mean run length |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | — | 0.00 | 0.00 | 0.00 | — | — |
| FullLoad | 100.00% | 0.00 | 64.00 | 399.35 | — | — |
| QA-Select25 | 25.00% | 54.61 | 157.54 | 472.59 | 1.95 | 17.81 |
| Ours25 | 26.28% | 0.00 | 65.00 | 131.30 | 1.00 | 8.53 |

## Online phase costs (cache hits)

| Method | Raters | Q projection | Probe I/O | Q scoring | Top-k | ID D2H | Chunk plan | Chunk I/O | Scatter | Prefill | TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 501.59 | 516.50 |
| FullLoad | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 645.36 | 650.73 |
| QA-Select25 | 1.74 | 12.31 | 35.94 | 8.60 | 3.55 | 25.36 | 0.95 | 567.19 | 152.22 | 852.51 | 903.46 |
| Ours25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 157.90 | 50.44 | 44.32 | 258.97 |

## Direct comparison

- QA-Select25 − Ours25 accuracy (all turns): +2.08 pp
- QA-Select25 − Ours25 accuracy (cache hits): +2.50 pp
- QA-Select25 − Ours25 cache-hit TTFT: +644.49 ms (3.49x)
- QA token-selection mean pairwise Jaccard: 0.7334
- QA consecutive-query mean token Jaccard: 0.7395
- QA identical-selection rate: 0.00%
- QA selection evidence: 200 requests, 400 within-image query pairs; exact per-layer token/chunk IDs are in `selection.json.gz` and `raw.jsonl.gz`.

## Existing-method consistency

- Gate: ReComp exact; rebuilt FullLoad/Ours stores allow at most max(1, ceil(2% × n)) prediction differences and the same discrete accuracy-gap bound.
- ReComp: 240/240 exact predictions; accuracy gap +0.00 pp
- FullLoad: 197/200 exact predictions; accuracy gap +1.50 pp
- Ours25: 199/200 exact predictions; accuracy gap +0.00 pp

## Validation

- PASS: `row_count`
- PASS: `unique_rows`
- PASS: `method_set`
- PASS: `stable_method_metadata`
- PASS: `all_ttft_finite_positive`
- PASS: `turn1_prompt_prediction_first_token_agreement`
- PASS: `cache_hits_no_vision`
- PASS: `qa_query_scoring_called`
- PASS: `ours_query_scoring_zero`
- PASS: `qa_fixed_budget_no_fallback`
- PASS: `qa_raster_no_repack`
- PASS: `ours_image_only_repacked`
- PASS: `selected_chunks_exact_for_selected_tokens`
- PASS: `actual_bytes_accounting`
- PASS: `independent_sparse_bytes_and_preads`
- PASS: `independent_full_load_bytes`
- PASS: `full_load_unchanged_contract`
- PASS: `ours_first_k_sequential_contract`
- PASS: `qa_selector_inside_ttft`
- PASS: `query_dependent_selection_observed`
- PASS: `ours_same_prefix_per_image`
- PASS: `causal_prompt_current_question_only`
- PASS: `future_query_leakage_zero`
- PASS: `persistence_integrity`
- PASS: `recomp_zero_ssd`
- PASS: `reference_comparison_reported`
- PASS: `existing_method_reference_consistency`

## Configuration

- Images/questions: 40 / 240
- Index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`
- Workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- QA layout: canonical/original raster; no repacking
- Ours layout: image-only importance-aware physical repacking
- Both nominal budgets: 25%
- QA fallback/adaptive ratio/recycling/merging/diversity: disabled

## Limitations

- GQA questions are treated as independent cache-hit turns; no conversation history exists in this pilot.
- Turn 1 deliberately has no QA selection set: all four arms use normal pixel inference. Query-overlap evidence therefore covers the five measured cache-hit questions (turns 2..6) per image.
- QA scoring averages the configured probe heads rather than all decoder heads; this is an SSD adaptation, not exact SparseVLM.
- v_hidden.pt is loaded once into CPU RAM when an image context is opened, outside per-request TTFT and SSD accounting; its per-request H2D and rater compute remain inside TTFT.
- Pairwise token Jaccard is the macro mean of per-decoder-layer Jaccards; the global layer-token Jaccard is also retained for every question pair.
- Buffered pread with POSIX_FADV_DONTNEED does not flush an SSD controller cache.

QUERY-AWARE BASELINE VALIDATED: YES

## Compressed raw artifacts

Large JSON artifacts are published losslessly with gzip to keep the Git repository manageable:
`selection.json.gz`, `summary.json.gz`, `validation.json.gz`, and `raw.jsonl.gz`.
Use `gzip -dc FILE.gz > FILE` to restore any original file.
