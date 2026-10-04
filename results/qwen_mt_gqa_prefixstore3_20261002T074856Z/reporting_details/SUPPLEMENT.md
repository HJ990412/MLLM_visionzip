
## Measured I/O, residency and reporting boundary details

The root report uses the recorded outer writer return for persistence and session accounting. The original final/ report preserves the inner durable-publication endpoint; its wrapper receipt-return tail was outside that endpoint. This is a reporting-only amendment after main completion. No inference, correctness criterion, source freeze or raw measurement changed. reporting_details/paired_analysis.json supplies the matching outer-endpoint CIs.

| Method | Durable publication mean ms | Outer return mean/p50/p95 ms | Receipt-return tail mean ms |
|---|---|---|---|
| FullLoad | 64.232 | 64.294/63.553/73.003 | 0.062 |
| Ours B | 45.911 | 45.969/45.621/51.920 | 0.058 |

| Method | T1 TTFT mean/p50/p95 ms | T2 TTFT mean/p50/p95 ms | T3 TTFT mean/p50/p95 ms |
|---|---|---|---|
| ReComp | 112.050/112.675/119.910 | 112.761/112.932/120.265 | 113.797/113.327/126.943 |
| FullLoad | 112.029/112.710/119.886 | 69.404/69.304/73.174 | 74.151/72.791/84.852 |
| Ours B | 112.034/112.673/119.747 | 52.623/52.497/54.345 | 55.398/55.358/56.967 |

| Method | Original content MB | Stored valid MB | Padding MB | Structure/system/boundary MB | Metadata+seal file MB | Resident metadata MB | Payload FDs (derived) | Activation read MB / preads |
|---|---|---|---|---|---|---|---|---|
| FullLoad | 20.477 | 20.477 | 2.612 | 1.204 | 0.048 | 0.097 | 57.000 | 24.333 / 58.000 |
| Ours B | 20.477 | 5.145 | 0.000 | 1.204 | 0.050 | 0.101 | 57.000 | 6.390 / 58.000 |

Metadata residency is the native recursive Python object-size estimate of decoded meta; it excludes kernel FD memory and other process allocations. The payload FD count is derived from the verified file inventory/open path (one FD per payload), not separate kernel memory measurement. Resident metadata/FDs persist for an active image store; visual KV payloads are reread on every hit. Structure includes system and vision-boundary rows in one sidecar; those overlapping categories are not summed again. FullLoad padding is serialized and read but discarded before H2D; B has zero content padding.

| Recorded component (ms) | FullLoad mean/p50/p95 | Ours B mean/p50/p95 |
|---|---|---|
| prefix_slice_serialization_ms | 0.978/0.959/1.197 | 0.366/0.356/0.445 |
| permutation_ms | 0.633/0.631/0.727 | 0.840/0.833/0.985 |
| kv_materialize_ms | 8.480/8.335/10.086 | 7.418/7.318/8.687 |
| kv_repack_ms | 0.000/0.000/0.000 | 0.652/0.632/0.832 |
| ssd_write_ms | 8.239/8.008/10.231 | 2.944/2.895/3.683 |
| fsync_ms | 32.271/31.793/35.458 | 26.933/26.777/28.434 |
| completion_publication_ms | 0.912/0.897/0.972 | 0.961/0.876/1.003 |
| capture_score_ms | 0.000/0.000/0.000 | 1.351/1.373/1.589 |
| capture_clone_ms | 0.341/0.335/0.372 | 0.347/0.339/0.396 |
| source_post_generation_tail_ms | 0.762/0.744/0.864 | 2.198/2.210/2.529 |

These are native named intervals, some nested: do not sum them into a critical path. kv_materialize includes CPU materialization/D2H; an isolated D2H-only timer and isolated serialization-only timer are NOT_MEASURED. Prefix slicing/serialization and write/fsync retain their native boundaries. Capture score/clone/post-generation tail is already charged once in source T1 E2E and is not added to persistence. Native per-payload full hashes are reused; no second full-payload checksum pass is claimed.

| Method | Followup GPU peak allocated / reserved MB (mean) | Padding read MB (mean) | Extra valid content rows (mean) |
|---|---|---|---|
| ReComp | 6165.429 / 7933.526 | NOT_MEASURED | NOT_MEASURED |
| FullLoad | 6116.931 / 7933.526 | 2.642 | 0.000 |
| Ours B | 6101.512 / 7933.526 | 0.000 | 0.000 |

GPU peaks are request-scoped torch allocator observations in the shared process, including model/coexisting contexts; they are not incremental KV size or independently deployed memory footprints. Per-scope mean/p50/p95 and measured counts are in request_gpu_io_summary.csv. ReComp has no cached visual-KV read; unrelated image-file I/O is not zero. All 36,549 outputs ended with EOS; no empty output or generation cap occurred.
