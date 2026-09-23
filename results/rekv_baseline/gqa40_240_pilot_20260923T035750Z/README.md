# ReKV-Chunk25 (adapted): validated six-arm GQA pilot

`raw.jsonl.gz` and `summary.csv` are same-run runner exports. Restore the raw
log with `gzip -dc raw.jsonl.gz > raw.jsonl`. `validation.json` is the runner
measurement gate; `report_validation.json` is the independent final gate.

`ANALYSIS.md` contains the comparisons and limitations. `parity_tests.json`, `cache_handoff_validation.json`, and `position_validation.json` preserve the three-image smoke evidence. `source_revision.json` records source hashes and the historical MPIC smoke server telemetry difference.
