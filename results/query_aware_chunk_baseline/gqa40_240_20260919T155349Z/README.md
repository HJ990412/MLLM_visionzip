# QA-Chunk25 GQA pilot artifact

Final validation: **YES**

Five methods were evaluated in the same frozen GQA 40-image/240-question run. Accuracy uses all turns; latency and I/O use cache-hit turns 2–6.

| Method | Accuracy | TTFT (ms) | SSD MB/request | Touched chunks |
|---|---:|---:|---:|---:|
| ReComp | 62.50% | 517.40 | 0.00 | — |
| FullLoad | 62.50% | 648.35 | 1165.07 | 100.00% |
| QA-Token25 | 60.00% | 894.79 | 1196.50 | 96.36% |
| QA-Chunk25 | 58.33% | 439.31 | 348.06 | 24.28% |
| Ours25 | 57.92% | 253.72 | 306.08 | 24.28% |

Artifacts:

- `ANALYSIS.md`: numbered 1–26 report and A–F research answers.
- `results_final.jsonl.gz`: losslessly compressed durable raw request evidence.
- `summary.csv`: five-method headline results.
- `selection_analysis.json`: exact chunk IDs, pairwise Jaccard, and QA-Token overlap.
- `latency_breakdown.csv` / `io_breakdown.csv`: stage and storage metrics.
- `validation.json`: runner validation gates.
- `report_validation.json`: independent reporter/protection gates.

Run evidence: `/home/dblab/hj/mllm_v2/runs/query_aware_chunk_baseline/gqa40_240_20260919T155349Z`

Index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`

Workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`

QA-CHUNK25 BASELINE VALIDATED: YES

Restore the raw request rows with
`gzip -dc results_final.jsonl.gz > results_final.jsonl`.
