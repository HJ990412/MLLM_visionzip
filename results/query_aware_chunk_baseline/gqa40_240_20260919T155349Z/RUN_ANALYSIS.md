# QA-Chunk25 GQA pilot

Accuracy는 전체 질문, TTFT/I/O는 cache-hit turns 2–6의 request mean이다. MB는 10^6 bytes다.

| Method | Accuracy | TTFT | Query-aware | Physical Chunk Budget | SSD MB | SSD Ratio | Selector ms | Touched Chunks | Preads |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.5000% | 517.3994 | No | — | 0.0000 | 0.0000% | 0.0000 | — | 0.0000 |
| FullLoad | 62.5000% | 648.3500 | No | 100.0000% | 1165.0728 | 100.0000% | 0.0000 | 100.0000% | 64.0000 |
| QA-Token25 | 60.0000% | 894.7906 | Yes | 96.3596% | 1196.5016 | 102.7220% | 115.2995 | 96.3596% | 157.5400 |
| QA-Chunk25 | 58.3333% | 439.3075 | Yes | 24.2814% | 348.0589 | 29.8929% | 97.7944 | 24.2814% | 305.7900 |
| Ours25 | 57.9167% | 253.7177 | No | 24.2814% | 306.0793 | 26.2766% | 0.2094 | 24.2814% | 65.0000 |

## QA-Chunk25 definition

SparseVLM raters와 probe Q/K token importance는 QA-Token25와 동일하다. Separator를 제외한 valid spatial token score를 canonical raster SSD chunk별 mean으로 집계하고, `round(0.25 × n_chunks)`와 동일한 Ours budget helper로 상위 chunk를 직접 선택한다. Probe/selected K/V/separator I/O와 모든 online selection 단계는 TTFT critical path에 포함한다.

## Query dependence and overlap

- Query scoring calls: 6400
- Selected normal chunk ratio: 24.2814%
- Pairwise chunk Jaccard: 0.718287
- Consecutive-query Jaccard: 0.724685
- Identical selection rate: 0.0000%
- QA-Token touched vs QA-Chunk selected Jaccard: 0.252764

## QA-Chunk25 latency and I/O

- Selector wall/alias: 97.7944 ms
- Rater/projection/probe-I/O/scoring/aggregation/top-k: 1.7218 / 13.1386 / 34.6916 / 8.1445 / 9.0996 / 5.6933 ms
- ID-D2H/planning/chunk-I/O/scatter/prefill: 1.6852 / 1.1540 / 226.0271 / 47.3204 / 416.8239 ms
- Probe/selected payload/total SSD: 54.6128 / 273.4183 / 348.0589 MB
- Runs/layer, mean/max run, preads: 4.2623, 2.0156, 11.0000, 305.7900

## Direct answers

- QA-Token25 대비 SSD 감소: 70.9103%
- QA-Token25 − QA-Chunk25 accuracy: +1.6667 pp
- QA-Chunk25 − Ours25 accuracy: +0.4167 pp
- QA-Chunk25 − Ours25 TTFT: +185.5898 ms (1.7315×)

## Completion

- Expected/completed/failed/duplicates: 1200/1200/0/0
- Workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- Canonical source store: `/home/dblab/hj/mllm_v2/runs/query_aware_baseline/gqa40_240_final_store` (read-only reuse)

## Limitations

- GQA questions are independent cache-hit turns; no conversation history exists.
- Turn 1 is normal pixel inference and has no query-aware chunk selection.
- The validated canonical raster/image-only stores are reused read-only; store build cost is not rerun.
- v_hidden is opened before request TTFT; per-request H2D/rater work remains inside TTFT.
- Buffered pread plus POSIX_FADV_DONTNEED cannot flush an SSD controller cache.
- Component timings can overlap; TTFT and selector wall fields are authoritative, not their arithmetic sum.

QA-CHUNK25 BASELINE VALIDATED: YES
