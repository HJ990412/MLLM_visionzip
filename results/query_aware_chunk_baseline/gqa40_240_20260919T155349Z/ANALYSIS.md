# QA-Chunk25 GQA 40/240 최종 분석

지속 저장된 five-arm raw 결과를 CPU-only reporter가 다시 읽어 독립 집계했고, runner 산출물·provenance·artifact protection을 교차 검증했다. 최종 판정은 **YES**다.

## Main pilot table

Accuracy는 240개 전체 질문, TTFT/I/O는 cache-hit turns 2–6의 request mean이다. MB는 decimal MB(10^6 bytes)다.

| Method | Accuracy | TTFT | Query-aware | Physical Chunk Budget | SSD MB | SSD Ratio | Selector ms | Touched Chunks | Preads |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | 517.40 ms | No | — | 0.00 | 0.00% | 0.00 | — | 0.00 |
| FullLoad | 62.50% | 648.35 ms | No | 100.00% | 1165.07 | 100.00% | 0.00 | 100.00% | 64.00 |
| QA-Token25 | 60.00% | 894.79 ms | Yes | 96.36% | 1196.50 | 102.72% | 115.30 | 96.36% | 157.54 |
| QA-Chunk25 | 58.33% | 439.31 ms | Yes | 24.28% | 348.06 | 29.89% | 97.79 | 24.28% | 305.79 |
| Ours25 | 57.92% | 253.72 ms | No | 24.28% | 306.08 | 26.28% | 0.21 | 24.28% | 65.00 |

## A–F 핵심 답변

### A. QA-Token25 대비 SSD I/O 감소

QA-Chunk25는 1196.50에서 348.06 MB/request로 70.91% 줄였다.

### B. 그 대가의 accuracy 변화

QA-Chunk25−QA-Token25 accuracy는 -1.67 pp다.

### C. 약 25% physical budget에서 Ours 대비 quality

QA-Chunk25−Ours25 accuracy는 +0.42 pp다.

### D. Ours 대비 속도

QA-Chunk25는 Ours25보다 +185.59 ms 차이가 나며 TTFT ratio는 1.7315×다.

### E. latency gap의 관측 구성

prefill Δ≈371.35 ms, selector wall Δ≈97.58 ms, scattered chunk I/O Δ≈74.60 ms. Component는 overlap 가능하므로 이 순위는 진단용이고, TTFT와 observed selector wall이 권위 있는 수치다.

### F. 질문별 chunk 선택 변화

400 within-image query pairs 중 400 pairs가 달랐고, mean pairwise Jaccard=0.718287, consecutive=0.724685, identical rate=0.0000%다.

## 요청된 1–26 항목

### 1. QA-Chunk25 exact algorithm

현재 질문과 causal history에서 SparseVLM text raters를 고르고, 저장된 probe K와 query Q로 layer별 visual-token importance를 계산한다. 이를 physical raster chunk score로 집계해 Top chunks만 K/V pread하고 GPU scatter/mask 후 prefill·generation한다.

### 2. Chunk score aggregation 정의

Main baseline은 `mean_valid_spatial_token_importance` 하나로 고정했다. separator/newline/structural rows와 final-chunk padding은 numerator와 denominator에서 제외하며 실제 valid spatial row 수만 분모로 쓴다.

### 3. 25% chunk budget 계산 방식

`cvpr25.budget_chunk_count(n_chunks, 0.25)`의 `round(n_chunks × 0.25)` semantics를 Ours25와 공유한다. 따라서 token ceil이 아니라 이미지별 Ours와 정확히 같은 normal chunk count다.

### 4. Physical layout

QA-Chunk25와 QA-Token25/FullLoad는 canonical original raster layout을 사용하며 repacking=false다. Ours25만 image-only saliency로 importance-aware repacked layout의 first-k prefix를 사용한다.

### 5. Query-dependent scoring 호출 횟수

QA-Chunk25 query_score_calls 총합은 6,400, chunk_score_calls 총합은 6,400다. Cache-hit request마다 모든 decoder layer에서 한 번씩 실행됐다.

### 6. Selected chunk ratio

normal selected chunk ratio=24.2814%, total touched chunk ratio=24.2814%다. Probe와 separator sidecar는 ratio 밖이지만 byte/latency/pread에는 포함된다.

### 7. Query-pair chunk Jaccard

전체 pair mean=0.718287, consecutive mean=0.724685 (pairs=400, consecutive=160).

### 8. Identical selection rate

Exact layerwise chunk selection identical rate는 0.0000%이며, 다른 pair는 400/400다.

### 9. QA-Token25와 selection overlap

동일 query/layer에서 QA-Token touched chunks와 QA-Chunk selected chunks의 mean Jaccard=0.252764다. 평균 chunk 수는 33.84 대 8.53, intersection은 8.53/layer다. Exact IDs와 per-request intersection은 `selection_analysis.json`에 있다.

### 10. Accuracy

- ReComp: 62.50% (cache-hit 63.00%)
- FullLoad: 62.50% (cache-hit 63.00%)
- QA-Token25: 60.00% (cache-hit 60.00%)
- QA-Chunk25: 58.33% (cache-hit 58.00%)
- Ours25: 57.92% (cache-hit 57.50%)

### 11. TTFT

- ReComp: mean 517.40 ms, p50 524.17, p95 592.55 ms
- FullLoad: mean 648.35 ms, p50 644.63, p95 774.37 ms
- QA-Token25: mean 894.79 ms, p50 904.35, p95 1035.72 ms
- QA-Chunk25: mean 439.31 ms, p50 437.87, p95 537.38 ms
- Ours25: mean 253.72 ms, p50 253.23, p95 312.86 ms

TTFT는 request start부터 synchronized first output token까지며 prompt, tokenization, initial H2D, selector/probe/I/O/scatter, prefill을 포함한다.

### 12. Selector wall-clock

QA-Chunk25 online_selector_total_ms=97.79 ms, decision-host wall=96.07 ms다. Component 합은 overlap 때문에 TTFT decomposition이 아니다.

### 13. Probe I/O

QA-Chunk25 probe=54.612787 MB/request, probe_io=34.69 ms/request다. Probe는 offline으로 숨기지 않고 total SSD/TTFT critical path에 포함했다.

### 14. Selected-chunk I/O

Selected K/V payload=273.418322 MB/request, selected K/V raw pread=185.93 ms/request다. Separator raw pread=6.59 ms이며, 더 넓은 host chunk-I/O interval(두 K/V read, separator/setup·변환 포함)은 226.03 ms다.

### 15. Total SSD MB

QA-Chunk25 total=348.058911 MB/request이며 selected K/V + probe + separator sidecar를 모두 포함한다.

### 16. FullLoad 대비 SSD ratio

QA-Chunk25/FullLoad actual SSD ratio=29.8929%다.

### 17. Touched chunks

QA-Chunk25=24.2814%, QA-Token25=96.3596%, Ours25=24.2814%다.

### 18. Contiguous runs/layer

QA-Chunk25는 4.26 runs/layer, mean run length 2.02, max 11.00 chunks다. Ours25는 1.00 run/layer의 prefix다.

### 19. Preads/request

QA-Chunk25=305.79, QA-Token25=157.54, Ours25=65.00 preads/request다.

### 20. Scatter/prefill breakdown

QA-Chunk25 rater/projection/probe/scoring/aggregation/top-k/ID-D2H/planning/host-chunk-I/O/scatter/prefill은 각각 1.72/13.14/34.69/8.14/9.10/5.69/1.69/1.15/226.03/47.32/416.82 ms다. Host chunk-I/O 중 계측된 selected-K/V/separator raw pread는 185.93/6.59 ms다.

### 21. QA-Token25 대비 변화

Accuracy 변화=-1.67 pp, SSD 감소=70.91%, TTFT 변화=-455.48 ms다.

### 22. Ours25 대비 quality gap

QA-Chunk25−Ours25=+0.42 pp다.

### 23. Ours25 대비 TTFT gap

QA-Chunk25−Ours25=+185.59 ms, ratio=1.7315×다.

### 24. Tests

- CPU test result (verbatim CLI evidence): `Ran 222 tests in 5.615s; OK`
- Runner validation: 35/35 PASS.
- Reporter independent gates: 26/26 PASS.

### 25. Artifact protection

Prior-artifact verification passed=True; before/after manifest=ed9e9620625a26401a7af42532d053a785d384852c869160ed03e5bf7b393e68 / ed9e9620625a26401a7af42532d053a785d384852c869160ed03e5bf7b393e68. Source-store read-only verification passed=True; fingerprint=0cc425d4e5a9ac50bd9b5ea3e452c93e4dfb9f2df48827e1bb2bfefab82ea0a9. Runner/result exports are byte-identical and existing files were not clobbered.

### 26. Limitations

- GQA questions are independent cache-hit turns; no conversation history exists.
- Turn 1 is normal pixel inference and has no query-aware chunk selection.
- The validated canonical raster/image-only stores are reused read-only; store build cost is not rerun.
- v_hidden is opened before request TTFT; per-request H2D/rater work remains inside TTFT.
- Buffered pread plus POSIX_FADV_DONTNEED cannot flush an SSD controller cache.
- Component timings can overlap; TTFT and selector wall fields are authoritative, not their arithmetic sum.
- 이 결과는 고정 GQA 40-image/240-question pilot이며 independent questions를 cache-hit turns처럼 평가했다. 장기 conversational history로 일반화하지 않는다.
- Buffered pread와 `POSIX_FADV_DONTNEED`는 SSD controller cache까지 제거하지 못한다.
- Mean aggregation과 25% budget은 결과를 본 뒤 튜닝하지 않았다.

## Provenance

- Run directory: `/home/dblab/hj/mllm_v2/runs/query_aware_chunk_baseline/gqa40_240_20260919T155349Z`
- Results directory: `/home/dblab/hj/mllm_v2/results/query_aware_chunk_baseline/gqa40_240_20260919T155349Z`
- GQA index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`
- Workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`

### Runner evidence SHA256

| File | SHA256 |
|---|---|
| `results_final.jsonl` | `83c23414b4d42a51c721212272fc33cf1e6d6396b3eda844433255f5db717aa7` |
| `summary.json` | `757f98077d4c4fee8c5364e4b3d598e8eac8ee2dc7d2d77d6b40cba662e8a2dc` |
| `summary.csv` | `068b6d8b56d89c6231644556b776476b5580f7c06998da73a37ef819a5827fc3` |
| `selection_analysis.json` | `2aabce2673e2883776476d1a22f3c2b3444b2c45a183806ee630168489133cd4` |
| `latency_breakdown.csv` | `0a57e6ed324dc4fa5a1360ff4103af1ee4c98c01a62bfa0600bb744374431bee` |
| `io_breakdown.csv` | `37ea45e4fb48534c78614cc446777dc98ab5f2d783ca2568d7d3cfa7b0eccbc0` |
| `validation.json` | `08d9741e6174c8ec38318ea5472e160e2a2a5c59ecd00c971cf7de3c90af6225` |
| `RUN_ANALYSIS.md` | `78c299ff4e265e7a82c74fd4e2c37eff34070d013eac32ac25ded8c57946c045` |

### Report-time source SHA256

| File | SHA256 |
|---|---|
| `mmimpress/sparsevlm.py` | `7b24d592eb36bb5da11fe628d6012192ec1348d5680a8962640251ca73ce2e18` |
| `mmimpress/serve.py` | `116ed0d21f3bdce65e3de663cd794bdfc628014f0d01d9766dbe7046c2f953d7` |
| `mmimpress/store.py` | `0732fc883b7cab8185c2bdf9349b00e0f316b12a2eb504d7106f3ab7793fd97b` |
| `mmimpress/cvpr25.py` | `c3159d2d25797e2264900fe1af06ac522e8452dd436dab09ca8a690c16653b88` |
| `scripts/49_eval_query_aware_baseline.py` | `ea6e3d0d3a50580e3c4cde8c578d377b4e7e92f2dcb6313610bf52b10f4c042d` |
| `scripts/50_protect_query_aware_artifacts.py` | `f0014e1e1147bafef7c8064e739ef811b080b60712a5e2231012ade922f4dd6b` |
| `scripts/51_protect_qa_chunk_source_stores.py` | `67ac4bff6867626c6cd997b40bc30d1134b3975f19e54cc1f4a825bc710150cd` |
| `scripts/52_eval_query_aware_chunk_baseline.py` | `fa1fc8a3193e18910873c367aceb4f95c0cc039ff719bd15ce90ce8274f90e76` |
| `scripts/53_report_query_aware_chunk_baseline.py` | `64b997fccc881f61f3017e758f677c9105b1dd5e6371c8ccda89fe88fb199895` |

## Reproduction

GPU pilot는 detached launcher로 실행하고, 완료 후 protection verify와 이 CPU-only reporter를 실행한다.

```bash
bash scripts/run_qa_chunk25_gqa_background.sh --help
python scripts/53_report_query_aware_chunk_baseline.py --run-dir /home/dblab/hj/mllm_v2/runs/query_aware_chunk_baseline/gqa40_240_20260919T155349Z --results-dir /home/dblab/hj/mllm_v2/results/query_aware_chunk_baseline/gqa40_240_20260919T155349Z --test-result '<observed test summary>'
```

QA-CHUNK25 BASELINE VALIDATED: YES
