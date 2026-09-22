# QA-Select25 GQA pilot 최종 분석

**판정:** SparseVLM-based query-aware SSD baseline과 image-only repacked Prefix25를 동일한 nominal 25% budget에서 비교한 고정 GQA 40-image/240-question pilot가 모든 검증을 통과했다.

이 40-image/240-question GQA pilot에서는 QA-Select25가 +2.0833 pp 높은 all-turn accuracy를 보였고, cache-hit TTFT는 Ours25 대비 3.4886배였다. 따라서 이 pilot의 관측 범위에서는 per-query adaptive selection의 품질 효과와 Ours의 selector/locality 이점을 함께 보고해야 하며, 대화형 multi-turn 전체 데이터셋으로 일반화해서는 안 된다.

## Main result

Accuracy는 전체 240개 질문 기준이며, TTFT와 SSD/I/O 열은 cache-hit turns 2–6의 request mean이다. MB는 decimal MB(10^6 bytes)다.

| Method | Accuracy | TTFT | Nominal KV | Actual SSD MB | SSD Ratio | Selector ms | Touched Chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.5000% | 516.5029 ms | — | 0.0000 | 0.0000% | 0.0000 | — |
| FullLoad | 62.5000% | 650.7332 ms | 100.0000% | 1165.0728 | 100.0000% | 0.0000 | 100.0000% |
| QA-Select25 | 60.0000% | 903.4585 ms | 25.0000% | 1196.5016 | 102.7220% | 112.5151 | 96.3596% |
| Ours25 | 57.9167% | 258.9717 ms | 25.0000% | 306.0793 | 26.2766% | 0.2036 | 24.2814% |

## 요청된 1–20 항목

### 1. QA-Select exact algorithm

Turn 1은 네 arm 모두 동일한 normal pixel inference다. QA arm은 그 answer-producing forward에서 canonical Visual-KV, decoder visual hidden state, probe K를 piggyback 저장한다. 각 cache-hit request에서는 현재 질문 suffix와 저장된 visual hidden으로 text raters를 한 번 고르고, decoder layer별 fixed probe-head Q/K attention을 head-mean하여 visual score를 만든다. separator를 제외한 Top-25% token을 고른 뒤 token→64-token chunk→contiguous pread plan으로 변환하고, K/V를 읽어 GPU cache에 scatter/mask한 뒤 prefill·generation한다.

### 2. SparseVLM에서 재사용한 코드

`select_raters()`와 `rater_visual_scores_from_qk(..., head_reduce="mean")`, `select_topk()`/`topk_budget()`를 직접 재사용했다. `rater_visual_scores()`의 rater-to-visual attention 정의는 Q/K 전용 함수가 동일하게 계산하므로 full attention matrix나 full Visual-KV preload는 하지 않는다.

### 3. IMPRESS-specific mechanism

기존 historical path는 보존했지만 QA-Select25에서는 probe-head Jaccard voting/threshold, similarity fallback, full-layer fallback, adaptive layer ratio를 우회했다. SparseVLM의 recycling, merging, diversity/Static+Diverse, calibration question/training도 사용하지 않았다. 측정 validation에서 fallback=0, static/diversity calls=0을 확인했다.

### 4. Physical layout

QA-Select25는 original/canonical raster 순서이며 repacking/order metadata가 없다. separator KV와 probe K만 sidecar다. Ours25는 Turn-1 image-only Vision Encoder saliency 순서로 물리 repack한 뒤 같은 이미지의 후속 질문마다 동일 first-k prefix를 읽는다.

### 5. Retention 계산

두 selective method의 nominal budget은 0.25다. QA는 layer마다 `ceil(0.25 × n_spatial)`개를 골라 logical ratio를 고정하고 separator는 budget 밖 sidecar로 항상 유지한다. Ours는 repacked physical prefix의 chunk-aligned rows를 읽으므로 logical/actual ratio가 nominal과 조금 다를 수 있다. 실제 평균 logical ratio는 QA 25.0000%, Ours 26.2766%다.

### 6. Query-dependent selection 검증

QA cache-hit 200건에서 매 layer query scoring call과 current-question-only prompt를 검증했고, 400개 within-image pair 중 400개가 다른 selection이었다. Ours query scoring call은 0이며 image별 prefix hash는 모든 turn에서 동일했다. future query leakage는 0이다.

### 7. 질문별 selected-token Jaccard

layer-macro pairwise token Jaccard=0.733423, consecutive=0.739549, chunk Jaccard=0.978861, identical-selection rate=0.0000%다. 400개 pair의 question IDs와 layer별 token/chunk Jaccard, 그리고 200개 request의 정확한 selected IDs는 `selection.json`에 보존했다.

### 8. QA selector overhead

QA selector_ms=112.5151 ms, online_selector_total_ms=112.5151 ms/request다. 세부 평균은 rater=1.7375, query projection=12.3107, query scoring=8.6025, top-k=3.5534, ID D2H=25.3594, chunk planning=0.9491 ms다. 이 component들은 CUDA/host overlap이 있어 합을 TTFT decomposition으로 해석하면 안 되며, 모두 measured TTFT critical path 안에 있다.

### 9. Probe I/O

QA probe read=54.612787 MB/request, probe I/O latency=35.9404 ms/request다. Ours는 0.000000 MB와 0.0000 ms로 query probe I/O가 없다.

### 10. Actual SSD MB

- ReComp: 0.000000 MB/request (0.0000% of FullLoad)
- FullLoad: 1165.072794 MB/request (100.0000% of FullLoad)
- QA-Select25: 1196.501565 MB/request (102.7220% of FullLoad)
- Ours25: 306.079334 MB/request (26.2766% of FullLoad)

logical 25%를 actual 25% bytes로 보이게 만들기 위한 selection 왜곡은 하지 않았다.

### 11. Touched chunk fraction

QA=96.3596%, Ours=24.2814%다. QA의 sparse token 분산 때문에 logical ratio와 physical chunk footprint가 다르다.

### 12. Contiguous runs/locality

QA는 layer당 1.9459 runs, 평균 run length 17.8107 chunks, 157.5400 preads/request다. Ours는 layer당 1.0000 run, 평균 8.5250 chunks, 65.0000 preads/request인 first-k sequential prefix(+separator sidecar)다.

### 13. GQA accuracy

- ReComp: all turns 62.5000%, cache hits 63.0000%
- FullLoad: all turns 62.5000%, cache hits 63.0000%
- QA-Select25: all turns 60.0000%, cache hits 60.0000%
- Ours25: all turns 57.9167%, cache hits 57.5000%

### 14. GQA TTFT

- ReComp: mean 516.5029 ms, p50 523.3549 ms, p95 586.2913 ms
- FullLoad: mean 650.7332 ms, p50 654.4443 ms, p95 763.3141 ms
- QA-Select25: mean 903.4585 ms, p50 913.3390 ms, p95 1061.3004 ms
- Ours25: mean 258.9717 ms, p50 258.9660 ms, p95 316.0812 ms

TTFT 경계는 prompt construction 전에 시작해 tokenization, initial H2D, online selection/I/O/scatter, prefill을 거쳐 synchronized first-token availability에서 끝나며 네 method에 동일하다.

### 15. Ours25와 quality gap

QA−Ours accuracy gap은 all turns +2.0833 pp, cache hits +2.5000 pp다.

### 16. Ours25와 TTFT gap

QA−Ours cache-hit mean TTFT gap은 +644.4869 ms이며, QA/Ours ratio는 3.488638×다.

### 17. ReComp/FullLoad 기존 결과 일관성

- ReComp: prediction agreement 240/240, accuracy gap +0.0000 pp
- FullLoad: prediction agreement 197/200, accuracy gap +1.5000 pp
- Ours25: prediction agreement 199/200, accuracy gap +0.0000 pp

### 18. Test result

- Run-level validation: 27/27 checks PASS (`validation.json`).
- CPU test suite: 196 tests passed in 5.421s

### 19. Artifact protection

기존 `runs/`/`results/` 보호 항목 13,120개 (files 12,361, directories 759, symlinks 0)의 before/after manifest SHA256가 `42ec8d56e113412640a56ac40aaa0fe12113aa3958e79c388ff9719a6efb4a88`로 동일했다. missing/added/changed=0/0/0이다. 새 query-aware roots만 제외됐고, top-level KV-store tree는 의도적으로 hash scope 밖이다.

### 20. 발견된 limitation

- GQA questions are treated as independent cache-hit turns; no conversation history exists in this pilot.
- Turn 1 deliberately has no QA selection set: all four arms use normal pixel inference. Query-overlap evidence therefore covers the five measured cache-hit questions (turns 2..6) per image.
- QA scoring averages the configured probe heads rather than all decoder heads; this is an SSD adaptation, not exact SparseVLM.
- v_hidden.pt is loaded once into CPU RAM when an image context is opened, outside per-request TTFT and SSD accounting; its per-request H2D and rater compute remain inside TTFT.
- Pairwise token Jaccard is the macro mean of per-decoder-layer Jaccards; the global layer-token Jaccard is also retained for every question pair.
- Buffered pread with POSIX_FADV_DONTNEED does not flush an SSD controller cache.
- 이 단계는 GQA pilot까지만 수행했다. MT-GQA-reconstructed, ConvBench, VisDial full rerun은 수행하지 않았으므로 conversational history가 있는 multi-turn 일반화 결론은 아직 내릴 수 없다.

## Hashes and provenance

- Run directory: `/home/dblab/hj/mllm_v2/runs/query_aware_baseline/gqa40_240_final`
- Results directory: `/home/dblab/hj/mllm_v2/results/query_aware_baseline/gqa40_240_final`
- GQA index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`
- Frozen full workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- Selected workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- Protection manifest file SHA256: `64fe572dbc2f0ddbcd9a3d386ad616ab123602a5fe0a2715036eaa54ce2c5ec4`
- Protection validation file SHA256: `d80970558b40d3fea1b6c78af0a2dd1b5cb16ed8514d54e50631ba0cd8a435c6`
- Protected entries canonical SHA256: `42ec8d56e113412640a56ac40aaa0fe12113aa3958e79c388ff9719a6efb4a88`

### Run evidence SHA256

| File | SHA256 |
|---|---|
| `config.json` | `dfe1629869f7ba6662556bb359d6c52e796c7b7b2c25893f6a89d0e75de6191c` |
| `raw.jsonl` | `cb39d32d5490a1356ffa307493f8613cd018f06394338c996302bb08035ce63b` |
| `persistence.jsonl` | `1145cc0e7f79c8515060cd9b97752accc9772a972acd9570320e1e6b8e4481e4` |
| `per_request.csv` | `8b00ae37adecaf15924455d90f1a2dd5d1c21287190145ed7a66d4f9c70a7a42` |
| `summary.json` | `a75abac0152dc6e3b69c3d6d125594d41f540a383faf84e268e483ed7805db7a` |
| `validation.json` | `dc9d58e41a1ec954c3c941ef18c07501dfc555457ff78f94edd5129412f74c00` |
| `selection.json` | `df65c39b5b961e2d448e718217e397b5b128139e088c8a1246bbe3992b4e532d` |
| `summary.csv` | `009cf2dec1fe7bea8d618c7d4f0d31c1f72fa3448abf7cc2c4060bc2cb9b1583` |
| `README.md` | `23003048df6b6566d3f0638de7ca18a6ed3e74ea26fe03e70d54e233cce09d9b` |

### Compact result bundle SHA256

| File | SHA256 |
|---|---|
| `summary.json` | `a75abac0152dc6e3b69c3d6d125594d41f540a383faf84e268e483ed7805db7a` |
| `validation.json` | `dc9d58e41a1ec954c3c941ef18c07501dfc555457ff78f94edd5129412f74c00` |
| `selection.json` | `df65c39b5b961e2d448e718217e397b5b128139e088c8a1246bbe3992b4e532d` |
| `summary.csv` | `009cf2dec1fe7bea8d618c7d4f0d31c1f72fa3448abf7cc2c4060bc2cb9b1583` |
| `README.md` | `23003048df6b6566d3f0638de7ca18a6ed3e74ea26fe03e70d54e233cce09d9b` |
| `run_artifacts.json` | `7dee962b7b22792d4972cf35aa465a37e41d2fdf268d50b7a2d1a0bbc2966b5c` |

### Report-time source SHA256

| File | SHA256 |
|---|---|
| `mmimpress/sparsevlm.py` | `7b24d592eb36bb5da11fe628d6012192ec1348d5680a8962640251ca73ce2e18` |
| `mmimpress/serve.py` | `0a5ec0ec4754c89e2f4d155339699b6ca30ccb7cea778af18c9bb3284eb88f7d` |
| `mmimpress/store.py` | `0732fc883b7cab8185c2bdf9349b00e0f316b12a2eb504d7106f3ab7793fd97b` |
| `mmimpress/piggyback.py` | `ed45e69ae17c25e64ccbc1ff9bd5afbf608119c6fb23d03e7a046600abcc0092` |
| `mmimpress/reorder.py` | `d6c5b40bafc18183cdb8793c7e87d916a2c4cca86751f6d57f83a5a9f4798b3f` |
| `scripts/49_eval_query_aware_baseline.py` | `ea6e3d0d3a50580e3c4cde8c578d377b4e7e92f2dcb6313610bf52b10f4c042d` |
| `scripts/50_protect_query_aware_artifacts.py` | `f0014e1e1147bafef7c8064e739ef811b080b60712a5e2231012ade922f4dd6b` |
| `scripts/51_report_query_aware_baseline.py` | `b0b6b19f603e10c39ab6de4b28f852a0bf2a38dae3fb1437f96cbf6ebeef1fff` |

## Reproduction

출력 경로는 반드시 새 경로를 사용한다.

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python -m unittest discover -s tests -p 'test_*.py'
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/50_protect_query_aware_artifacts.py --before --run-root runs/query_aware_repro --results-root results/query_aware_repro
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/49_eval_query_aware_baseline.py \
  --index data/index.json \
  --run-dir runs/query_aware_repro/gqa40_240 \
  --store-dir runs/query_aware_repro/gqa40_240_store \
  --results-dir results/query_aware_repro \
  --max-images 40 --skip 4 --questions 6 --seed 1234 \
  --max-new-tokens 16 \
  --expected-index-sha256 514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a \
  --expected-workload-sha256 97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66 \
  --expected-images 40 --expected-questions 240
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/50_protect_query_aware_artifacts.py --verify --run-root runs/query_aware_repro --results-root results/query_aware_repro
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/51_report_query_aware_baseline.py --run-dir runs/query_aware_repro/gqa40_240 --results-dir results/query_aware_repro
```

QUERY-AWARE BASELINE VALIDATED: YES
