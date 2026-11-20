# MT-GQA 다중 턴 — 공개 IMPRESS 구현의 로직을 이식한 IMPRESS arm (ReComp / FullLoad / IMPRESS / Ours)

측정값만 적는다. 측정하지 않은 값은 "미실행"이다. 계약: `docs/mt_gqa_impress_official_contract.md` (사본 `CONTRACT.md`).

## 설정

- 실행: `runs/mt_gqa_impress_official_20261117T052201Z` (요청 단위 원본), 이 디렉터리(`results/mt_gqa_impress_official_20261117T052201Z`)는 집계와 보고서.
- 환경: Python 3.10.20, torch 2.5.1+cu121, transformers 4.57.6 (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`). 직전 본실행과 같은 환경이다.
- 공개 구현: https://github.com/ISCS-ZJU/HyperInfer, 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e` (체크아웃 `/home/dblab/hj/_external_ref/HyperInfer`, 레포 밖, 읽기 전용 참조; 코드는 복사하지 않았다).
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd76…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2.
- 모집단·절차(직전 본실행과 동일): MT-GQA-reconstructed 398 images / 4,061 dialogues / 3턴, method-local generated history, strict normalized EM, 대화 단위 arm 순서 회전, 턴 1은 모든 방법이 픽셀부터 계산, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건, 저장 KV 방법은 비동기 읽기.
- IMPRESS(새 구현, `mmimpress/impress_official.py`): probe head [0, 1, 2], 유지 25%, 유사도 임계값 0.3(이상이면 합의), chunk 64 토큰. 중요도 = 턴 1에서 저장한 prefix 행 기여분 + 요청 때 계산한 suffix 행 기여분. fallback은 그 layer의 이미지 토큰 K·V 전부를 읽어 mask 없이 사용. 재배치는 첫 대화 턴 1 요청 하나의 통계(모든 head·모든 행 열 합)로 한 번. 다음 layer probe sidecar 선행 읽기.
- FullLoad, Ours, ReComp: 직전 본실행과 같은 코드·설정.

## 검증 단계

| 단계 | 결과 | 내용 |
|---|---|---|
| 1. 공개 구현과의 동등성 | PASS | 공개 함수 호출 테스트 8개 모두 실행·통과(skip 0), 신규 테스트 파일 전체 20개 |
| 2. prefix 행 기여분 (gate 4장, GPU) | PASS | n181355: prefix 행 일치, 재배치 통계 일치; n204894: prefix 행 일치, 재배치 통계 일치; n229548: prefix 행 일치, 재배치 통계 일치; n95369: prefix 행 일치, 재배치 통계 일치 |
| 3. 재배치 정합성: 유지 100% = FullLoad (gate) | FAIL | 147건 중 불일치 1건 |
| 4. 재현성: 직전 본실행과 토큰 일치 (gate) | PASS | ReComp: 147건 중 불일치 0건; FullLoad: 147건 중 불일치 0건; Ours: 147건 중 불일치 0건 |
| 4. 재현성 (DEV 전 요청) | PASS | ReComp: 1134건 중 불일치 0건; FullLoad: 1134건 중 불일치 0건; Ours: 1134건 중 불일치 0건; IMPRESS (이전 구현): 1134건 중 불일치 0건 (기준 외) |
| 5. 기존 테스트 | PASS | transformers 4.57.6: 736개 실행, 실패 3개(모두 변경 전부터의 환경 실패), 새 실패 0개; transformers 5.13.1: 754개 실행, 실패 2개(모두 변경 전부터의 환경 실패), 새 실패 0개 |
| 6. DEV 40장 실행·감사 | PASS | 5670/5670건, 40장 |
| 추가. 새 arm 비동기 = 동기 (gate) | PASS | 147건 중 불일치 0건 |
| gate phase 실행·감사 | PASS | 882/882건, 4장 |
| 7. 본실행 진입 | FAIL | 1–6 모두 PASS일 때만 실행 |

2번의 참고값(기준 아님): 같은 이미지를 다른 질문의 요청 안에서 다시 캡처한 prefix 행 기여분과 저장값의 차이는 최대 절댓값 2.39, 평균 절댓값 0.00226, 저장값이 1e-3 이상인 원소의 최대 상대 차이 1.13; prefix 행만으로 뽑은 head별 상위 25% 집합의 Jaccard는 평균 0.9805, 최소 0.9305 (이미지 4장, layer × probe head 전체). 원소 단위로 완전히 같은 이미지는 0장이다.

## 본실행 (398장)

미실행.

## DEV 40장 — 새 구현과 이전 구현 (같은 실행, T2–T3 followup)

DEV 40장의 MT-GQA 대화 378개, arm당 followup 756건. 두 구현은 같은 실행에서 같은 이미지·대화를 받았고 각자 자기 store(턴 1에서 생성)를 읽었다. 이전 구현은 직전 본실행의 IMPRESS arm 그대로다.

| 구현 | 정답률 | TTFT 평균(ms) | p50 | p95 | SSD MB (probe / K / V) | read 호출 수 | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | GPU 읽기 대기(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 새 구현 | 61.38% | 738.4 | 736.8 | 842.2 | 1042.4 (55.6 / 493.4 / 493.4) | 123.6 | 52.4% | 0.3398 | 84.2% | 84.2% | 394.8 |
| 이전 구현 | 68.52% | 709.6 | 713.9 | 814.2 | 1013.9 (54.7 / 479.8 / 479.4) | 140.7 | 27.5% | 0.3642 | 81.7% | 81.6% | 366.2 |

- 새 구현 − 이전 구현 (이미지 단위 paired bootstrap 95% CI, 40장, 756건): 정답률 -7.14 pp [-10.49, -3.97], TTFT +28.8 ms [+18.3, +39.9], 읽기량 +28.5 MB [+17.2, +40.8].
- 두 구현의 fallback은 뜻이 다르다. 새 구현: 그 layer의 K·V 전부를 읽고 mask 없이 사용. 이전 구현: K 전부를 읽고 head별로 25%를 골라 mask, V는 선택된 chunk만.
- 임계값도 다르다. 새 구현: Jaccard ≥ 0.3이면 합의. 이전 구현: Jaccard > (r/(2−r))^0.6 ≈ 0.311이면 합의.

같은 DEV 실행의 다른 arm(참고):

| 방법 | 정답률 | TTFT 평균(ms) | p50 | p95 | SSD MB/followup |
|---|---:|---:|---:|---:|---:|
| ReComp | 68.92% | 528.9 | 529.5 | 579.5 | 0.0 |
| FullLoad | 69.44% | 481.7 | 489.1 | 557.2 | 1166.9 |
| IMPRESS | 61.38% | 738.4 | 736.8 | 842.2 | 1042.4 |
| IMPRESS (이전 구현) | 68.52% | 709.6 | 713.9 | 814.2 | 1013.9 |
| Ours | 68.52% | 149.2 | 149.8 | 167.7 | 306.8 |

## IMPRESS 내부 통계 (새 구현)

### DEV 40장 — followup 756건

| 항목 | 값 |
|---|---:|
| fallback 비율 (layer·요청 평균) | 52.4% |
| fallback layer가 하나라도 있는 요청의 비율 | 100.0% |
| 평균 Jaccard (probe head 3개 쌍 평균) | 0.3398 |
| K chunk 비율 (실측) | 84.2% |
| V chunk 비율 (실측) | 84.2% |
| 합의 layer만: K·V chunk 비율 (실측) | 66.8% |
| 중요도 합에서 prefix 행이 차지하는 비중 | 98.7% |
| 재배치 순서 상위 25%와 head별 상위 k의 겹침 (모든 layer) | 59.8% |
| 재배치 순서 상위 25%와 합의 선택의 겹침 (합의 layer만) | 79.3% |
| raster 순서였다면 K·V chunk 비율 (**계산값**, 읽지 않음) — 전체 / 합의 layer만 | 100.0% / 100.0% |
| separator도 중요도 순서에 놓았다면 K·V chunk 비율 (**계산값**, 읽지 않음) — 전체 / 합의 layer만 | 86.2% / 71.0% |
| GPU가 읽기를 기다린 시간 (요청 평균) | 394.8 ms |
| 그중 probe sidecar를 얻느라 hook이 막힌 시간 | 4.5 ms |
| 선행 읽기한 probe sidecar 31개 중 도착 시 이미 읽혀 있던 수 (요청 평균) | 29.8 |

chunk 비율은 layer당 읽은 chunk 수 / layer chunk 수의 평균이다(probe sidecar 제외; fallback layer는 100%). 계산값은 요청의 layer별 keep mask(합의: 선택된 토큰 + separator)를 다른 저장 순서의 위치로 옮겨 64 토큰 chunk로 센 것이며 실제로 읽은 값이 아니다.

layer별:

| layer | fallback | Jaccard | K·V chunk 실측 | prefix 행 비중 | 상위 25% 겹침(head별) | 상위 25% 겹침(합의 선택) | raster 계산값 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 0.746 | 31.6% | 97.5% | 90.7% | 96.4% | 100.0% |
| 1 | 0.0% | 0.805 | 34.5% | 99.0% | 92.8% | 95.8% | 100.0% |
| 2 | 0.0% | 0.655 | 43.4% | 99.2% | 84.3% | 89.6% | 100.0% |
| 3 | 0.0% | 0.367 | 65.3% | 98.3% | 64.3% | 79.0% | 100.0% |
| 4 | 75.8% | 0.278 | 89.5% | 99.1% | 62.5% | 84.4% | 100.0% |
| 5 | 25.8% | 0.316 | 71.6% | 99.6% | 63.8% | 85.4% | 100.0% |
| 6 | 0.0% | 0.498 | 72.6% | 98.4% | 69.5% | 78.2% | 100.0% |
| 7 | 67.2% | 0.287 | 88.7% | 99.0% | 59.0% | 78.3% | 100.0% |
| 8 | 44.2% | 0.315 | 84.8% | 99.3% | 57.5% | 73.7% | 100.0% |
| 9 | 70.8% | 0.280 | 93.7% | 99.0% | 54.3% | 73.5% | 100.0% |
| 10 | 100.0% | 0.254 | 100.0% | 99.3% | 57.6% | — | 100.0% |
| 11 | 100.0% | 0.181 | 100.0% | 99.2% | 46.9% | — | 100.0% |
| 12 | 100.0% | 0.214 | 100.0% | 99.1% | 52.1% | — | 100.0% |
| 13 | 66.8% | 0.294 | 95.1% | 99.4% | 55.3% | 71.9% | 100.0% |
| 14 | 100.0% | 0.234 | 100.0% | 99.4% | 52.3% | — | 100.0% |
| 15 | 97.4% | 0.240 | 99.5% | 99.0% | 52.5% | 74.1% | 100.0% |
| 16 | 79.6% | 0.278 | 95.4% | 98.7% | 55.1% | 73.6% | 100.0% |
| 17 | 83.1% | 0.273 | 95.3% | 98.8% | 55.6% | 75.1% | 100.0% |
| 18 | 96.3% | 0.244 | 98.8% | 98.7% | 55.5% | 82.3% | 100.0% |
| 19 | 100.0% | 0.198 | 100.0% | 99.4% | 43.1% | — | 100.0% |
| 20 | 40.5% | 0.307 | 76.1% | 98.2% | 62.7% | 82.7% | 100.0% |
| 21 | 58.2% | 0.292 | 94.9% | 97.1% | 51.3% | 64.4% | 100.0% |
| 22 | 2.6% | 0.389 | 94.0% | 98.9% | 52.3% | 65.9% | 100.0% |
| 23 | 100.0% | 0.237 | 100.0% | 99.2% | 54.9% | — | 100.0% |
| 24 | 0.0% | 0.409 | 78.0% | 95.7% | 58.7% | 71.4% | 100.0% |
| 25 | 0.0% | 0.410 | 74.8% | 95.6% | 62.0% | 73.9% | 100.0% |
| 26 | 0.0% | 0.382 | 75.0% | 98.5% | 61.8% | 75.6% | 100.0% |
| 27 | 79.6% | 0.276 | 97.1% | 98.8% | 53.4% | 69.0% | 100.0% |
| 28 | 3.6% | 0.361 | 76.4% | 98.9% | 60.3% | 78.4% | 100.0% |
| 29 | 0.0% | 0.369 | 72.1% | 98.7% | 62.3% | 77.8% | 100.0% |
| 30 | 100.0% | 0.217 | 100.0% | 99.4% | 53.7% | — | 100.0% |
| 31 | 84.7% | 0.272 | 96.3% | 98.7% | 57.1% | 78.0% | 100.0% |

## 저장·턴 1 비용

출처: DEV 40장. store는 이미지당 하나(첫 대화 턴 1에서 생성). 시간은 이미지 평균(ms).

| 방법 | store 크기 평균(MB) | 최대(MB) | K·V | probe sidecar | separator sidecar | v_hidden | 기타(meta·system KV 등) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 1260.6 | 1658.9 | 1165.1 | 54.6 | 20.0 | 18.2 | 2.7 |
| IMPRESS | 1262.0 | 1660.8 | 1165.1 | 55.5 | 20.0 | 18.2 | 3.3 |
| IMPRESS (이전 구현) | 1261.2 | 1659.7 | 1165.1 | 54.6 | 20.0 | 18.2 | 3.3 |
| Ours | 309.0 | 405.4 | 306.3 | 0.0 | 0.0 | 0.0 | 2.7 |

- IMPRESS의 probe sidecar에는 prefix 행 기여분 0.85 MB(이미지 평균)가 포함된다. layer별 재배치 순서는 `meta.json`(기타)에 있다. FullLoad store의 probe sidecar·separator sidecar·`v_hidden.pt`는 같은 writer가 쓰지만 FullLoad는 읽지 않는다. IMPRESS는 K·V와 probe sidecar만 읽는다.

| 방법 | 저장 전체(쓰기+무결성 hash) | writer(쓰기+fsync) | 무결성 hash | activation |
|---|---:|---:|---:|---:|
| FullLoad | 1721 | 1062 | 613 | 1214 |
| IMPRESS | 2706 | 1064 | 612 | 1224 |
| IMPRESS (이전 구현) | 3070 | 1061 | 613 | 1214 |
| Ours | 718 | 531 | 151 | 185 |

IMPRESS의 턴 1 쪽 추가 비용(이미지당 평균, followup TTFT에는 들어 있지 않다):

| 항목 | 시간(ms) | 어디서 |
|---|---:|---|
| prefix 행 기여분 + 재배치 통계 캡처 (32 layer) | 17.0 | 턴 1 요청 안 (그 요청의 TTFT에 포함) |
| 캡처 결과를 host로 옮기기 | 0.3 | 요청 밖 |
| 증거 파일(.npz) 저장 | 1.1 | 요청 밖 (실험 기록용) |
| 재배치 순서 계산 (32 layer 정렬) | 59.9 | 요청 밖 |
| 재배치 쓰기 전체 (K / V / probe sidecar 읽기+치환+쓰기, meta, separator sidecar) | 980.8 | 요청 밖 |
|   그중 K / V / probe sidecar | 407 / 411 / 43 |  |
|   그중 순수 쓰기 | 602 |  |
|   그중 separator sidecar 재작성 | 34 |  |

- 캡처가 걸린 턴 1 요청의 TTFT는 평균 545.7 ms, 같은 대화의 ReComp 턴 1은 평균 519.1 ms다(캡처 요청에는 KV 캡처 등 다른 저장용 캡처도 함께 걸려 있다). 캡처된 행 수는 prefix 길이 + 그 턴 1 suffix 길이(suffix 평균 34.6 토큰)다.
- "저장 전체"는 writer + 이미지별 store 전체 SHA256 1회(수명주기 공통) + (IMPRESS는) 재배치 쓰기를 포함한 벽시계 시간이다. page cache에 쓰는 시간과 fsync 시간이 섞여 있으므로 디스크 순수 쓰기 속도로 해석하지 않는다.

### 3턴 세션 시간 (직전 보고서와 같은 방식)

세션 = 한 대화의 3턴 요청 E2E 합 + store 준비(저장 전체 + activation). shared: 이미지의 첫 대화에만 준비 시간을 더함(실측 흐름). standalone: 모든 대화에 더함(DERIVED).

| 방법 | 요청만 평균(ms) | shared 평균 / p50 / p95 (ms) | standalone 평균 / p50 / p95 (ms) |
|---|---:|---:|---:|
| ReComp | 1661.0 | 1661.0 / 1681.2 / 1830.6 | 1661.0 / 1681.2 / 1830.6 |
| FullLoad | 1568.2 | 1878.7 / 1615.4 / 4714.1 | 4507.4 / 4563.7 / 5066.3 |
| IMPRESS | 2084.7 | 2500.5 / 2110.5 / 6227.5 | 6016.4 / 6004.3 / 6804.5 |
| IMPRESS (이전 구현) | 2026.3 | 2479.6 / 2067.5 / 6324.8 | 6309.9 / 6429.9 / 6949.9 |
| Ours | 902.5 | 998.0 / 921.4 / 1835.7 | 1806.8 / 1828.5 / 1963.0 |

- 디스크 최대 사용량: 이미지 하나의 scratch store 묶음 최대 5.38 GB (평균 4.09 GB, 40장). 실행 중 관측된 최소 여유 공간 437.5 GiB. 이미지 commit 뒤 scratch는 지운다.

## 공개 구현의 함수와 새 구현의 대응

| 공개 구현 (`h2o_flexgen/flexgen/`) | 새 구현 (`mmimpress/impress_official.py`) |
|---|---|
| `sele_tokenid` / `llama_sele_tokenid` (probe head의 attention: 전체 행 × prefix key + suffix key, causal softmax) | `causal_attention`, `column_importance`; 요청 때는 `suffix_row_importance_from_qk` + 저장된 prefix 행 기여분; 턴 1 캡처 `Turn1AttentionStats` |
| `_sele_important_tokens` (열 합, head별 상위 `ceil(n·percent/100)`, 유사도 판정 `sim >= sim_thred`, 득표 합의, fallback = prefix 전체) | `keep_count`, `head_topk`, `vote_consensus`, `select_tokens` |
| `cal_jaccard_sim` (head 쌍의 Jaccard 평균) | `mean_pairwise_jaccard` |
| `mha_with_filled_selected_prefixkv` (선택 안 된 토큰의 열에 `finfo.min`) | `OfficialImpressSelector._run`의 keep mask → 기존 `bias_from_keep` / `BIAS` (additive mask) |
| `mha_with_sele_percent_prefixkv(generate_mapping_list=True)` (모든 head·모든 행 열 합의 내림차순, `reverse_indices`) | `Turn1AttentionStats`의 재배치 통계, `descending_order`, `reverse_index`, `layer_order` |
| `my_pcache_fast.py` `PrefixKVLayer` (`KV_Division`: K·V 분리, 64 토큰 chunk, `head_token`, `reorder`) | `write_official_layout`(K·V·probe 파일을 같은 순서로), `read_probe_block`, 기존 `ChunkReader.read_chunks` / `read_full` |
| `flex_opt_fast_pcache.py`의 sele-load 분기 (probe key 읽기 → 선택 → 선택된 K·V 읽기 → 계산) | `OfficialImpressSelector`(layer pre-hook), `request_official` |

## 원본과 다른 점

"방향"은 IMPRESS(baseline)의 TTFT·정답률에 유리한지, 불리한지, 중립인지다. 방향은 설계상의 판단이며 항목별 효과를 따로 재지는 않았다(미실행).

| # | 항목 | 공개 구현 | 새 구현 | 방향 |
|---|---|---|---|---|
| 1 | 선택 대상 | prefix 토큰 전부 | 이미지 토큰만. 시스템 프롬프트와 separator는 항상 유지, 예산 밖 | 정답률에 유리 / 읽기량에 소폭 불리 |
| 2 | prefix 행 기여분 | 정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태) | 턴 1(mask 없음)의 값을 저장해 요청 때 더함 | 유리 (요청 때 prefix 행 연산 없음) |
| 3 | 캐시 계층 | GPU·DRAM·SSD 다층, CKLFU | 없음. 모든 방법이 cold SSD | 불리 (모든 방법 공통 조건) |
| 4 | 다음 layer probe 선행 읽기 | IMPRESS 설정은 `--no-prefetch` | 켬 | 유리 |
| 5 | separator의 저장 위치 | 해당 없음 | 저장 순서 맨 앞에 모음(항상 읽으므로 chunk 1개에 담김) | 유리 |
| 6 | 득표 동점 | 순서 미정의(unstable argsort) | 저장 위치가 앞선 토큰 우선 | 유리 또는 중립 |
| 7 | 재배치 통계의 추론 상태 | head별 25% mask 추론(`--full-load --sele-inf`) | mask 없는 턴 1 추론 | 중립 |
| 8 | 재배치 적용 | `key[reorder]` (token→rank 인덱스로 gather; 내용이 어긋남) | 위치 r = rank r 토큰, K·V·probe 정렬 유지 | 중립 (접근 위치는 같다) |
| 9 | 중요도 수치 정밀도 | float16 softmax, float16 합 | BF16 확률 + float32 합 / float32 softmax | 중립 |
| 10 | Fallback 읽기 단위 | chunk 파일별 | layer의 K 파일 1회, V 파일 1회 pread | 유리 (호출 수 적음, 바이트 같음) |
| 11 | 선택 chunk 읽기 단위 | chunk 파일별 | 인접한 chunk는 pread 하나로 합침 | 유리 |
| 12 | prefix 행 기여분 읽기 | 없음 | probe sidecar 끝 12·N bytes/layer를 같은 read로 | 불리 (probe 읽기 +1.6%) |
| 13 | mask | 정확도 경로만(성능 경로는 0으로 채운 K/V, mask 없음, 답을 내지 않음) | mask | 중립 |
| 14 | 모델·생성 | OPT(FP16), 1토큰 logits | LLaVA-NeXT(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정 | 중립 |

## 기존 산출물 보호

- `results/`, `runs/`(이 run 제외), `kvstore*`의 기존 파일 2,284,290개 (경로, 크기, mtime) 대조: **PASS**.

## 3번 실패의 내용과 원인

- 기준(사전 고정): gate 4장의 followup·턴 1 요청 147건 전부에서 `impress_official_r100`(새 경로, 유지 100%, 재배치된 store)의 생성 토큰열이 FullLoad와 같아야 한다. 결과: 146건 일치, 1건 불일치 → **FAIL**. 이 기준에 따라 본실행(398장)은 시작하지 않았다.
- 불일치 1건: 이미지 `n181355`, 대화 `mtgqa_000018`, 턴 2. FullLoad `Man` (토큰 2315), 새 경로 `Girl` (토큰 18620).
- 진단(`diagnostics/step3_mismatch_diagnostic.json`, 같은 이미지·요청을 GPU에서 다시 실행):
  - 새 경로가 읽어 올린 prefix K·V는 모든 layer에서 FullLoad가 읽은 값과 비트 단위로 같다(저장 순서만 다르다).
  - FullLoad의 첫 토큰 logit은 `Man` 21.25, `Girl` 21.25로 **정확히 같다**. argmax는 id가 작은 `Man`을 고른다.
  - key 순서가 재배치된 상태에서는 `Girl` 21.375, `Man` 21.25다(전체 logit 최대 차이 0.21875).
  - 새 선택 경로를 쓰지 않고 FullLoad의 loader(`LayerSelector(mode="fullload")`)로 재배치된 store를 읽어도 같은 `Girl`이 나온다.
- 해석: 읽은 값과 mask에는 차이가 없고, attention의 key 순서가 달라져 BF16 덧셈의 반올림이 한 단계 달라졌으며 그 차이가 동점이던 두 logit의 순서를 바꿨다. 선택·읽기 로직의 오류로 볼 증거는 없다. 다만 사전에 고정한 기준은 "전부 일치"였으므로 판정은 FAIL로 둔다. 기준을 사후에 바꾸지 않았다.
- 같은 종류의 차이는 재배치된 store를 읽는 모든 방법에 있을 수 있다(저장 순서가 raster가 아니면 key 순서가 달라진다).

## DEV 정답률에 대한 진단 (기준 외, 실행 뒤 추가)

DEV에서 새 구현의 정답률이 이전 구현보다 7.14 pp 낮았다(위 표). 구현 오류인지 확인하려고 DEV 이미지 6장(디렉터리 이름순 앞 6장, 결과를 보기 전에 정함)의 followup 106건을 다시 실행했다. 스크립트와 결과: `diagnostics/dev_accuracy_diagnostic.py`, `diagnostics/dev_accuracy_diagnostic.json`. 이 진단은 고정된 실행의 일부가 아니다.

| 확인 | 결과 |
|---|---|
| 새 arm을 다시 실행한 생성 토큰열 = 실행 기록 | 106 / 106 |
| 독립 기준(raster store를 FullLoad loader로 전부 읽고, 새 arm이 고른 토큰 집합만 mask로 남김)의 생성 토큰열 = 새 arm | 106 / 106 (첫 토큰 logit 최대 차이 0.3125) |
| layer별 합의/fallback 판정: 독립 계산(저장된 prefix 행 + 모델의 q·k로 직접 계산한 suffix 행에 선택 함수를 적용) = 새 arm | 3,391 / 3,392 (유사도 차이 최대 0.003) |
| probe head별 상위 25% 집합: 독립 계산 = 새 arm | 9,733 / 10,176 완전 일치, 나머지의 Jaccard 최소 0.991 |
| 합의 선택이 독립 계산의 득표수 조건을 만족 (선택 안 된 토큰의 득표가 선택된 토큰보다 많지 않음) | 1,636 / 1,683 |

완전히 일치하지 않은 경우는 상위 25% 경계의 토큰 몇 개가 다른 것이다(독립 계산은 float32 행렬곱, 새 arm은 모델과 같은 BF16 행렬곱). 득표수가 같은 토큰 사이의 순서는 공개 구현에서 정해져 있지 않으므로 선택 집합 자체의 일치는 기준으로 쓰지 않았다(합의 선택의 7–20%가 최저 득표 구간에서 뽑힌다).

이 106건에서 정답률은 새 구현 72.6%, 이전 구현 80.2%, FullLoad 80.2%로 DEV 전체와 같은 방향이다.

새 구현이 무엇을 고르는지(같은 106건):

- 같은 이미지에서 질문이 달라도 합의 선택은 거의 같다. 같은 이미지·같은 layer의 서로 다른 요청 사이 Jaccard 평균 0.994.
- 저장된 prefix 행 기여분만으로 고른 집합과 요청 때의 합의 선택의 겹침은 layer별 84–99%다. 중요도 합에서 prefix 행이 차지하는 비중이 98.7%이기 때문이다(prefix 행 약 2,150개, suffix 행 수십 개).
- layer 0과 1에서는 선택된 토큰의 95%와 90%가 저해상도 기본 이미지(이미지 토큰 2,112개 중 앞 576개)에 있다. causal attention에서 앞쪽 토큰은 더 많은 행에서 보이므로 열 합이 크다.
- layer 0–3은 모든 요청에서 합의였다(fallback 0%). 즉 가장 앞 layer들에서 질문과 무관하게 고해상도 토큰 대부분이 가려진다.

판단: 정답률 하락은 mask·읽기 오류가 아니라 "모든 행의 열 합"이라는 공개 구현의 중요도 정의를 이미지 prefix에 적용한 결과로 보인다. 이전 구현은 suffix 행만 썼으므로 선택이 질문에 따라 달랐다. suffix 행만 쓰는 변형을 같은 조건에서 실행해 비교하지는 않았다(미실행).

## DEV에서 예상과 달랐던 점

- 새 구현의 정답률은 61.38%로 FullLoad 69.44%, 이전 구현 68.52%보다 낮다. 새 구현과 FullLoad의 생성 토큰열이 같은 요청은 756건 중 585건(이전 구현은 703건)이다.
- 새 구현의 fallback 비율은 52.4%로 이전 구현(27.5%)의 약 두 배다. fallback이 layer의 K·V를 모두 읽으므로 읽기량은 FullLoad의 89.3%(1,042.4 / 1,166.9 MB)다.
- 새 구현의 TTFT 평균 738.4 ms는 FullLoad 481.7 ms의 1.53배, ReComp 528.9 ms의 1.40배다. GPU가 읽기를 기다린 시간이 요청당 394.8 ms다. 이전 구현도 같은 순서였다(709.6 ms).
- 재배치가 없었다면(raster 순서) 합의 layer에서도 chunk를 100% 읽었을 것이다(계산값). 실측은 합의 layer 66.8%다.
- 턴 1의 통계로 정한 순서의 상위 25%와 요청 때 합의 선택의 겹침은 79.3%다.

## 공개 구현을 읽고 확인한 것 중 작업 설명과 달랐던 점

- mask(`finfo.min`)는 정확도 경로(`figure15.sh` → `overall_acc.sh` → `flex_opt.py`)에만 있다. 성능 경로(`lm_performance_fast_pcache.sh`: `--suffix-comp --fill-keys-zero --no-prefetch --reorder`)는 읽지 않은 K·V를 0으로 채우고 mask를 쓰지 않으며 출력으로 `torch.empty`를 돌려준다. 새 구현은 한 경로에서 정답률과 시간을 함께 재야 하므로 mask를 쓴다.
- `PrefixKVLayer`는 재배치를 `key[reorder]`로 적용하는데 `reorder`는 token → rank 인덱스다. 접근 위치는 의도대로지만 내용은 어긋난다. 새 구현은 위치 r에 rank r 토큰을 둔다.
- 재배치 통계는 `--full-load --sele-inf`(head별 25% mask) 추론에서 나오고, 같은 prefix에 요청이 여럿이면 마지막 요청의 값이 남는다. 새 구현은 작업 지시대로 첫 대화 턴 1 요청 하나(mask 없음)의 값을 쓴다.
- 정확도 경로는 prefix 행을 요청마다 다시 계산한다(선택된 KV만 보이는 상태). 새 구현은 작업 지시대로 턴 1의 값을 저장해 쓴다.
- 중요도 계산은 float16 softmax와 float16 합이다. 새 구현은 BF16 확률과 float32 합(턴 1), float32 softmax(요청 때)다.
- 성능 설정은 `--no-prefetch`다. 새 구현은 작업 지시대로 다음 layer probe 선행 읽기를 켰다.

## 작업 지시에 없어서 직접 정한 것

- separator(행 구분 토큰)를 저장 순서 맨 앞에 모았다. 항상 읽는 토큰이므로 chunk 1개에 담긴다. IMPRESS에 유리한 결정이다. 중요도 순서에 두었다면 chunk 비율은 84.2% → 86.2%(합의 layer 66.8% → 71.0%)였을 것이다(계산값).
- 득표수가 같을 때 저장 위치가 앞선 토큰을 고른다. 공개 구현은 순서가 정해져 있지 않다. 재배치 순서가 앞선 토큰이 뽑히므로 읽는 chunk가 줄어드는 쪽이다.
- prefix 행 기여분을 probe sidecar 끝에 붙여 같은 read로 읽는다(probe 읽기 +1.6%).
- 검증용 arm `impress_official`(동기 읽기), `impress_official_r100`(유지 100%)을 gate phase에만 두었다. 어떤 표에도 넣지 않았다.

## 하지 못한 것

- 본실행(398장, 네 arm): 미실행. 3번이 FAIL이다.
- 본실행 표, paired bootstrap(Ours − IMPRESS 등), 본실행의 IMPRESS 내부 통계와 저장 비용: 미실행. 위 DEV 값은 40장의 값이며 본실행을 대신하지 않는다.
- `PrefixKVLayer`를 직접 호출하는 동등성 테스트: 하지 않았다. 생성자가 GPU chunk 캐시를 만들고 프로세스 공용 디렉터리에 `.npy` chunk 파일을 쓰며, 해당 커밋에서 재배치를 token → rank 인덱스로 적용한다. 대신 저장 테스트로 의도한 동작(위치 r = rank r 토큰, K·V·probe 정렬 유지)을 고정했다.
- "원본과 다른 점" 각 항목의 효과를 따로 재는 실험: 미실행.
- 커밋·푸시: 하지 않았다.
