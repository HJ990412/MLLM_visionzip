# IMPRESS 이식 경로 충실도 파일럿 (중요도 지표 × fallback V 읽기)

측정값만 적는다. 측정하지 않은 값은 "미실행"이다.

## 실행 조건

- gate: n=240 / images=40 / ratio=0.25 / alpha=0.6 / probe=3 / cold=True / importance_metric=sparsevlm / store_importance_metric=['sparsevlm'] / store=kvstore / workload_sha256=97afe02f924a49ca
- sparsevlm 실행: n=240 / images=40 / ratio=0.25 / alpha=0.6 / probe=3 / cold=True / importance_metric=sparsevlm / store_importance_metric=['sparsevlm'] / store=kvstore / workload_sha256=97afe02f924a49ca
- h2o 실행: n=240 / images=40 / ratio=0.25 / alpha=0.6 / probe=3 / cold=True / importance_metric=h2o / store_importance_metric=['h2o'] / store=runs/impress_fidelity_20261116T052410Z/kvstore_h2o / workload_sha256=97afe02f924a49ca

## 결과 표

FullLoad 대비 비율은 같은 실행(같은 store)의 FullLoad 기준이다. probe/K/V MB는 `os.pread`가 실제로 돌려준 바이트의 요청 평균이다. K·V chunk 비율은 layer당 읽은 chunk 수 / layer chunk 수의 평균(probe sidecar 제외). 점수 행 수는 중요도 점수에 쓰인 suffix query 행 수(sparsevlm: rater 수, h2o: suffix 토큰 수).

| 설정 | 정확도 | TTFT 평균(ms) | p50 | p95 | SSD MB | FullLoad 대비 | probe MB | K MB | V MB | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | hook(ms) | 점수 행 수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparsevlm+full | 60.00% | 841.5 | 855.6 | 961.8 | 928.1 | 79.7% | 54.6 | 436.7 | 436.7 | 15.8% | 0.3880 | 74.3% | 74.3% | 769.9 | 11.9 |
| sparsevlm+selected | 60.00% | 851.5 | 866.3 | 970.9 | 927.5 | 79.6% | 54.6 | 436.7 | 436.1 | 15.8% | 0.3880 | 74.3% | 74.1% | 778.5 | 11.9 |
| ReComp (sparsevlm 실행) | 62.50% | 507.7 | 513.0 | 593.1 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | - | - | - | - | - | - |
| FullLoad (sparsevlm 실행) | 62.08% | 779.7 | 776.4 | 903.3 | 1165.1 | 100.0% | 0.0 | 582.5 | 582.5 | - | - | 100.0% | 100.0% | 720.6 | - |
| h2o+full | 61.25% | 773.3 | 783.6 | 870.5 | 1010.4 | 86.7% | 54.6 | 477.9 | 477.9 | 33.3% | 0.3509 | 81.6% | 81.6% | 703.7 | 26.5 |
| h2o+selected | 61.25% | 784.2 | 794.5 | 888.3 | 1009.8 | 86.7% | 54.6 | 477.9 | 477.3 | 33.3% | 0.3509 | 81.6% | 81.4% | 713.0 | 26.5 |
| ReComp (h2o 실행) | 62.50% | 508.3 | 511.6 | 599.3 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | - | - | - | - | - | - |
| FullLoad (h2o 실행) | 62.08% | 700.5 | 703.1 | 812.8 | 1165.1 | 100.0% | 0.0 | 582.5 | 582.5 | - | - | 100.0% | 100.0% | 639.9 | - |

### gate 실행 (통과 기준 1 전용, 별도 실행)

| 설정 | 정확도 | TTFT 평균(ms) | p50 | p95 | SSD MB | FullLoad 대비 | probe MB | K MB | V MB | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | hook(ms) | 점수 행 수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparsevlm+full (gate) | 60.00% | 813.9 | 829.2 | 925.3 | 928.1 | 79.7% | 54.6 | 436.7 | 436.7 | 15.8% | 0.3880 | 74.3% | 74.3% | 738.8 | 11.9 |
| ReComp (gate) | 62.50% | 504.2 | 510.0 | 596.0 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | - | - | - | - | - | - |
| FullLoad (gate) | 62.08% | 746.3 | 743.2 | 872.8 | 1165.1 | 100.0% | 0.0 | 582.5 | 582.5 | - | - | 100.0% | 100.0% | 684.9 | - |

## fallback layer의 V 읽기량: 실측과 가상 하한

`fallback V 실측 MB`는 fallback layer에서 실제로 pread한 V 바이트다. `V token-exact 가상 하한 MB`는 head별 토큰 단위로 정확히 읽을 수 있다고 가정했을 때의 V 바이트로, **실제로 읽지 않은 참고값**이며 위 표의 SSD MB·V MB에 포함되지 않는다. 값은 요청당 평균이다.

| 설정 | fallback layer 수 | fallback K chunk 수 | fallback V chunk 수 | fallback V 실측 MB | V token-exact 가상 하한 MB (읽지 않음) |
|---|---:|---:|---:|---:|---:|
| sparsevlm+full | 5.05 | 175.4 | 175.4 | 91.0 | 23.9 |
| sparsevlm+selected | 5.05 | 175.4 | 173.5 | 90.3 | 23.9 |
| h2o+full | 10.66 | 366.9 | 366.9 | 190.2 | 50.0 |
| h2o+selected | 10.66 | 366.9 | 364.7 | 189.7 | 50.0 |

## 통과 기준 1 — sparsevlm+full이 `results/eval_b25.json`의 `sparsevlm`을 재현하는가

- 판정: **PASS** (허용 오차는 이 보고서가 정한 값: 정확도 ±2.0 pp, SSD MB ±1%, fallback ±1 pp)
- 정확도: 기준 60.00% → 측정 60.00% (차이 +0.00 pp)
- SSD MB: 기준 928.098 → 측정 928.098 (차이 +0.000 MB, +0.000%)
- fallback 비율: 기준 15.79% → 측정 15.79% (차이 +0.00 pp)
- FullLoad 대비: 기준 79.66% → 측정 79.66%
- 요청 단위(240개 공통): 답 동일 240, 정오 동일 240, SSD MB 동일 240, fallback 비율 동일 240, MB 최대 차이 0.000

## 통과 기준 2 — 같은 지표에서 full과 selected의 출력 토큰이 모두 같은가

- sparsevlm: **PASS** — 240/240개 요청의 `generated_token_ids` 동일 (다름 0, 한쪽 누락 0, token id 없음 0). fallback 비율 동일 240/240, K 바이트 동일 240/240, probe 바이트 동일 240/240, selected의 V 바이트가 full보다 큰 요청 0
- h2o: **PASS** — 240/240개 요청의 `generated_token_ids` 동일 (다름 0, 한쪽 누락 0, token id 없음 0). fallback 비율 동일 240/240, K 바이트 동일 240/240, probe 바이트 동일 240/240, selected의 V 바이트가 full보다 큰 요청 0

## 참고 예상치 대조 (목표가 아니라 이상 여부 판단용)

- sparsevlm 기준 FullLoad 대비 읽기량: full 79.66% → selected 79.61% (예상 범위 73.7~77.2%: 범위 밖)
- h2o+full의 FullLoad 대비 읽기량: 86.72%
- h2o+selected의 FullLoad 대비 읽기량: 86.68%

## 실행 간 기준선 일치

- ReComp (sparsevlm 실행 vs h2o 실행): 공통 240개 중 답 동일 240, token id 동일 240, 읽기 바이트 동일 240
- FullLoad (sparsevlm 실행 vs h2o 실행): 공통 240개 중 답 동일 238, token id 동일 238, 읽기 바이트 동일 240
- sparsevlm+full (gate vs sparsevlm 실행): 공통 240개 중 답 동일 240, token id 동일 240, 읽기 바이트 동일 240

## 절차와 해석 메모

### 실행 순서
1. gate: `04_eval.py --selectors sparsevlm` (kvstore) → 통과 기준 1 판정 후 나머지 진행.
2. h2o store 생성: `logs/build_h2o_store.sh`. 검증은 `logs/h2o_store_verification.txt`.
3. sparsevlm 실행(kvstore) → h2o 실행(h2o store). 각 실행은 문항마다
   ReComp → FullLoad → full → selected 순서로 측정한다(순서 고정).

### h2o store를 만든 방법
- 저장소에 있던 raster store(`runs/query_aware_baseline/gqa40_240_final_store/raster`)는
  turn-1 piggyback 캡처라서 layer 0 이후 K/V 바이트가 `kvstore`와 다르다(상대 차이 약 1%).
  그대로 쓰면 지표 차이와 KV 값 차이가 섞이므로 쓰지 않았다.
- 대신 `kvstore`를 `runs/impress_fidelity_20261116T052410Z/kvstore_h2o`로 복사하고
  `02_reorder.py --order raster --allow-composed-raster-target`로 raster로 되돌린 뒤
  `--order importance --calib-questions 4 --importance-metric h2o`를 적용했다.
  추가로 `--record-layout-provenance --rebuild-separator-sidecar`를 줬다(메타데이터와
  separator sidecar만 바뀌고 K/V 순서에는 영향 없음).
- `v_hidden.pt`와 `static.pt`는 복사하지 않았다. h2o 실행은 이 파일 없이 돌았다.
- 확인: 3개 이미지 × layer 0/13/31에서 K·V·probe_k가 순서를 되돌리면 `kvstore`와
  비트 단위로 같다. 두 store의 앞 576개 토큰(25%) 겹침은 평균 0.852(0.820~0.884).

### selected의 절감이 작은 이유 (직접 측정)
- `logs/fallback_union_sparsevlm.json`, `logs/fallback_union_h2o.json`: 10장 × 3문항(30요청),
  warm cache, 시간 측정 아님.
- fallback layer에서 head 하나가 유지하는 토큰은 26.3%지만, 32개 head의 합집합은
  sparsevlm에서 토큰의 평균 83.1%(64.9~91.4%)이고 chunk로는 98.9%(94.1~100%)를 덮는다.
  h2o에서는 chunk의 99.4%.
- 그래서 chunk 단위 V 읽기는 fallback layer당 평균 0.4개 미만의 chunk만 건너뛴다.
  요청당 V 절감은 sparsevlm 0.63 MB, h2o 0.54 MB(전체 읽기량의 0.05%p 안팎).

### 가상 하한을 대입한 계산값 (측정 아님)
- fallback layer의 V를 head별 토큰 단위로 정확히 읽을 수 있다고 가정하고
  `fallback_v_token_exact_bytes`를 대입하면 FullLoad 대비 sparsevlm 73.9%, h2o 74.7%가 된다.
  이 값은 실제로 읽은 양이 아니며, 현재 token-major 저장 형식으로는 얻을 수 없다.

### TTFT 해석 주의
- 같은 store·같은 바이트인 FullLoad의 TTFT 평균이 실행마다 746.3 / 779.7 / 700.5 ms로
  달랐다. 서로 다른 실행의 절대 TTFT는 비교하지 말고 같은 실행의 FullLoad 대비로 본다
  (sparsevlm+full 1.079×, h2o+full 1.104×).
- selected는 full보다 paired 평균 +10.0 ms(sparsevlm, 95% CI +1.8~+18.3),
  +11.0 ms(h2o, 95% CI +0.5~+21.4) 느렸다. selected가 항상 full 다음에 측정되므로
  순서 효과와 분리되지 않는다(`logs/paired_stats.txt`).

### 정확도 해석 주의
- h2o+full 147/240, sparsevlm+full 144/240. 엇갈린 문항은 5개(h2o만 정답 4, sparsevlm만 정답 1)로,
  240문항에서 지표 간 정확도 차이를 주장하기에는 부족하다.
- FullLoad는 두 store에서 답이 238/240 일치한다(정확도는 같음). key 순서가 달라 생기는
  부동소수점 합산 순서 차이로 보이며 따로 검증하지는 않았다.

### 기본 옵션 동등성
- `logs/default_equivalence_gpu.log`: 변경 전 `serve.py` 백업과 현재 코드를 같은 프로세스에서
  실제 요청 30건(impress 15, fullload 15)에 돌려 답, token id, pread 바이트·횟수, 로그, mask가
  모두 같음을 확인했다.
- gate 실행의 sparsevlm 240건은 `results/eval_b25.json`과 요청 단위로 답·MB·fallback이 같다.
- `kvstore/`와 `results/eval_b25.json`의 파일 크기·수정 시각은 전후로 같다
  (`logs/protected_manifest_before.txt`).

### 하지 않은 것
- MT-GQA 본실험 재실행, 재배치 통계 출처 변경, I/O와 연산의 중첩, 캐시 계층: 미실행(범위 밖).
- warm cache 측정, 측정 순서를 바꾼 반복 실행, 다른 비율(0.25 외)·다른 α: 미실행.
- h2o store에서 sparsevlm 지표(또는 그 반대)로 서빙하는 교차 조합: 미실행.
