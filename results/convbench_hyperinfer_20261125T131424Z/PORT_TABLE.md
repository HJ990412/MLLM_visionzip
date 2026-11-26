# HyperInfer 이식: 원본 대조표 (0단계)

작성 2026-11-24, 개정 2026-11-25. 구현 전에 논문과 공개 코드를 직접 읽고 적었다.

**2026-11-25 개정.** prefetch를 멈추는 방식을 논문 IV-C와 Fig. 10(c)의 서술(계산이 끝나면 새 읽기 요청을 내지 않는다)로 바꿨다. 바뀐 행은 c, f, i, k이고 t, u를 더했다. 2026-11-24 실행(`runs/mt_gqa_hyperinfer_20261124T171039Z`)은 공개 코드의 시간 예산 방식이었고, 그때의 표는 그 실행의 `PORT_TABLE.md`에 그대로 있다. "우리 이식본" 열은 지금의 이식본을 적고, 이전 이식본과 다른 곳은 "(이전: …)"로 적는다.

- 논문: Zhang et al., "HyperInfer: Accelerating Large Language Model Inference via Importance-Informed Prefix-KV Caching and Prefetching", IEEE TC 2026 (`papers/HyperInfer.md`).
- 공개 코드: https://github.com/ISCS-ZJU/HyperInfer, commit `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`. 체크아웃 위치는 프로젝트 밖 `/home/dblab/hj/_external_ref/HyperInfer` (작업 트리 변경 없음). LICENSE 파일이 없어 코드는 복사하지 않고 동작만 대조한다.
- 우리 이식본: `mmimpress/hyperinfer_v2.py` (이전: `mmimpress/hyperinfer.py`, 수정하지 않고 둔다). IMPRESS 이식본(`mmimpress/impress_official.py`, 두 모델 공용) 위에 prefetch만 더한다. IMPRESS 이식본의 선택, fallback, 저장 형식, 재배치는 바꾸지 않는다.

파일 약칭 (모두 `h2o_flexgen/flexgen/` 아래):
**F** = `flex_opt_fast_pcache.py`, **P** = `my_pcache_fast.py`, **B** = `pytorch_backend.py`, **S** = `scripts/lm_performance_fast_pcache.sh`, **R** = `README-ASYNC-PREFETCH-REVIEW.md`.

방향은 baseline(HyperInfer)의 TTFT 기준이다. "유리" = 우리 이식본이 원본보다 HyperInfer에 유리, "불리" = 불리, "중립" = 영향 없음 또는 방향을 말할 수 없음.

## 1. 요청하신 항목 a–j

| # | 항목 | 논문의 서술 (절) | 공개 코드의 동작 (파일·줄) | 우리 이식본 | 차이와 방향 |
|---|---|---|---|---|---|
| a | 예측 집합 | 현재 layer의 중요 토큰 집합으로 다음 layer의 것을 근사한다 (IV-C Design) | layer l의 선택 `sele_tokenids` 전체를 복사해 l+1의 prefetch 입력으로 쓴다 (F 1100–1108, `prefetch_tokenids = sele_tokenids.clone()`). 고정 비율 방식(`solid_ratio_prefetch`)은 기본값이 꺼져 있고 실행 설정에서도 쓰지 않는다 (S 211–229) | layer l의 사용 집합 G_l(선택된 이미지 토큰 + 항상 유지하는 separator) 전체를 l+1의 예측으로 쓴다 | 같음. separator는 MLLM 적응(IMPRESS 이식본과 같은 규칙) — 중립 |
| b | 읽는 단위 | "중요도가 높은 토큰부터" prefetch buffer에 읽는다 (IV-C) | 예측 토큰을 layer l+1의 재배치 순서로 옮기고(P 1671–1672) `// chunk_size`로 chunk 번호를 구해 `torch.unique`로 오름차순 정렬한다(P 1674–1675). 남은 chunk마다 K chunk, V chunk를 통째로 읽는다(P 1481–1531). prefetch된 토큰 집합은 읽은 chunk의 모든 토큰이다(P 1505–1509, 원래 토큰 번호로 되돌림 P 1711–1713) | 같은 계산. 오름차순으로 chunk마다 K chunk → V chunk를 읽는다(읽기 하나 = pread 1회 = K chunk 하나 또는 V chunk 하나). 멈춘 시점에 따라 마지막 chunk는 K만 읽혔을 수 있다. P_{l+1} = K와 V가 모두 읽힌 chunk의 모든 토큰이고, K만 읽힌 chunk는 그 K를 그대로 쓰고 V만 동기로 읽는다 | 같음. (재배치 순서가 중요도 내림차순이므로 오름차순 chunk = 중요도 높은 순.) K만 읽힌 chunk를 살려 쓰는 것은 유리 |
| c | 자르기 | GPU가 현재 layer 계산을 끝내는 즉시 새 요청을 멈춘다 (IV-C, reactive truncation) | GPU 상태를 보지 않는다. 직전 layer에서 잰 `compute_time + per_layer_dt_sele_head`를 예산으로 두고(F 1263–1264) chunk당 예상 시간 `AVG_CHUNK_TIME`(K와 V 각각, 저장 계층별 상수; disk 0.40818 ms, P 10–13)을 빼 가며 예산을 넘는 첫 chunk 앞에서 자른다(P 1679–1698). 예산이 0이면 예측 chunk의 앞 절반(`chunk_ids[:n // 2]`)을 읽는다(P 1699–1701). **정정:** 예산 변수는 모델 객체에 있고(F 1585) 요청이 바뀌어도 0으로 되돌리지 않는다. 따라서 "앞 절반" 규칙은 프로세스의 첫 요청의 첫 layer에만 적용되고, 그 뒤 요청의 첫 layer는 직전 요청의 끝에서 두 번째 layer에서 잰 예산을 쓴다. 예산이 0보다 크지만 첫 chunk의 예상 시간보다 작으면 아무것도 읽지 않는다 | **논문의 서술을 따른다.** 시간 예산, chunk당 예상 시간 상수, 앞 절반 규칙을 쓰지 않는다. worker는 읽기 하나를 시작하기 전마다 멈춤 표시를 확인하고, layer l의 GPU 계산이 실제로 끝나면(t행) 새 읽기를 시작하지 않는다. 이미 시작한 읽기 하나는 끝낸다. 첫 prefetch(layer 0 → 1)도 같은 규칙이다. (이전: 공개 코드의 시간 예산 방식, 상수는 이 장비에서 잰 값) | 논문과 같고 공개 코드와 다름. 공개 코드는 예상 시간으로 미리 자른다. 예상이 빗나가 계산이 끝난 뒤까지 읽는 일이 없어진다 — 유리 |
| d | 검증과 복구 | 다음 layer에서 실제 선택 G를 구한 뒤 빠진 것만 동기로 읽는다. 집합 차 연산 약 40 µs (IV-C, VI-F) | `missing = sele[~isin(sele, prefetched)]` (F 1120–1134), `pcache.get(missing)`으로 동기 읽기(F 1141). 복구는 IMPRESS의 요청 시 읽기 경로를 그대로 쓴다 | G_l \ P_l이 들어 있는 chunk만 IMPRESS 이식본의 읽기 함수로 동기 읽기. K와 V를 따로 본다(K만 prefetch된 chunk는 V만 읽는다) | 같음 |
| e | 버리기 | prefetch됐지만 중요하지 않다고 판정된 KV는 버린다 (IV-C, Fig. 14) | 성능 측정 경로(`--suffix-comp`)는 prefetch된 KV를 모두 `filled_prefix_k/v`에 채워 넣고(F 1170–1173) mask 없는 `mha_with_prefixkv_suffix`로 계산한다(F 1192–1195, B 451). 정확도 경로(`mha_with_filled_selected_prefixkv`)는 선택에서 빠진 토큰(`del_tokenids`)을 `finfo.min`으로 가린다(B 1267) | attention에는 G_l만 쓴다. prefetch된 chunk 중 G_l 토큰이 하나도 없는 chunk는 GPU에 올리지 않고 버린다. G_l 밖의 토큰은 mask로 가린다. 한 경로에서 정답률과 시간을 함께 잰다 | 논문의 서술과 같고 공개 성능 경로와 다름. 출력이 IMPRESS와 같아진다 — 중립 (GPU로 옮기는 양은 IMPRESS와 같음) |
| f | probe key | 식별은 IV-B 방식 그대로. prefetch 대상은 다음 layer의 중요 KV | HyperInfer 설정(S 211–229)은 `--prefetch`를 주지 않으므로 `prefetched_head`가 비어 있고, probe key를 임계 경로에서 동기로 읽는다(F 1044–1048, `pcache.get_head`). 이 시간은 TTFT에 들어간다(F 1257) | 다음 layer의 probe key를 worker가 미리 읽는다(IMPRESS 이식본과 같음). worker는 probe key를 먼저 읽고 이어서 예측 chunk를 읽는다. probe key 읽기는 멈춤 표시와 무관하게 끝까지 하고, 선택이 필요로 하는 시점에 아직 안 끝났으면 요청 thread가 그것만 기다린다. layer 0의 probe key는 동기 | **공개 코드와 다름 — 유리.** IMPRESS 이식본에도 같은 차이가 있다(IMPRESS 계약 6절 #4). probe key 읽기가 계산 창의 일부를 쓰므로 그만큼 chunk를 덜 읽는다 |
| g | fallback layer | 유사도가 임계값보다 낮으면 표준 방식(전체 로드)으로 돌아간다 (IV-B) | 선택 함수가 prefix 전체를 돌려준다(B 1450–1451). 그대로 다음 layer의 예측이 되므로 예측 chunk는 전체가 되고, 예산으로 잘린다 | 같음. fallback layer에서 G_l = 전체. 그 layer 자신은 prefetch된 chunk를 뺀 나머지 chunk를 동기로 읽는다 | 같음 |
| h | 마지막 layer | 다음 layer가 없다 | `layer_id < num_hidden_layers - 1`일 때만 제출한다(F 1100, 1223–1238) | 같음 | 같음 |
| i | worker와 대기 | prefetch는 계산과 비동기로 돈다 (V) | `ThreadPoolExecutor(max_workers=1)`(P 2036). 다음 layer 시작에서 `resolve_pending_prefetch`로 끝나기를 기다리고(F 1038–1040, 1636–1657) 기다린 시간은 그 layer의 TTFT에 더한다(F 1257, R "TTFT definition") | worker 1개. layer l 시작에서 GPU 계산 완료를 확인하고 멈춤을 표시한다. 요청 thread가 기다리는 것은 둘뿐이다: (1) probe key 읽기가 아직 안 끝났으면 그것, (2) 진행 중이던 읽기 하나 — 이것은 선택 계산과 겹쳐 흐르고, 선택이 끝난 뒤 첫 KV 읽기 직전에 worker를 합류시킨다. 두 대기 모두 TTFT에 들어가고 따로 기록한다. (이전: 예측 chunk 전체를 다 읽을 때까지 기다림) | 논문과 같고 공개 코드와 다름. 공개 코드는 제출한 prefetch 전체가 끝나기를 기다린다 — 유리 |
| j | 캐시 계층, 주기적 재배치, warm-up | GPU 10 GB / CPU 32 GB 캐시, CKLFU 점수 기반 교체(IV-D 2), 약 10분 주기 재배치(IV-D 1), 측정 전 캐시 warm-up(VI-B) | `--gpu-size 10240 --cpu-size 32768 --cache-type CKLFU --reorder`(S 27–28, 211–229). 재배치 순서는 사전 실행(`--generate-mapping-list`, S 118–124)으로 만든다. prefetch는 캐시 상태를 바꾸지 않는다(P 2123–2126) | 캐시 계층 없음, 요청마다 page cache를 비운 cold SSD, warm-up 없음. 재배치는 첫 대화 T1의 통계로 한 번(IMPRESS 이식본과 같음) | 불리 (모든 방법 공통 조건, IMPRESS 계약 6절 #3과 같음) |

## 2. 추가로 확인한 항목

| # | 항목 | 논문의 서술 (절) | 공개 코드의 동작 (파일·줄) | 우리 이식본 | 차이와 방향 |
|---|---|---|---|---|---|
| k | 예산에 쓰는 시간 | GPU 계산 창 | `compute_time` = 계산 stream의 CUDA event 경과 시간(attention + MLP, F 1254), 여기에 선택 시간을 더한다(F 1264). probe 읽기와 동기 KV 읽기 시간은 넣지 않는다 | 쓰지 않는다. layer별 계산 창(제출 시각 ~ 다음 layer 시작에서 GPU 동기화가 끝난 시각)은 보고용으로만 잰다. (이전: 이 실측 시간 + 선택 시간을 예산으로 씀) | 해당 없음 (c행 참조) |
| l | 제출 시점 | 현재 layer 계산과 동시에 시작 (IV-C) | 현재 layer의 선택과 동기 읽기가 끝나고 계산을 계산 stream에 올린 직후(F 1223–1238) | layer l의 선택·읽기·GPU 적재가 끝난 직후, layer 계산 직전 | 같음 |
| m | worker가 하는 일 | — | 디스크 읽기 + pinned buffer 복사 + 별도 CUDA stream으로 GPU 복사까지(P 1481–1531, 2087–2121). 주 thread는 event만 기다린다(P 16–35) | worker는 디스크 읽기와 host tensor 변환만 한다. GPU 복사와 cache 기록은 주 thread가 한다(FullLoad, IMPRESS, Ours의 비동기 읽기와 같은 규칙) | **공개 코드와 다름 — 불리.** GPU 복사 시간이 임계 경로에 남는다. 다른 방법과 조건을 맞추기 위한 고정 사항 |
| n | 복구 읽기 단위 | — | chunk별 모음 | 인접 chunk는 pread 하나로 합친다(IMPRESS 이식본과 같음). fallback layer에서는 prefetch된 chunk를 뺀 나머지 구간을 합쳐 읽는다 | 유리 (IMPRESS 계약 6절 #10, #11과 같음) |
| o | 첫 layer | 첫 layer는 미리 읽은 것이 없다 | layer 0은 prefetch 결과를 쓰지 않고 전부 동기로 읽는다(F 1120의 `layer_id == 0` 조건) | 같음. layer 0에는 prefetch가 없고, layer 1용 prefetch부터 c행의 규칙을 쓴다 | 같음 |
| p | 디코딩 단계 | 디코딩은 바꾸지 않는다 (IV-A) | 측정은 `--gen-len 1`(1토큰) | 최대 16토큰 greedy. 선택과 prefetch는 prefill에서만 하고 디코딩 중에는 고정 | IMPRESS 계약 6절 #14와 같음 — 중립 |
| q | TTFT의 범위 | — | layer별 (대기 + probe 읽기 + 선택 + 동기 읽기 + 계산)의 합(F 1257, R). 토큰화, 임베딩, 출력층은 들어가지 않는다 | 요청 시작부터 첫 토큰까지의 실측 시간(다른 방법과 같은 구간) | 방법 간 비교에는 중립 |
| r | 예측 → chunk 변환에 쓰는 표 | — | prefix를 캐시에 넣을 때 만든 `reorder` / `reverse_reorder`(P 1671, 1712) | store의 `meta.json` 순서에서 만든 layer l → l+1 위치 표. 요청 타이머 밖(context 준비 시)에 한 번 만든다 | 같음 |
| s | Oracle, History 비교 대상 | VI-A (6), VI-D | 이 실험의 대상이 아니다 | 만들지 않는다 | 해당 없음 |
| t | GPU 계산 완료의 판단 | "monitoring the GPU status", GPU가 현재 layer 계산을 끝내는 즉시 (IV-C) | GPU 상태를 보지 않는다(예상 시간으로 미리 자름). 계산 시간은 사후에 CUDA event로 잰다(F 1254) | layer l+1의 hook 첫머리에서 `torch.cuda.synchronize()`가 돌아온 시점을 layer l의 계산 완료로 본다(추정이 아니라 실제 완료). 바로 이어 job의 lock 안에서 멈춤을 표시한다. worker의 읽기 시작 판단도 같은 lock 안에서 한다 | 논문과 같음. 동기화 호출은 layer마다 한 번 더 들어간다(선택이 직전 layer의 출력을 필요로 하므로 그 계산을 기다리는 것 자체는 원래 있는 일). 비용은 보고서에 적는다 |
| u | 진행 중이던 읽기 | 새 요청을 내지 않는다(이미 낸 요청에 대한 서술은 없다) | 해당 없음(미리 자른 목록을 끝까지 읽고 그것을 기다림) | 이미 시작한 읽기 하나는 끝내고 그 결과를 쓴다. 읽기는 취소하지 않는다 | 논문의 서술 범위 안에서 정한 것 — 중립 |

## 3. 요청하신 내용 중 고쳐 적은 것

- **c.** "예산이 0인 첫 layer는 앞 절반을 읽는다"는 맞지만, 공개 코드에서 예산이 0인 경우는 프로세스의 첫 요청뿐이다. 예산 변수를 요청마다 초기화하지 않기 때문이다. 2026-11-24 이식본은 공개 코드를 따랐다(요청 사이에 이어짐). 2026-11-25 이식본은 예산을 쓰지 않으므로 이 규칙이 없다.
- **e.** 공개 코드의 정확도 경로는 prefetch된 KV를 채워 넣은 뒤 선택에서 빠진 토큰을 mask로 가린다. 즉 "모두 채워 넣고 그대로 쓰는" 것은 성능 측정 경로에만 해당한다.
- 그 밖의 a, b, d, f, g, h, i, j는 읽은 내용과 일치한다.

## 4. 공개 코드를 직접 호출해 대조하는 범위 (2026-11-24 실행의 gate 1)

2026-11-25 이식본은 예측 집합과 chunk 번호 계산을 이전 이식본의 함수(`mmimpress/hyperinfer.py`의 `predicted_chunks`, `layer_position_maps`)를 그대로 불러 쓴다. 아래 대조 테스트(`tests/test_hyperinfer.py`)는 그 함수들을 계속 검증하고 기존 테스트 묶음에 들어 있다. 멈추는 방식은 공개 코드와 다르게 했으므로 공개 코드와 대조하지 않고 `tests/test_hyperinfer_v2.py`로 검증한다.

- 직접 호출: `PrefixKVLayer.prefetch_by_chunk`의 순서 변환, chunk 번호 계산, 예산 자르기, 앞 절반 규칙, prefetch된 토큰 집합(P 1665–1715). 객체를 디스크 없이 만들고 읽기 함수만 기록용으로 바꿔 같은 입력에서 같은 chunk 목록과 토큰 집합이 나오는지 본다.
- 직접 호출하기 어려운 부분: 예측 집합(`sele_tokenids.clone()`)과 빠진 토큰 계산(`~torch.isin`)은 F의 1,300줄짜리 layer 함수 안에 있어 모델 가중치와 전역 상태 없이는 부를 수 없다. 그 두 줄과 같은 연산을 테스트 안에 적고 같은 입력에서 이식본의 결과와 비교한다. 이 사실을 테스트와 보고서에 적는다.
