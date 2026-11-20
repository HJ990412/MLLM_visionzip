# 실험 A — LLaVA-NeXT × MT-GQA: ReComp / FullLoad / IMPRESS / Ours (IMPRESS = 공개 구현 방식)

측정값만 적는다. 측정하지 않은 값은 "미실행"이다. 계약: `docs/mt_gqa_impress_code_contract.md` (사본 `CONTRACT.md`).

## 설정

- 실행: `runs/mt_gqa_impress_code_dev_20261117T182946Z` (요청 단위 원본), 이 디렉터리(`results/mt_gqa_impress_code_dev_20261117T182946Z`)는 집계와 보고서.
- 환경: Python 3.10.20, torch 2.5.1+cu121, transformers 4.57.6 (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`). README 38절 본실행과 같은 환경이다.
- 공개 구현: https://github.com/ISCS-ZJU/HyperInfer, 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e` (체크아웃 `/home/dblab/hj/_external_ref/HyperInfer`, 레포 밖, 읽기 전용 참조; 코드는 복사하지 않았다).
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd76…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2.
- 모집단·절차(README 38절 본실행과 동일): MT-GQA-reconstructed 398 images / 4,061 dialogues / 3턴, method-local generated history, strict normalized EM, 대화 단위 arm 순서 회전, 턴 1은 모든 방법이 픽셀부터 계산, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건, 저장 KV 방법은 비동기 읽기.
- IMPRESS(공개 구현 방식, `mmimpress/impress_official.py`; 이 실험의 유일한 IMPRESS): probe head [0, 1, 2], 유지 25%, 유사도 임계값 0.3(이상이면 합의), chunk 64 토큰. 중요도 = 턴 1에서 저장한 prefix 행 기여분 + 요청 때 계산한 suffix 행 기여분. fallback은 그 layer의 이미지 토큰 K·V 전부를 읽어 mask 없이 사용. 재배치는 첫 대화 턴 1 요청 하나의 통계(모든 head·모든 행 열 합)로 한 번. 다음 layer probe sidecar 선행 읽기.
- FullLoad(layer별 K·V 전체를 워커가 순서대로 읽음), Ours(PYR-beta-raw packed, 워커가 구간 순서로 읽음), ReComp: README 38절 본실행과 같은 코드·설정. 캐시 계층 없음.

## 검증 단계

| 단계 | 결과 | 내용 |
|---|---|---|
| 1. 공개 구현과의 동등성 (CPU) | 미실행 |  |
| 2. prefix 행 기여분 (gate 4장, GPU) | PASS | n181355: prefix 행 일치, 재배치 통계 일치; n204894: prefix 행 일치, 재배치 통계 일치; n229548: prefix 행 일치, 재배치 통계 일치; n95369: prefix 행 일치, 재배치 통계 일치 |
| 3. 정합성: 유지 100% IMPRESS = 같은 재배치 store의 FullLoad loader (gate) | PASS | 147건 중 불일치 0건 |
| 3. 참고: 재배치 store FullLoad vs raster store FullLoad (판정 아님) | 참고 | 147건 중 생성 토큰이 다른 요청 1건 (그중 프롬프트가 같은 요청 1건) |
| 4. 비동기 정합성: 비동기 = 동기 (gate) | PASS | FullLoad: 147건 중 불일치 0건; IMPRESS: 147건 중 불일치 0건; Ours: 147건 중 불일치 0건 |
| 5. 재현성: README 38절 본실행과 토큰 일치 (gate) | PASS | ReComp: 147건 중 불일치 0건; FullLoad: 147건 중 불일치 0건; Ours: 147건 중 불일치 0건 |
| 5. 재현성: 모든 arm의 턴 1 토큰 일치 (gate) | PASS | 대화 49개 × arm 9개, 불일치 0개 |
| 5. 재현성: README 38절 본실행과 토큰 일치 (DEV) | 미실행 |  |
| 5. 재현성: 모든 arm의 턴 1 토큰 일치 (DEV) | 미실행 |  |
| 6. 기존 테스트 | 미실행 |  |
| 7. DEV 40장 실행·감사 (본조건) | 미실행 |  |
| gate phase 실행·감사 | PASS | 1323/1323건, 4장 |
| 8. 본실행 진입 | 미실행 |  |

2번의 참고값(기준 아님): 같은 이미지를 다른 질문의 요청 안에서 다시 캡처한 prefix 행 기여분과 저장값의 차이는 최대 절댓값 2.39, 평균 절댓값 0.00226; prefix 행만으로 뽑은 head별 상위 25% 집합의 Jaccard는 평균 0.9805 (이미지 4장). 원소 단위로 완전히 같은 이미지는 0장이다.

## 본실행 (398장)

미실행.

## DEV 40장 — 소규모 확인

미실행.

## IMPRESS 내부 통계

미실행.

## 저장·턴 1 비용

미실행.

## 공개 구현의 함수와 이 구현의 대응

| 공개 구현 (`h2o_flexgen/flexgen/`) | 이 구현 (`mmimpress/impress_official.py`) |
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

| # | 항목 | 원본 | 이 실험 | 방향 |
|---|---|---|---|---|
| 1 | 선택 대상 | prefix 토큰 전부 | 이미지 토큰만. 시스템 프롬프트와 separator는 항상 유지, 예산 밖 | 정답률에 유리 / 읽기량에 소폭 불리 |
| 2 | prefix 행 기여분 | 정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태) | 턴 1(mask 없음)의 값을 저장해 요청 때 더함 | 유리 (요청 때 prefix 행 연산 없음) |
| 3 | 캐시 계층 | 논문·공개 구현: GPU·DRAM·SSD 다층 캐시 | 없음. 모든 방법이 cold SSD | 불리 (모든 방법 공통 조건) |
| 4 | 다음 layer probe 선행 읽기 | 공개 구현의 IMPRESS 설정은 `--no-prefetch` | 켬 | 유리 |
| 5 | separator의 저장 위치 | 해당 없음(텍스트 prefix) | 저장 순서 맨 앞에 모음(항상 읽으므로 chunk 1개에 담김) | 유리 |
| 6 | 득표 동점 | 순서 미정의(unstable argsort) | 저장 위치가 앞선 토큰 우선 | 유리 또는 중립 |
| 7 | 재배치 통계의 추론 상태 | head별 25% mask 추론, 같은 prefix의 마지막 요청 | mask 없는 첫 대화 턴 1 요청 하나 | 중립 |
| 8 | 재배치 적용 | `key[reorder]` (token→rank 인덱스를 그대로 적용해 내용이 어긋남) | 위치 r = rank r 토큰, K·V·probe 같은 순서 | 중립 (접근 위치는 같다) |
| 9 | 중요도 수치 정밀도 | float16 softmax·합 | BF16 확률 + float32 합 / float32 softmax | 중립 |
| 10 | Fallback 읽기 단위 | chunk 파일별 | layer의 K 파일 1회, V 파일 1회 pread | 유리 (호출 수 적음, 바이트 같음) |
| 11 | 선택 chunk 읽기 단위 | chunk 파일별 | 인접한 chunk는 pread 하나로 합침 | 유리 |
| 12 | prefix 행 기여분 읽기 | 없음 | probe sidecar 끝 12·N bytes/layer를 같은 read로 | 불리 (probe 읽기 +1.6%) |
| 13 | mask | 정확도 경로에만 있음(성능 경로는 0으로 채우고 답을 내지 않음) | mask (한 경로에서 정답률과 시간을 함께 잼) | 중립 |
| 14 | 모델·생성 | OPT(FP16) 텍스트 LLM, 1토큰 logits | LLaVA-NeXT(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정 | 중립 |
| 15 | HyperInfer의 KV 선행 로딩(선택된 토큰의 다음 layer K·V 예측 읽기) | HyperInfer 확장 | 없음 (IMPRESS의 probe 선행 읽기만) | 불리 (HyperInfer 대비; IMPRESS 정의에는 해당 없음) |

## 기존 산출물 보호

- `results/`, `runs/`(이 run 제외), `kvstore*`의 기존 파일 2,441,103개 (경로, 크기, mtime) 대조: **PASS**.
