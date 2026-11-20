# MT-GQA 다중 턴 — 공개 IMPRESS 구현의 로직을 이식한 IMPRESS arm: 실행 계약

2026-11-17 UTC, 어떤 GPU 측정보다 먼저 고정. 저장소 `/home/dblab/hj/mllm_v2`.
Run root `runs/mt_gqa_impress_official_<UTC>`, 결과 `results/mt_gqa_impress_official_<UTC>` (같은 `<UTC>`).
이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다. 결과를 본 뒤 기준·표본·arm을 바꾸지 않는다.

## 1. 목적

직전 본실행(`results/mt_gqa_async_20261116T081635Z/`, 계약 `docs/mt_gqa_async_contract.md`)의 IMPRESS arm은 논문 본문을 보고 만든 이식이었다.
이번에는 저자들의 공개 구현을 기준으로 선택 경로를 새로 작성하고, 공개 구현의 함수와 같은 입력에서 같은 결과를 내는지 테스트로 고정한 뒤,
같은 MT-GQA 조건에서 ReComp, FullLoad, Ours와 함께 다시 잰다. 측정 보고이며 GO/NO-GO 판정 규칙은 없다.
사전 고정하는 것은 검증 단계의 통과 기준(8절)과 보고 항목의 정의(9절)다.

표와 보고서의 방법 이름은 "ReComp", "FullLoad", "IMPRESS", "Ours"만 쓴다. DEV 표에서만 직전 실행의 arm을 "IMPRESS (이전 구현)"으로 구분한다.

## 2. 환경과 참조 구현

- 실행 인터프리터: `/home/dblab/anaconda3/envs/mllm_ft/bin/python` — Python 3.10.20, torch 2.5.1+cu121, transformers 4.57.6 (직전 본실행과 같다).
  실행 시점 버전과 pip freeze는 `environment_before.json`에 기록하고 각 phase 시작 때 대조한다.
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 / BF16 / eager, FP16 SSD KV, greedy max 16,
  seed 1234, batch 1, CPU threads 2, RTX 4090 1장 단독 사용.
- 공개 구현: https://github.com/ISCS-ZJU/HyperInfer, 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`.
  체크아웃 위치는 이 레포 바깥의 `/home/dblab/hj/_external_ref/HyperInfer`(읽기 전용 참조). 최상위에 라이선스 파일이 없으므로 그 코드를 이 레포에
  복사하지 않는다. 테스트는 환경변수 `IMPRESS_OFFICIAL_REPO`로 경로를 받아 그 함수를 호출한다. freeze는 체크아웃의 HEAD가 위 커밋인지 확인하고 기록한다.

## 3. Arm과 설정값

| 표의 이름 | method id | store 종류 | 비동기 읽기 |
|---|---|---|---|
| ReComp | `recompute` | 없음 | — |
| FullLoad | `fullload_async` | `raster` | 워커가 layer 순서로 K·V 전체 (`layer_stream`) |
| IMPRESS | `impress_official_async` | `impress_official` | 다음 layer probe sidecar 선행 읽기 (`next_layer`) |
| Ours | `ours_prefixstore_kv25_PYR-beta-raw_packed_async` | `packed_PYR-beta-raw` | 워커가 packed 파일을 구간 순서로 (`layer_stream`) |
| IMPRESS (이전 구현) — DEV만 | `impress_async` | `impress` | 다음 layer probe key 선행 읽기 |

- **본실행 arm (이 순서로 고정, 4개)**: `recompute`, `fullload_async`, `impress_official_async`, `ours_prefixstore_kv25_PYR-beta-raw_packed_async`.
  직전 본실행의 arm 순서에서 IMPRESS 자리만 새 arm으로 바뀐다. 대화 단위 순서 회전(`method_order`)은 그대로다.
- ReComp, FullLoad, Ours, IMPRESS (이전 구현)의 코드·설정은 직전 본실행과 같다(`mmimpress/serve.py`, `packed_serve.py` 등은 수정하지 않는다).
  Ours는 `runs/layer_budget_kv25_pyrbeta_20261110T073604Z/vectors/layer_budget_llava_PYR-beta-raw.json` 그대로다.
- 검증 전용 arm(어떤 표에도 넣지 않는다): `impress_official`(새 arm의 동기 읽기), `impress_official_r100`(새 경로, 유지 비율 100%). gate phase에만 있다.
- **새 IMPRESS arm의 설정값**: probe head 0·1·2, 유지 25% (k = ceil(0.25·N_image)), 유사도 임계값 0.3 (`sim >= 0.3`이면 합의), chunk 64 토큰,
  다음 layer probe sidecar 선행 읽기. 공개 구현의 `scripts/lm_performance_fast_pcache.sh`(`sele_percents`의 25에 `sim_threds` 0.3)와
  `scripts/figure15.sh`(`run_sele_load_by_percent 25 0.3`), `--sele-head` 기본값 `0 1 2`, `--chunk-size` 기본값 64에서 가져왔다.

## 4. 데이터와 절차 (직전 본실행과 동일)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm. dialogues와 image identity가 직전 본실행의 manifest와 같음을 freeze에서 assert한다.
- 수명주기는 직전 본실행의 것(`scripts/187`, 그 안에서 `scripts/164`를 import)을 그대로 쓴다: method-local generated history, strict normalized EM,
  greedy max 16, 대화 단위 arm 순서 회전, image-atomic attempt, 이미지별 독립 감사 후 commit, commit 뒤 run-local scratch store만 정확한 SHA allowlist로 삭제,
  중단 시 `--resume`. 실행 스크립트 `scripts/190`은 `scripts/187`의 사본에 새 store 종류와 새 arm을 더한 것이다.
- **턴 1은 모든 방법이 픽셀부터 계산한다.** 저장 KV 읽기는 요청 직전에 그 store의 모든 `.bin`을 `posix_fadvise(DONTNEED)`로 비운 cold 조건이고 비우기는 타이머 밖이다.
  TTFT 경계도 같다: 요청 시작(프롬프트 조립·토큰화 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`.
- store는 이미지의 **첫 대화 턴 1**에서, store 종류마다 그 대화의 회전 순서에서 그 종류를 쓰는 첫 arm의 턴 1 캡처로 만든다.

| phase | 이미지 | 대화 | arm | 요청 수 |
|---|---|---|---|---|
| `gate` | 직전 실행과 같은 smoke 이미지 4장의 모든 대화 | 49 | 6: 본실행 4 + `impress_official` + `impress_official_r100` | 882 |
| `dev` | v2 `data_split.json`의 DEV 40장의 모든 MT-GQA 대화 | 378 | 5: 본실행 4 + `impress_async` | 5,670 |
| `main` | 398장 전체 | 4,061 | 4 | 48,732 |

gate·dev 이미지는 main에 다시 포함된다. dev는 구현 확인과 이전 구현과의 같은 실행 내 비교용이며 main의 arm이나 설정을 고르는 데 쓰지 않는다.

## 5. 새 IMPRESS arm의 정의

구현: `mmimpress/impress_official.py`(신규). 기존 `LayerSelector`와 별개의 클래스(`OfficialImpressSelector`)와 요청 함수(`request_official`)다.
재사용하는 것은 저장·읽기 기본 기능(`ChunkReader`, `PrefixCache`, `IOCounter`, raster writer, additive mask `BIAS`/`bias_from_keep`)과 서버의 디코딩 루프·워커 스레드다.

**턴 1 (이미지의 첫 대화, 그 arm의 픽셀 요청).**
- 그 요청의 prefill에서 모델이 계산한 attention 확률을 읽기만 하는 캡처(`Turn1AttentionStats`)를 건다. 별도 forward는 없다. layer마다 두 가지를 얻는다.
  - **prefix 행 기여분** (probe head 0·1·2 × 이미지 토큰): prefix 위치(시스템 프롬프트 + 이미지)의 행에 대한 attention 확률을 열 방향으로 더한 값.
    질문과 무관하다(causal mask 때문에 prefix 행은 suffix를 보지 않는다). float32.
  - **재배치 통계** (이미지 토큰): 모든 head, 모든 행(prefix + 그 턴 1 suffix)의 열 합. 공개 구현의 `--generate-mapping-list` 경로가 정렬하는 값이다.
- 저장: FullLoad와 같은 raster writer로 Visual KV 전체와 probe key 사본(head 0·1·2의 key)을 쓴 뒤, layer별로
  **[separator(원래 순서) | 이미지 토큰을 재배치 통계의 내림차순으로]** 순서로 K, V, probe key 파일을 다시 쓴다. probe key 파일 끝에 그 layer의 prefix 행 기여분을
  같은 저장 순서로 붙인다(little-endian float32, 3 × N). `meta.json`에 layer별 순서를 기록한다. **이후 대화로 갱신하지 않는다.**
- 캡처 시간은 그 턴 1 요청의 TTFT·E2E 안에 들어간다(그 요청 하나). 재배치 쓰기는 요청 밖이며 따로 잰다. 둘 다 followup TTFT에는 들어가지 않는다.

**턴 2·3 (layer마다, 디코더 layer의 pre-hook).**
1. probe sidecar를 한 번에 읽는다(probe key + prefix 행 기여분). 비동기 arm은 layer 0은 hook 안에서, layer i+1은 layer i의 hook이 끝날 때 워커 1개가 읽는다.
2. 중요도 = 저장된 prefix 행 기여분 + suffix 행 기여분. suffix 행 기여분은 probe head의 query(suffix 전체 행: 이전 턴의 질문·답 + 현재 질문 + 템플릿)와
   key(시스템 프롬프트 key, probe key, suffix key)로 causal softmax를 계산해 이미지 열을 더한 값이다(float32 softmax).
3. 선택: head별 상위 k → 세 head 쌍의 평균 Jaccard → `>= 0.3`이면 득표수 상위 k개(합의), 미만이면 fallback.
   separator는 후보가 아니며 예산 밖에서 항상 읽고 유지한다. 시스템 프롬프트 KV는 기존처럼 activation 때 올려 둔 것을 쓴다(모든 방법 공통).
4. 합의: 선택된 토큰 또는 separator가 속한 64 토큰 chunk의 K와 V를 읽고, 선택되지 않은 이미지 토큰은 mask한다(읽은 chunk 안의 선택 안 된 토큰 포함).
   Fallback: 그 layer의 Visual K·V를 전부 읽고 mask 없이 모두 쓴다.
5. 디코딩 단계에서는 읽기도 선택도 다시 하지 않는다.

**득표 동점.** 공개 구현은 득표수를 unstable `argsort`로 정렬해 앞의 k개를 쓰므로 같은 득표수 안의 순서가 정해져 있지 않다. 새 구현은 저장 위치가 앞선 토큰
(= 재배치 순서에서 앞선 토큰)을 먼저 고른다.

## 6. 공개 구현을 읽고 확인한 것

구현 전에 `h2o_flexgen/flexgen/`의 해당 코드를 직접 읽었다. 사전 조사 설명과 같은 점: 중요도는 probe head의 전체 행(prefix + suffix) attention의 열 합(`sele_tokenid`),
head별 상위 `ceil(n·percent/100)`, 평균 pairwise Jaccard(`cal_jaccard_sim`), `sim >= sim_thred`이면 득표 합의, 아니면 prefix 토큰 전부를 읽고 모두 사용
(`_sele_important_tokens`가 `arange(prefix_len)`과 빈 삭제 목록을 돌려준다), K·V 분리 저장과 64 토큰 chunk(`KV_Division`), probe head key는 따로 통째로 저장(`head_token`).

사전 조사 설명과 다르거나 설명에 없던 점(코드를 따르거나, 따를 수 없는 경우 이유를 적는다):

1. **mask는 정확도 경로에만 있다.** `figure15.sh` → `overall_acc.sh` → `flex_opt.py`(정확도)는 `--suffix-comp` 없이 실행되어 `mha_with_filled_selected_prefixkv`가
   선택 안 된 토큰의 열에 `finfo.min`을 넣는다. `lm_performance_fast_pcache.sh`(성능)의 IMPRESS 설정은 `--suffix-comp --fill-keys-zero`라서
   `mha_with_prefixkv_suffix`가 mask 없이 0으로 채운 K/V로 계산하고 출력 텐서는 `torch.empty`다(시간 측정 전용, 답을 내지 않는다).
   새 구현은 정확도 경로를 따라 mask한다.
2. **재배치 적용부는 그대로 따를 수 없다.** `--generate-mapping-list`가 저장하는 것은 token → rank 인덱스(`reverse_indices`)인데, `PrefixKVLayer`는 이를
   `key[reorder]`로 적용해 위치 i에 token `rank(i)`를 놓고, 조회는 `reorder[token]` 위치에서 읽는다. 접근 위치(중요한 토큰이 앞 chunk)는 의도대로지만
   읽히는 내용이 그 토큰의 KV가 아니다(성능 경로는 답을 내지 않으므로 드러나지 않는다). 새 구현은 위치 r에 rank r 토큰을 놓고 K·V·probe 행을 같은 순서로 유지한다.
   순서를 만드는 방식(모든 head·모든 행 열 합의 내림차순)은 공개 구현 그대로다.
3. **재배치 통계가 나오는 추론 상태.** 공개 구현의 mapping 경로는 `--full-load --sele-inf`(head별 25% mask로 추론하는 상태)에서 통계를 내고, 같은 prefix의 요청이
   여럿이면 마지막 요청의 값이 남는다(요청 간 누적 없음). 새 구현은 mask 없는 턴 1 추론의 attention에서 통계를 낸다(지정 사항: 첫 대화 턴 1 요청 하나).
4. **공개 정확도 경로의 prefix 행.** `sele_tokenid`는 매 요청 prefix 위치의 hidden state로 query를 만든다. 그 hidden state는 선택된 KV만 보는 상태로 다시 계산된 것이다.
   새 구현은 지정대로 턴 1(mask 없음)의 prefix 행 기여분을 저장해 쓴다.
5. 공개 구현의 중요도는 float16 softmax와 float16 합이다. 새 구현은 prefix 행은 모델의 BF16 attention 확률을 float32로 더하고, suffix 행은 float32 softmax를 쓴다.

### 원본과 다른 점 (보고서에 같은 표를 싣는다)

"방향"은 IMPRESS(baseline)의 TTFT·정답률에 유리한지, 불리한지, 중립인지다.

| # | 항목 | 공개 구현 | 새 구현 | 방향 |
|---|---|---|---|---|
| 1 | 선택 대상 | prefix 토큰 전부 | 이미지 토큰만. 시스템 프롬프트와 separator는 항상 유지, 예산 밖 | 정답률에 유리 / 읽기량에 소폭 불리 |
| 2 | prefix 행 기여분 | 정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태) | 턴 1(mask 없음)의 값을 저장해 요청 때 더함 | 유리 (요청 때 prefix 행 연산 없음) |
| 3 | 캐시 계층 | GPU·DRAM·SSD 다층, CKLFU | 없음. 모든 방법이 cold SSD | 불리 (모든 방법 공통 조건) |
| 4 | 다음 layer probe 선행 읽기 | IMPRESS 설정은 `--no-prefetch` | 켬 | 유리 |
| 5 | separator의 저장 위치 | 해당 없음 | 저장 순서 맨 앞에 모음(항상 읽으므로 chunk 1개에 담김) | 유리 |
| 6 | 득표 동점 | 순서 미정의 | 저장 위치가 앞선 토큰 우선 | 유리 또는 중립 |
| 7 | 재배치 통계의 추론 상태 | head별 25% mask 추론 | mask 없는 턴 1 추론 | 중립 |
| 8 | 재배치 적용 | `key[reorder]` (위 2번) | 위치 r = rank r 토큰 | 중립 (접근 위치는 같다) |
| 9 | 중요도 수치 정밀도 | float16 | BF16 확률 + float32 합 / float32 softmax | 중립 |
| 10 | Fallback 읽기 단위 | chunk 파일별 | layer의 K 파일 1회, V 파일 1회 pread | 유리 (호출 수 적음, 바이트 같음) |
| 11 | 선택 chunk 읽기 단위 | chunk 파일별 | 인접한 chunk는 pread 하나로 합침 | 유리 |
| 12 | prefix 행 기여분 읽기 | 없음 | probe sidecar 끝 12·N bytes/layer를 같은 read로 | 불리 (probe 읽기 +1.6%) |
| 13 | mask | 정확도 경로만 (위 1번) | mask | 중립 |
| 14 | 모델·생성 | OPT(FP16), 1토큰 logits | LLaVA-NeXT(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정 | 중립 |

## 7. 코드 변경 범위

- 신규: `mmimpress/impress_official.py`, `tests/test_impress_official.py`, `scripts/190_eval_mt_gqa_impress_official.py`(실행),
  `scripts/191_validate_mt_gqa_impress_official.py`(gate), `scripts/192_report_mt_gqa_impress_official.py`(보고; 측정에 관여하지 않으므로 소스 freeze에서 제외).
- 기존 파일은 수정하지 않는다(`mmimpress/serve.py` 등 기존 모듈, 기존 스크립트 01–189, 기존 테스트). 기존 `results/`, `runs/`, store도 건드리지 않는다.

## 8. 검증 단계와 통과 기준 (사전 고정; 하나라도 실패하면 본실행 없이 실패 내용만 보고)

1. **공개 구현과의 동등성** (`tests/test_impress_official.py`의 `PublicEquivalenceTests`, `IMPRESS_OFFICIAL_REPO` 지정). 같은 텐서를 공개 함수와 새 구현에 넣는다.
   통과 기준: 모든 case가 skip 없이 실행되어 통과.
   - `_sele_important_tokens`: head별 상위 k 집합, 평균 Jaccard 값, fallback 판정, 최종 선택. 경계에 동점이 없는 입력은 집합 일치, 동점이 있는 입력은 득표수 조건
     (양쪽 모두 k개, 선택 밖 토큰의 득표가 선택 안 토큰보다 많지 않음, 선택된 토큰들의 득표수 multiset이 같음). fallback·동점 없음·동점 있음 세 경우가 모두 나와야 한다.
   - `cal_jaccard_sim`: 값 일치. 임계값 판정이 `>=`임을 경계값으로 확인.
   - `sele_tokenid`: hidden state에서 선택까지 전체(OPT 형식 layer, CPU, 공개 구현과 같은 float16 softmax).
   - 요청 때 쓰는 분해(저장된 prefix 행 기여분 + q·k로 계산한 suffix 행 기여분)가 전체 행렬에 대한 공개 함수의 선택과 일치.
   - `mha_with_filled_selected_prefixkv`: 선택 안 된 토큰의 mask가 keep mask로 만든 additive mask와 같은 출력을 내는지.
   - `mha_with_sele_percent_prefixkv(generate_mapping_list=True)`: 재배치 순서. 이 둘은 공개 함수가 float16 softmax 결과를 value와 곱하고 `cuda:0`에 할당하므로
     GPU에서 float16으로 호출한다. float16 반올림보다 가까운 값의 순서는 비교하지 않는다(그런 쌍이 전체의 20% 미만이어야 한다).
   - 그대로 호출하지 않는 것: `my_pcache_fast.PrefixKVLayer`(생성자가 GPU chunk 캐시와 `.npy` chunk 파일을 만들고, 6절 2번의 인덱스 문제가 있다).
     대신 저장 정합성(K·V·probe 행 정렬, 위치 r = rank r)을 새 구현의 테스트로 고정한다.
2. **prefix 행 기여분** (GPU, gate phase 뒤). gate 이미지 4장 각각에 대해, 저장한 값(증거 파일; 이미지 감사가 store의 바이트와 묶는다)과 같은 턴 1 프롬프트로
   픽셀부터 다시 전체 계산해 캡처한 값이 **모든 원소에서 같아야 한다**(prefix 행 기여분, 재배치 통계 둘 다).
   다른 질문의 요청에서 캡처한 값과의 차이도 함께 계산해 보고한다(기준 아님: 시퀀스 길이가 다르면 BF16 연산의 반올림이 달라질 수 있다).
3. **재배치 정합성** (gate phase). `impress_official_r100`의 생성 토큰열이 모든 요청(147건)에서 `fullload_async`와 같아야 한다.
4. **재현성** (gate phase와 dev phase). `recompute`, `fullload_async`, Ours의 생성 토큰열이 직전 본실행(`runs/mt_gqa_async_20261116T081635Z/main`)의 같은
   (대화, 턴) 기록과 모든 요청에서 같아야 한다. 다르면 멈추고 원인을 보고한다. dev에서는 `impress_async`도 같은 방식으로 대조해 보고한다(이 arm은 통과 기준에 넣지 않는다).
5. **기존 테스트.** `python -m unittest discover -s tests`를 두 환경에서 돌린다. 변경 전부터의 환경별 실패 외에 새 실패가 없어야 한다.
   transformers 5.13.1 환경: `test_qwen25_vision…stock_forward_capture…`, `test_visdial_turn1_piggyback_core…real_tiny_clip…` 2개.
   transformers 4.57.6 환경(실행 환경): GLM-4.6V API가 없어 실패하는 3개(`test_glm46v_runner` setUpClass, `test_glm46v_validation…partial_rope…`, `test_glm46v_vision…sdpa_hook…`).
6. **DEV 40장** (dev phase, 5 arm). 40장 모두 독립 감사 PASS로 commit되어야 한다. 4번 기준이 DEV 전 요청에서 성립해야 한다.
7. 1–6이 모두 PASS면 `gates/admission.json`을 PASS로 쓰고 398장 전체를 4 arm으로 실행한다. `main` stage는 admission 없이는 시작하지 않는다.

추가 확인(통과 기준에 포함): gate phase에서 새 arm의 비동기 읽기와 동기 읽기가 모든 요청에서 같은 생성 토큰열과 같은 읽기 바이트(전체 및 종류별)를 내야 한다.

모든 phase에서 이미지별 독립 감사가 실패하면 그 이미지는 commit되지 않고 실행이 멈춘다. 새 arm에 대한 감사 항목: 선택 파라미터(k, 임계값 0.3),
layer별 결정과 실제 read syscall의 일치(합의: K·V chunk 집합 동일·chunk 정렬·separator chunk 포함·`sim >= 0.3`; fallback: K, V 각 1회 전체 read·`sim < 0.3`),
probe sidecar read가 layer당 1회이고 크기가 probe key + prefix 행 기여분인지, 종류별 바이트 합, 비동기 arm의 워커/요청 스레드 구분, cold 처리,
store의 순서가 증거 파일의 재배치 통계로 다시 계산한 순서와 같은지, 저장된 prefix 행 기여분(디스크에서 다시 읽은 바이트의 SHA)이 증거 파일과 같은지,
통계가 첫 대화 턴 1 요청에서 나왔는지(실행 ID, suffix 토큰 해시, 캡처된 행 수 = prefix 길이 + 그 턴 1 suffix 길이).

## 9. 보고 항목의 정의 (측정하지 않은 값은 "미실행")

- **본실행 표** (T2–T3 followup, arm별): 정답률(strict normalized EM), TTFT 평균·p50·p95, 같은 실행의 FullLoad 및 ReComp 대비 TTFT 평균 비율,
  SSD MB/followup(실제 read syscall이 돌려준 바이트). IMPRESS는 종류별 바이트(probe / K / V; probe에는 prefix 행 기여분 12·N bytes/layer가 포함되며 따로 적는다).
- **이미지 단위 paired bootstrap 95% CI** (10,000회, seed 1234, 요청 가중): Ours−IMPRESS, Ours−FullLoad, Ours−ReComp, IMPRESS−FullLoad의 followup 정답률과 TTFT.
- **DEV 표**: 새 구현과 이전 구현의 정답률, TTFT, 읽기량, fallback 비율, K·V chunk 비율(같은 실행, 같은 이미지·대화). 두 구현의 paired 차이(이미지 단위 bootstrap)도 적는다.
- **IMPRESS 내부 통계** (새 구현, followup 요청):
  - fallback 비율과 평균 Jaccard(전체 및 layer별; layer·요청 평균).
  - K·V chunk 비율: layer당 읽은 chunk 수 / layer chunk 수의 평균(probe sidecar 제외).
  - 중요도 합에서 prefix 행이 차지하는 비중: layer마다 Σ(prefix 행 기여분) / Σ(prefix 행 기여분 + suffix 행 기여분), probe head 3개와 이미지 토큰 전체에 대한 합. 요청 평균.
  - 재배치 순서의 상위 25%와 요청 시 선택이 겹치는 비율: (a) head별 상위 k와 저장 순서의 앞 k개 이미지 토큰의 겹침 / k (모든 layer, head 평균),
    (b) 합의로 선택된 k개와의 겹침 / k (합의 layer만).
  - 계산값(읽지 않은 값, 실측과 구분해 적는다): 같은 선택을 raster 순서에서 읽었다면 건드렸을 chunk 비율; separator도 다른 토큰처럼 중요도 순서에 놓았다면 건드렸을 chunk 비율.
  - GPU가 읽기를 기다린 시간: 요청 스레드가 reader 호출 안에 있었거나 워커의 읽기를 기다린 벽시계 시간(`read_blocked_ms`).
- **저장·턴 1 비용**: 방법별 이미지당 store 크기(구성별), 쓰기 시간, 새 IMPRESS의 prefix 행 기여분·재배치 통계 캡처 시간(턴 1 요청 안)과
  순서 계산·재배치 쓰기 시간(요청 밖).
- 공개 구현의 함수와 새 구현의 대응표, 6절의 "원본과 다른 점" 표.
- 검증 단계별 통과 여부, 예상과 다른 점, 하지 않은 것, 실행 환경과 공개 구현의 커밋.
- 사후 확인(통과 기준 아님, 본실행 뒤): 본실행 398장 전체에서 ReComp·FullLoad·Ours의 생성 토큰열이 직전 본실행과 같은지.

## 10. 저장 계획·삭제 범위·보호

- 디스크 여유 실측(계약 작성 시점): 442.6 GiB (475,248,517,120 bytes).
- FRESH_IMAGE_STREAMING: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch 삭제. 한 이미지의 묶음 상한(최대 N = 2880):
  전체 KV store 최대 3개(FullLoad, 새 IMPRESS, dev에서는 이전 IMPRESS; 각 K/V 1.535 GB + probe 0.072 GB + sidecar·hidden) + packed 0.41 GB ≈ 5.5 GB.
  최소 reserve 30 GiB + 작업 묶음 + margin 4 GiB + raw·evidence 증가 8 GiB가 없으면 `BLOCKED_STORAGE`로 멈춘다.
- 삭제 범위: `runs/<RUN>/scratch/**`(감사+commit 후 정확한 allowlist)뿐이다. 기존 `results/`, `runs/`, store는 수정·삭제하지 않는다.
  freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 소스: freeze 때 `mmimpress/**/*.py`, `scripts/*.py`(보고 스크립트 192 제외), `tests/test_*.py`의 SHA256을 `config.json`에 넣고 각 phase 시작 때 대조한다.
- 커밋과 푸시는 하지 않는다.

## 11. 하지 않는 것

- 기존 이식(`LayerSelector`)의 수정, 25% 외 유지 비율, 다른 임계값·probe head, warm cache, 반복 실행, 다른 모델·데이터셋.
- 공개 구현의 캐시 계층(GPU·DRAM, CKLFU), HyperInfer의 KV 선행 로딩(선택된 토큰의 다음 layer K·V 예측 읽기), 여러 요청에 걸친 재배치 갱신.
- 공개 구현 자체의 실행(OPT 모델로 그 저장소의 실험을 재현하는 것).
