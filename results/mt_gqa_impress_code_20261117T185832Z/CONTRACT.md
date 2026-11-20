# 실험 A — LLaVA-NeXT × MT-GQA: ReComp / FullLoad / IMPRESS / Ours (IMPRESS = 공개 구현 방식) 실행 계약

2026-11-17 UTC, 이 run의 어떤 GPU 측정보다 먼저 고정. 저장소 `/home/dblab/hj/mllm_v2`.
Run root `runs/mt_gqa_impress_code_<UTC>`, 결과 `results/mt_gqa_impress_code_<UTC>` (같은 `<UTC>`).
이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다. 결과를 본 뒤 기준·표본·arm을 바꾸지 않는다.

## 1. 목적과 배경

IMPRESS(FAST'25)와 그 확장판 HyperInfer(IEEE TC'26) 논문은 중요도를 "attention 행렬의 열 합"이라고만 적고 어느 행을 더하는지 밝히지 않는다.
그래서 baseline의 정의를 저자들의 공개 구현(https://github.com/ISCS-ZJU/HyperInfer, 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`)으로 정한다.
이 실험의 IMPRESS는 공개 구현 방식 하나뿐이다. 질문 토큰의 attention만 쓰는 이전 구현(README 38절의 IMPRESS arm)은 arm에 넣지 않는다.
README 38절 본실행과 같은 조건에서 ReComp, FullLoad, IMPRESS, Ours를 한 실행 안에서 잰다. 측정 보고이며 GO/NO-GO 판정 규칙은 없다.
사전 고정하는 것은 검증 단계의 통과 기준(8절)과 보고 항목의 정의(9절)다.

표와 보고서의 방법 이름은 "ReComp", "FullLoad", "IMPRESS", "Ours"만 쓴다. 옵션값은 3절에만 적는다.

## 2. 환경과 참조 구현

- 인터프리터 `/home/dblab/anaconda3/envs/mllm_ft/bin/python` — Python 3.10.20, torch 2.5.1+cu121, **transformers 4.57.6** (README 38절 본실행과 같다).
  실행 시점 버전과 pip freeze는 `environment_before.json`에 기록하고 각 phase 시작 때 대조한다. `.venv-glm46v`(5.13.1)는 단위 테스트 교차 확인에만 쓴다.
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 / BF16 / eager, FP16 SSD KV, greedy max 16,
  seed 1234, batch 1, CPU threads 2, RTX 4090 1장 단독 사용.
- 공개 구현 체크아웃: 레포 바깥 `/home/dblab/hj/_external_ref/HyperInfer` (읽기 전용, 환경변수 `IMPRESS_OFFICIAL_REPO`). 최상위에 라이선스 파일이 없어 코드를 복사하지 않는다.
  freeze가 체크아웃의 HEAD가 위 커밋인지 확인하고 `config.json`에 기록한다.

## 3. Arm과 설정값

| 표의 이름 | method id (본실행, 비동기 읽기) | 동기 쌍 (gate 전용) | store 종류 |
|---|---|---|---|
| ReComp | `recompute` | — | 없음 (매 턴 픽셀부터) |
| FullLoad | `fullload_async` | `fullload` | `raster`: 전체 Visual KV, raster 순서 |
| IMPRESS | `impress_official_async` | `impress_official` | `impress_official`: 전체 Visual KV + probe key 사본, 턴 1 통계로 재배치 |
| Ours | `ours_prefixstore_kv25_PYR-beta-raw_packed_async` | `ours_prefixstore_kv25_PYR-beta-raw_packed` | `packed_PYR-beta-raw` |

- **본실행 arm (이 순서로 고정, 4개)**: `recompute`, `fullload_async`, `impress_official_async`, `ours_prefixstore_kv25_PYR-beta-raw_packed_async`.
  README 38절 본실행의 arm 순서에서 IMPRESS 자리만 바뀐다. 대화 단위 순서 회전(`method_order`)은 그대로다.
- ReComp, FullLoad, Ours의 코드·설정은 README 38절 본실행과 같다(`mmimpress/serve.py`, `packed_serve.py` 등은 수정하지 않는다).
  Ours는 `runs/layer_budget_kv25_pyrbeta_20261110T073604Z/vectors/layer_budget_llava_PYR-beta-raw.json` 그대로다.
  비동기 읽기: FullLoad는 워커가 layer 순서로 K·V 전체(`layer_stream`), Ours는 워커가 packed 파일을 구간 순서로(`layer_stream`).
- **IMPRESS 설정값**: probe head 0·1·2, 유지 25% (k = ceil(0.25·N_image)), 유사도 임계값 0.3 (`sim >= 0.3`이면 합의), chunk 64 토큰,
  다음 layer probe sidecar 선행 읽기(`next_layer`, 비동기 arm). 공개 구현의 `scripts/lm_performance_fast_pcache.sh`(`sele_percents` 25, `sim_threds` 0.3),
  `scripts/figure15.sh`(`run_sele_load_by_percent 25 0.3`), `--sele-head` 기본값 `0 1 2`, `--chunk-size` 기본값 64에서 가져온 값이다.
- **검증 전용 arm** (gate phase에만, 어떤 표에도 넣지 않는다): 위의 동기 쌍 세 개, `impress_official_r100`(IMPRESS 경로, 유지 비율 100%, 동기),
  `fullload_reordered`(IMPRESS의 재배치 store를 FullLoad loader — `Server.request(mode="fullload")`, 선택·mask 없음, 동기 — 로 읽음).
  `impress_official`, `impress_official_async`, `impress_official_r100`, `fullload_reordered`는 같은 store를 쓴다.

## 4. 데이터와 절차 (README 38절 본실행과 동일)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm. dialogues와 image identity가 README 38절 본실행의 manifest와 같음을 freeze에서 assert한다.
- 수명주기는 README 38절의 것(`scripts/187`, 그 안에서 `scripts/164`를 import)이다: method-local generated history, strict normalized EM, greedy max 16,
  대화 단위 arm 순서 회전, image-atomic attempt, 이미지별 독립 감사 후 commit, commit 뒤 run-local scratch store만 정확한 SHA allowlist로 삭제, 중단 시 `--resume`.
  실행 스크립트 `scripts/196`은 `scripts/190`(= `187` + 새 store 종류)에서 이전 구현의 arm·분기를 빼고 gate 전용 arm을 더한 것이다.
- **턴 1은 모든 방법이 픽셀부터 계산한다.** 저장 KV 읽기는 요청 직전에 그 store의 모든 `.bin`을 `posix_fadvise(DONTNEED)`로 비운 cold 조건이고 비우기는 타이머 밖이다.
  캐시 계층은 없다(모든 방법이 cold SSD). TTFT 경계: 요청 시작(프롬프트 조립·토큰화 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`.
- store는 이미지의 **첫 대화 턴 1**에서, store 종류마다 그 대화의 회전 순서에서 그 종류를 쓰는 첫 arm의 턴 1 캡처로 만든다.

| phase | 이미지 | 대화 | arm | 요청 수 |
|---|---|---|---|---|
| `gate` | README 38절과 같은 smoke 이미지 4장의 모든 대화 | 49 | 9: 본실행 4 + 동기 쌍 3 + `impress_official_r100` + `fullload_reordered` | 1,323 |
| `dev` | v2 `data_split.json`의 DEV 40장의 모든 MT-GQA 대화 | 378 | 4 (본실행 arm, 본조건) | 4,536 |
| `main` | 398장 전체 | 4,061 | 4 | 48,732 |

gate·dev 요청은 main 분모에 넣지 않는다. gate·dev 이미지는 main에 다시 포함된다. dev는 구현 확인용이며 main의 arm이나 설정을 고르는 데 쓰지 않는다.

## 5. IMPRESS의 정의

구현: `mmimpress/impress_official.py` (수정 없음; LLaVA-1.5 실험과 같은 코드). `OfficialImpressSelector`, `request_official`, `Turn1AttentionStats`, `write_official_layout`.

- **중요도**: probe head 0·1·2에 대해 전체 행(prefix + suffix)의 attention 열 합.
  - prefix 행(시스템 프롬프트 + 이미지 위치의 행)의 기여분은 이미지의 **첫 대화 턴 1** 요청의 prefill에서 모델이 계산한 attention 확률을 읽어 구한다(별도 forward 없음, float32 합).
    causal mask 때문에 질문과 무관하다. probe key 파일 끝에 저장 순서로 붙여 둔다(little-endian float32, 3 × N_visual).
  - 요청 시 suffix 행(이전 턴의 질문·답 + 현재 질문 + 템플릿 전체)의 기여분을 probe head의 query와 key(시스템 프롬프트 key, probe key, suffix key)의 causal softmax(float32)로 계산해 더한다.
- **선택**: head별 상위 k → 세 head 쌍의 평균 pairwise Jaccard → 0.3 이상이면 득표수 상위 k개(합의), 미만이면 fallback.
  선택 대상은 이미지 토큰이다. 시스템 프롬프트와 separator(AnyRes 행 구분 토큰)는 항상 유지하고 예산에 넣지 않는다.
  득표수가 같으면 저장 위치가 앞선 토큰을 고른다(공개 구현은 unstable `argsort`라 순서가 정해져 있지 않다).
- **Fallback**: 그 layer의 Visual K·V를 전부 읽고 mask 없이 모두 쓴다.
- **저장·로드**: Visual KV 전체와 probe key 사본(head 0·1·2의 key)을 FullLoad와 같은 raster writer로 쓴 뒤, layer별로
  [separator(원래 순서) | 이미지 토큰을 재배치 통계의 내림차순]으로 K, V, probe key 파일을 다시 쓴다(K·V 분리 저장).
  재배치 통계 = 첫 대화 턴 1 요청 하나의 prefill에서 모든 head·모든 행(prefix + 그 턴 1 suffix)의 열 합(공개 구현 `--generate-mapping-list`가 정렬하는 값). 이후 갱신하지 않는다.
  합의 layer는 선택된 토큰 또는 separator가 속한 64 토큰 chunk의 K와 V를 읽고(인접 chunk는 한 번의 pread로), 선택 안 된 이미지 토큰은 mask한다(읽힌 chunk 안의 것 포함).
- **선행 로딩**: 비동기 arm은 layer 0의 probe sidecar는 hook 안에서, layer i+1의 것은 layer i의 hook이 끝날 때 워커 1개가 읽는다(구현에 이미 있음: `prefetch="next_layer"`).
  선택된 K·V chunk와 fallback 읽기는 선택 뒤 hook 안에서 동기로 읽는다. 디코딩 단계에서는 읽기도 선택도 다시 하지 않는다.
- 캡처 시간은 그 턴 1 요청의 TTFT·E2E 안에 들어간다(그 요청 하나). 재배치 쓰기는 요청 밖이며 따로 잰다. 둘 다 followup TTFT에는 들어가지 않는다.

## 6. 원본과 다른 점 (보고서에 같은 표를 싣는다)

"원본"은 공개 구현(커밋 91df7a2)의 동작이다. 논문은 중요도를 attention 열 합이라고만 적으므로, 논문이 밝히지 않은 부분은 공개 구현을 원본으로 본다.
"방향"은 IMPRESS(baseline)의 TTFT·정답률에 유리한지, 불리한지, 중립인지다.

| # | 항목 | 원본 | 이 실험 | 방향 |
|---|---|---|---|---|
| 1 | 선택 대상 | prefix 토큰 전부 | 이미지 토큰만. 시스템 프롬프트와 separator는 항상 유지, 예산 밖 | 정답률에 유리 / 읽기량에 소폭 불리 |
| 2 | prefix 행 기여분 | 정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태) | 턴 1(mask 없음)의 값을 저장해 요청 때 더함 | 유리 (요청 때 prefix 행 연산 없음) |
| 3 | 캐시 계층 | 논문·공개 구현: GPU·DRAM·SSD 다층 캐시 | 없음. 모든 방법이 cold SSD | 불리 (모든 방법 공통 조건) |
| 4 | 다음 layer probe 선행 읽기 | 공개 구현의 IMPRESS 설정은 `--no-prefetch` | 켬 | 유리 |
| 5 | separator의 저장 위치 | 해당 없음(텍스트 prefix) | 저장 순서 맨 앞에 모음(항상 읽으므로 chunk 1개에 담김) | 유리 |
| 6 | 득표 동점 | 순서 미정의 | 저장 위치가 앞선 토큰 우선 | 유리 또는 중립 |
| 7 | 재배치 통계의 추론 상태 | head별 25% mask 추론, 같은 prefix의 마지막 요청 | mask 없는 첫 대화 턴 1 요청 하나 | 중립 |
| 8 | 재배치 적용 | `key[reorder]` (token→rank 인덱스를 그대로 적용해 내용이 어긋남) | 위치 r = rank r 토큰, K·V·probe 같은 순서 | 중립 (접근 위치는 같다) |
| 9 | 중요도 수치 정밀도 | float16 softmax·합 | BF16 확률 + float32 합 / float32 softmax | 중립 |
| 10 | Fallback 읽기 단위 | chunk 파일별 | layer의 K 파일 1회, V 파일 1회 pread | 유리 (호출 수 적음, 바이트 같음) |
| 11 | 선택 chunk 읽기 단위 | chunk 파일별 | 인접한 chunk는 pread 하나로 합침 | 유리 |
| 12 | prefix 행 기여분 읽기 | 없음 | probe sidecar 끝 12·N bytes/layer를 같은 read로 | 불리 (probe 읽기 +1.6%) |
| 13 | mask | 정확도 경로에만 있음(성능 경로는 0으로 채우고 답을 내지 않음) | mask (한 경로에서 정답률과 시간을 함께 잼) | 중립 |
| 14 | 모델·생성 | OPT(FP16) 텍스트 LLM, 1토큰 logits | LLaVA-NeXT(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정 | 중립 |
| 15 | HyperInfer의 KV 선행 로딩(선택된 토큰의 다음 layer K·V 예측 읽기) | HyperInfer 확장 | 없음 (IMPRESS의 probe 선행 읽기만) | 불리 (HyperInfer 대비; IMPRESS 정의에는 해당 없음) |

## 7. 코드 변경 범위

- 신규: `scripts/196_eval_mt_gqa_impress_code.py`(실행), `scripts/197_validate_mt_gqa_impress_code.py`(검증), `scripts/198_report_mt_gqa_impress_code.py`(보고; 측정에 관여하지 않으므로 소스 freeze에서 제외),
  `tests/test_impress_code_runs.py`, 이 문서. 실험 B용 `scripts/199`·`200`(·`201` 보고)도 같은 시점에 만들어져 소스 freeze에 함께 들어간다.
- 수정 없음: `mmimpress/**`(`impress_official.py` 포함), 기존 스크립트 01–195, 기존 테스트. 기존 `results/`, `runs/`, store도 건드리지 않는다.

## 8. 검증 단계와 통과 기준 (사전 고정; 하나라도 실패하면 본실행 없이 실패 내용만 보고)

1. **공개 구현과의 동등성** (CPU; `tests/test_impress_official.py`와 `tests/test_impress_code_runs.py`의 `PublicEquivalenceTests`, `IMPRESS_OFFICIAL_REPO` 지정).
   같은 텐서를 공개 구현의 함수(`_sele_important_tokens`, `cal_jaccard_sim`, `sele_tokenid`)와 우리 구현에 넣어 head별 상위 k, 평균 Jaccard, fallback 판정, 선택 집합이 같아야 한다.
   통과 기준: 두 파일의 모든 `PublicEquivalenceTests` case가 skip 없이 통과. 득표 동점이 경계에 걸린 입력은 공개 구현의 순서가 미정의이므로 득표수 조건
   (양쪽 모두 k개, 선택 밖 토큰의 득표가 선택 안 토큰보다 많지 않음, 선택된 토큰들의 득표수 multiset 동일)으로 비교한다.
   (공개 구현의 mask·재배치 함수 대조 case는 GPU가 필요하다. `PrefixKVLayer`는 그대로 호출하지 않는다 — 6절 8번.)
2. **prefix 행 기여분** (GPU, gate 뒤). gate 이미지 4장 각각에 대해, 저장한 값(증거 파일; 이미지 감사가 store의 바이트와 묶는다)과 같은 턴 1 프롬프트로
   픽셀부터 다시 전체 계산해 캡처한 값이 **모든 원소에서 같아야 한다**(prefix 행 기여분, 재배치 통계 둘 다).
3. **정합성** (gate). `impress_official_r100`의 생성 토큰열이, 같은 재배치 store를 FullLoad loader(선택·mask 없음)로 읽은 `fullload_reordered`와
   모든 요청(147건)에서 같아야 한다(프롬프트도 같아야 한다).
   참고(판정에 쓰지 않는다): `fullload_reordered`와 raster store의 `fullload`의 생성 토큰이 다른 요청 수를 보고한다.
4. **비동기 정합성** (gate). FullLoad, IMPRESS, Ours 각각에서 비동기 arm과 동기 arm의 생성 토큰열과 읽은 바이트 수(전체 및 종류별)가 모든 요청(147건)에서 같아야 한다.
5. **재현성** (gate와 dev). `recompute`, `fullload_async`, Ours(비동기)의 생성 토큰열이 README 38절 본실행 기록(`runs/mt_gqa_async_20261116T081635Z/main`)의
   같은 (대화, 턴)과 모든 요청에서 같아야 한다. 그리고 각 대화에서 모든 arm의 턴 1 생성 토큰열이 같아야 한다(gate 9 arm, dev 4 arm).
6. **기존 테스트.** `python -m unittest discover -s tests`를 두 환경에서 돌린다. 환경 문제로 실패하던 것 외에 새 실패가 없어야 한다.
   4.57.6(실행 환경): GLM-4.6V API가 없어 실패하는 3개(`test_glm46v_runner` setUpClass, `test_glm46v_validation…partial_rope…`, `test_glm46v_vision…sdpa_hook…`).
   5.13.1: `test_qwen25_vision…stock_forward_capture…`, `test_visdial_turn1_piggyback_core…real_tiny_clip…` 2개.
7. **DEV 40장** (dev, 본조건 4 arm). 40장 모두 독립 감사 PASS로 commit되어야 하고 5번 기준이 DEV 전 요청에서 성립해야 한다.
8. 1–7이 모두 PASS면 `gates/admission.json`을 PASS로 쓰고 398장 전체를 4 arm으로 실행한다. `main` stage는 admission 없이는 시작하지 않는다.
   실패하면 본실행 없이 보고하고 실험 B로 넘어간다.

모든 phase에서 이미지별 독립 감사가 실패하면 그 이미지는 commit되지 않고 실행이 멈춘다. IMPRESS에 대한 감사 항목: 선택 파라미터(k, 임계값 0.3),
layer별 결정과 실제 read syscall의 일치(합의: K·V chunk 집합 동일·chunk 정렬·separator chunk 포함·`sim >= 0.3`; fallback: K, V 각 1회 전체 read·`sim < 0.3`),
probe sidecar read가 layer당 1회이고 크기가 probe key + prefix 행 기여분인지, 종류별 바이트 합, 비동기 arm의 워커/요청 스레드 구분, cold 처리,
store의 순서가 증거 파일의 재배치 통계로 다시 계산한 순서와 같은지, 저장된 prefix 행 기여분(디스크에서 다시 읽은 바이트의 SHA)이 증거 파일과 같은지,
통계가 첫 대화 턴 1 요청에서 나왔는지(실행 ID, suffix 토큰 해시, 캡처된 행 수 = prefix 길이 + 그 턴 1 suffix 길이).

사후 확인(통과 기준 아님, 본실행 뒤): 398장 전체에서 5번의 두 항목을 다시 세어 보고한다.

## 9. 보고 항목의 정의 (측정하지 않은 값은 "미실행")

- **본실행 표** (T2–T3 followup, arm별): 정답률(strict normalized EM), TTFT 평균·p50·p95, 같은 실행의 FullLoad 및 ReComp 대비 TTFT 평균 비율,
  SSD MB/followup(실제 read syscall이 돌려준 바이트, MB = 1e6 bytes). IMPRESS는 종류별 바이트(probe / K / V; probe에는 prefix 행 기여분 12·N bytes/layer가 포함된다).
- **이미지 단위 paired bootstrap 95% CI** (10,000회, seed 1234, 요청 가중): Ours−IMPRESS, Ours−FullLoad, Ours−ReComp, IMPRESS−FullLoad의 followup 정답률과 TTFT.
  CI가 0을 포함하면 "차이 미입증"으로 쓴다(동등성 아님).
- **IMPRESS 내부 통계** (followup 요청): fallback 비율과 평균 Jaccard(전체 및 layer별; layer·요청 평균), K·V chunk 비율(layer당 읽은 chunk 수 / layer chunk 수의 평균, probe sidecar 제외),
  중요도 합에서 prefix 행이 차지하는 비중(layer마다 Σ prefix 행 기여분 / Σ(prefix + suffix), probe head 3개와 이미지 토큰 전체에 대한 합, 요청 평균),
  GPU가 읽기를 기다린 시간(요청 스레드가 reader 호출 안에 있었거나 워커의 읽기를 기다린 벽시계 시간 `read_blocked_ms`; FullLoad·Ours도 같은 정의로 적는다).
  읽기와 연산이 겹치므로 hook 시간과 디스크 시간의 합을 TTFT로 해석하지 않는다.
- **저장·턴 1 비용**: 방법별 이미지당 store 크기(구성별), 쓰기 시간, IMPRESS의 턴 1 추가 비용(턴 1 요청 안의 attention 통계 캡처 시간, 요청 밖의 순서 계산·재배치 쓰기 시간).
- 6절의 "원본과 다른 점" 표, 중지한 실행의 상태, 검증 단계별 통과 여부, 예상과 다른 점, 하지 않은 것, 실행 환경과 공개 구현의 커밋, 디스크 여유와 SSD 쓰기량(예상·실측).

## 10. 저장 계획·SSD 쓰기량·삭제 범위·보호

- 디스크 여유 실측(계약 작성 시점): 439.4 GiB (471,748,730,880 bytes), `/dev/nvme0n1p2`.
- FRESH_IMAGE_STREAMING: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch 삭제. 한 이미지의 묶음 상한(최대 N = 2880):
  전체 KV store 2개(FullLoad, IMPRESS; 각 K/V 1.535 GB + probe 0.072 GB + sidecar·hidden) + packed 0.41 GB ≈ 3.9 GB(계획 파일의 상한은 이전 실행의 3개 기준 5.5 GB를 유지).
  최소 reserve 30 GiB + 작업 묶음 + margin 4 GiB + raw·evidence 증가 8 GiB가 없으면 `BLOCKED_STORAGE`로 멈춘다.
- **예상 SSD 쓰기량** (README 38절 본실행이 같은 이미지에서 잰 store 크기로 계산; 이미지당 raster store 1회 + IMPRESS store 2회(raster 쓰기 + 재배치 재기록) + packed 1회):
  gate 17.0 GB + dev 163.7 GB + main 1,633.8 GB = **약 1.81 TB** (요청 원본·증거 파일 수 GB는 별도). freeze가 같은 계산을 `storage_plan.json`에 기록한다.
  실측은 `/sys/block/nvme0n1/stat`의 쓰기 섹터 수 차이(장치 전체)로 보고한다.
- 삭제 범위: `runs/<RUN>/scratch/**`(감사+commit 후 정확한 allowlist)뿐이다. 기존 `results/`, `runs/`, store는 수정·삭제하지 않는다.
  freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 소스: freeze 때 `mmimpress/**/*.py`, `scripts/*.py`(보고 스크립트 198·201 제외), `tests/test_*.py`의 SHA256을 `config.json`에 넣고 각 phase 시작 때 대조한다.
- 커밋과 푸시는 하지 않는다.

## 11. 하지 않는 것

- 이전 IMPRESS 구현(suffix 행만 쓰는 `LayerSelector`)의 arm, 25% 외 유지 비율, 다른 임계값·probe head, warm cache, 반복 실행, 다른 모델·데이터셋(실험 B는 별도 계약).
- 공개 구현의 캐시 계층(GPU·DRAM, CKLFU), HyperInfer의 KV 선행 로딩, 여러 요청에 걸친 재배치 갱신, 공개 구현 자체의 실행(OPT 모델).
- 본실행에서의 동기 arm, O_DIRECT·io_uring, 워커 여러 개, SMART·NAND 수준 측정.

## 12. 이전 실행과의 관계 (공개)

- `runs/mt_gqa_impress_official_20261117T052201Z`(계약 `docs/mt_gqa_impress_official_contract.md`)는 같은 IMPRESS 구현으로 gate 4장과 DEV 40장까지 돌았고,
  그 계약의 검증 3번("유지 100% IMPRESS == raster store의 FullLoad, 전 요청")이 147건 중 1건의 BF16 logit 동점으로 FAIL이라 398장 본실행은 하지 않았다.
  그 DEV 결과(IMPRESS 61.38% / TTFT 738 ms / fallback 52.4%, 40장)는 이 계약을 쓰기 전에 알려져 있었다.
- 이 계약의 arm, 설정값, 검증 3번의 기준(같은 재배치 store를 FullLoad loader로 읽은 결과와 비교; raster store와의 차이는 참고로만 보고)은 사용자가 지정한 것이다(2026-11-18 KST 작업 지시).
  IMPRESS의 구현(`mmimpress/impress_official.py`)과 설정값은 그 실행과 같다. 이전 실행의 값은 이 run의 판정과 표에 쓰지 않는다.
- 이 계약에 대한 개발 리허설(`runs/mt_gqa_impress_code_dev_<UTC>`, gate phase만, 측정 아님)을 freeze 전에 한 번 돌려 스크립트 오류를 확인한다. 그 값도 판정과 표에 쓰지 않는다.
- 같은 날 중지한 LLaVA-1.5 × MT-VQA 실행(`runs/llava15_mtvqa_impress_20261117T085444Z`, 5,956/25,000장에서 중지)은 실험 B의 계약에 적는다.
