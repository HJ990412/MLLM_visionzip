# MT-GQA 다중 턴 — ReComp / FullLoad / IMPRESS / Ours, 저장 KV 비동기 읽기 실행 계약

2026-11-16 UTC, 어떤 GPU 측정보다 먼저 고정. 저장소 `/home/dblab/hj/mllm_v2` (`.git` 없음; 소스 보호는 SHA256 manifest).
Run root `runs/mt_gqa_async_20261116T081635Z`, 결과 `results/mt_gqa_async_20261116T081635Z`.
이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다. 결과를 본 뒤 기준·표본·arm을 바꾸지 않는다.

## 1. 목적

텍스트 LLM용 KV 저장 시스템 IMPRESS(FAST'25)를 MLLM의 Visual KV에 그대로 적용한 결과를 MT-GQA 다중 턴 조건에서 재고,
같은 실행 안에서 FullLoad, Ours와 비교한다. 저장 KV를 읽는 세 방법(FullLoad, IMPRESS, Ours)은 모두 읽기와 연산을 겹치는 비동기 방식으로 잰다.
이 실행은 측정 보고이며 GO/NO-GO 판정 규칙은 두지 않는다. 사전 고정하는 것은 검증 단계의 통과 기준(8절)과 보고 항목의 정의(9절)다.

표와 보고서의 방법 이름은 "ReComp", "FullLoad", "IMPRESS", "Ours"만 쓴다. 옵션값은 3절에만 적는다.

## 2. 환경

- 실행 인터프리터: `/home/dblab/anaconda3/envs/mllm_ft/bin/python` — Python 3.10.20, torch 2.5.1+cu121, **transformers 4.57.6**.
  기존 HOLD 실행 두 개(`runs/layer_budget_kv25_v2_20261106T073502Z/hold`, `runs/layer_budget_kv25_pyrbeta_20261110T073604Z/hold`)의
  `environment_before.json`과 같은 버전이다. README의 4.57.6과 일치한다.
- 이전 IMPRESS 단일 턴 작업(`results/impress_fidelity_*`, `results/impress_prefetch_*`)은 `.venv-glm46v`(transformers **5.13.1**)에서 실행됐다.
  이번 실행은 4번 검증(기존 HOLD 기록과 토큰 일치) 때문에 HOLD와 같은 4.57.6 환경을 쓴다. 5.13.1 환경은 단위 테스트 교차 확인에만 쓴다.
- 실행 시점 버전은 `config.json`의 `environment`와 `environment_before.json`(pip freeze 포함)에 기록하고, 각 phase 시작 때 일치하지 않으면 중단한다.
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 / BF16 / eager, FP16 SSD KV, greedy max 16,
  seed 1234, batch 1, CPU threads 2, RTX 4090 1장 단독 사용.

## 3. Arm과 설정값

| 표의 이름 | method id (동기) | method id (비동기) | store 종류 |
|---|---|---|---|
| ReComp | `recompute` | — | 없음 (매 턴 픽셀부터) |
| FullLoad | `fullload` | `fullload_async` | `raster`: 전체 Visual KV, raster 순서 |
| IMPRESS | `impress` | `impress_async` | `impress`: 전체 Visual KV + probe key 사본, 턴 1 중요도 순서로 재배치 |
| Ours | `ours_prefixstore_kv25_PYR-beta-raw_packed` | `ours_prefixstore_kv25_PYR-beta-raw_packed_async` | `packed_PYR-beta-raw`: packed PrefixStore |

- **본실행 arm (이 순서로 고정, 4개)**: `recompute`, `fullload_async`, `impress_async`, `ours_prefixstore_kv25_PYR-beta-raw_packed_async`.
- **Ours**: `runs/layer_budget_kv25_pyrbeta_20261110T073604Z/vectors/layer_budget_llava_PYR-beta-raw.json` 그대로
  (ratio SHA256 `51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724`, 파일 SHA256 `8bfc4cbc55eb778a7ef0433ade4bf13f7a24107b76515f23a72ac01e0740ea5d`,
  `r_min = 0.0`, `r_max = 0.60`). 유지 예산 Σ k_l = 32·ceil(0.25N), image-only VisionZip 순서, packed 파일 1개. 저장 형식과 저장 바이트는 pyrbeta HOLD와 같다.
  균등 배분(U_packed) 등 다른 Ours 변형은 넣지 않는다.
- **IMPRESS**: `LayerSelector(mode="impress")`, `importance_metric="h2o"`, `fallback_v="selected"`, probe head 3개, α = 0.6, layer마다 25% 유지
  (k = ceil(0.25·N_content)), chunk 64 토큰. 비동기 arm은 여기에 `prefetch="next_layer"`. 6절 참고.
- **FullLoad**: `Server.request(mode="fullload")`. 비동기 arm은 `prefetch="layer_stream"`.
- 검증 전용 arm `impress_r100`(IMPRESS 경로, 유지 비율 100%, 동기)은 gate phase에만 있고 어떤 표에도 넣지 않는다.

## 4. 데이터와 절차 (기존 LLaVA-NeXT MT-GQA HOLD와 동일)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm.
  INDEX_SHA `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`, WORKLOAD_SHA `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`.
  dialogues·image identity가 pyrbeta HOLD manifest와 같음을 freeze에서 assert한다.
- 수명주기는 `scripts/164_eval_layer_budget_pyrbeta_hold.py`(= 162/156)의 코드를 import해 그대로 쓴다: method-local generated history, strict normalized EM,
  greedy max 16, 대화 단위 arm 순서 회전(`method_order(methods, 대화 ordinal)`, arm 수에 그대로 적용), image-atomic attempt, 이미지별 독립 감사 후 commit,
  commit 뒤 run-local scratch store만 정확한 SHA allowlist로 삭제, 중단 시 `--resume`.
- **턴 1은 모든 방법이 픽셀부터 계산한다**(모든 대화, 모든 arm). 캐시 예열은 없다. 저장 KV 읽기는 요청마다 직전에 그 store의 모든 `.bin`을
  `posix_fadvise(DONTNEED)`로 page cache에서 비운 cold 조건이고, 이 비우기는 타이머 밖이다.
- TTFT는 기존과 같은 경계다: 요청 시작(프롬프트 조립·토큰화 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`. 비동기화로 이 경계와 cold 처리를 바꾸지 않는다.
- store는 이미지의 **첫 대화 턴 1**에서 만든다. store 종류마다, 그 대화의 회전 순서에서 그 종류를 쓰는 첫 arm의 턴 1 캡처로 만든다.
  본실행에서는 store 종류와 arm이 1:1이라 기존 HOLD의 "자기 arm의 T1으로 자기 store를 만든다"와 같다.
  gate·dev phase에서는 동기 arm과 비동기 arm이 **같은 store**를 쓴다(같은 바이트를 두 방식으로 읽는 비교이기 때문).

### Phase

| phase | 이미지 | 대화 | arm | 요청 수 |
|---|---|---|---|---|
| `gate` | HOLD smoke dialogue 4개가 속한 이미지 4장의 모든 대화 | 49 | 8 (7 + `impress_r100`) | 1,176 |
| `dev` | v2 `data_split.json`의 DEV 40장(모두 MT-GQA 398장 안에 있음)의 모든 MT-GQA 대화 | 378 | 7 | 7,938 |
| `main` | 398장 전체 | 4,061 | 4 | 48,732 |

dev·gate 이미지는 main에 다시 포함된다(main은 398장 전체가 모집단). dev는 구현 확인과 동기·비동기 비교용이며 main의 arm이나 설정을 고르는 데 쓰지 않는다
(main의 arm·설정은 이 문서에서 이미 고정).

## 5. 비동기 읽기

**공통 원칙.** 해당 layer의 질문 연산 결과 없이도 위치가 정해지는 읽기는 백그라운드 워커 1개(`Server._prefetch_pool()`의 단일 스레드, 요청 전에 미리 시작)가
layer 순서대로 미리 읽고, 각 layer는 자기 데이터가 도착하면 적재하고 연산한다. 워커는 디스크 읽기(와 host 버퍼 변환)만 하고, GPU 적재·scatter·마스크는
메인 스레드가 한다. 선행 읽기는 요청의 TTFT 타이머가 시작된 뒤에만 걸린다. 구현은 `mmimpress/readahead.py`의 `LayerReadAhead`.

- **FullLoad** (`prefetch="layer_stream"`): 타이머 시작 직후 워커가 layer 0의 K 전체, V 전체, layer 1의 K, V, … 순서로 `read_full`을 이어서 호출한다
  (동기와 같은 64회 pread, 같은 범위). layer i의 hook은 layer i의 두 블록이 도착하기를 기다렸다가 GPU cache에 쓰고 그 layer를 연산한다.
- **IMPRESS** (`prefetch="next_layer"`, 이미 구현된 것 그대로): layer 0의 probe key는 hook 안에서 읽고, layer i의 hook이 끝날 때 layer i+1의 probe key 읽기를
  워커에 넘긴다. 선택된 토큰의 K·V chunk와 fallback의 K 전량·V chunk는 선택 뒤에 hook 안에서 동기로 읽는다(위치가 질문 연산 결과에 달려 있다).
- **Ours** (`request_packed(readahead="layer_stream")`): 타이머 시작 직후 워커가 packed 파일을 [separator 구간] → [layer 0의 K|V 구간] → … → [layer 31의 K|V 구간]
  순서로 `preadv`해서 pinned buffer의 제자리에 채운다. separator 구간(파일 끝)은 모든 layer가 자기 separator 행을 필요로 하므로 맨 먼저 읽는다.
  layer i의 hook은 (layer 0이면 separator 구간과) 자기 구간이 도착하기를 기다렸다가 동기 경로와 같은 `_scatter_layer`로 GPU cache에 쓰고 마스크를 건 뒤 연산한다.
  구간들은 겹치지 않고 파일 전체를 정확히 한 번 덮는다. 저장 형식·저장 바이트는 그대로이고, pinned buffer 준비는 지금처럼 activation(TTFT 밖)에서 한다.
- **앞서 읽는 깊이 상한: 두지 않는다**(`max_ahead = None`, FullLoad·Ours 공통). 호스트 메모리 125 GB에 대해 FullLoad가 최대로 쌓을 수 있는 양은 이미지 하나의
  전체 KV(약 1.5 GB 이하)이고 Ours는 미리 잡아 둔 pinned buffer에 읽으므로 늘지 않는다. IMPRESS의 probe key 선행 읽기는 구현상 한 layer 앞까지다(과제 지정).
  요청마다 실제로 쌓인 최대 깊이(`peak_outstanding`)를 기록해 보고한다.
- **동기 경로는 기본값으로 남는다.** `Server.request`의 `prefetch` 기본값은 `"none"`, `request_packed`의 `readahead` 기본값은 `"none"`이고, 기존 method id의 동작과
  감사 조건(예: Ours 동기 arm의 "hit당 preadv 1회, 파일 전체")은 그대로다. FullLoad에 `prefetch="next_layer"`를 줘도 동기로 동작하는 기존 계약도 그대로다.
- Python GIL 전환 간격 등 런타임 설정은 건드리지 않는다(기본값).

**비동기 arm의 감사 조건 (새 method id).**
- `fullload_async`: pread 64회, 범위·순서가 동기와 같음, 모든 읽기가 워커 스레드에서, 타이머 시작 뒤에 발생, 선행 읽기 항목 32개 모두 소비.
- `impress_async`: probe key 읽기 32회 중 layer 0은 요청 스레드, layer 1–31은 워커 스레드. K·V 읽기는 모두 요청 스레드. 종류별 바이트 합 = 전체.
- `ours_..._packed_async`: `preadv`만(pread 0회), 호출 수 = 1(separator) + 길이가 0이 아닌 layer 구간 수, 범위가 [separator, layer 0, …, layer 31] 순서로
  파일을 빈틈·겹침 없이 덮음, 반환 바이트 합 = 파일 크기, 모든 읽기가 워커 스레드에서 타이머 시작 뒤에 발생.
- 공통: 요청이 끝난 뒤 hook·마스크가 남지 않음, 읽은 파일이 모두 직전에 cold 처리됨, vision tower 호출 0회.

## 6. IMPRESS arm의 정의

- **저장**: 이미지의 첫 대화 턴 1(픽셀부터 계산한 그 요청의 KV 캡처)에서 Visual KV 전체와 probe key 사본(각 layer의 첫 3개 head의 key)을 SSD에 쓴다.
  토큰을 버리지 않는다. writer는 FullLoad의 raster store와 같은 `persist_captured_raster_prefix(probe_heads=3)`다
  (그래서 FullLoad store에도 쓰이지 않는 probe 사본이 들어 있다. 기존 HOLD와 같으며, 보고서에 크기를 나눠 적는다).
- **재배치**: 저장 순서는 그 턴 1 질문에 대한 중요도의 내림차순으로 layer별로 정한다. 중요도 = 턴 1 프롬프트의 `<image>` 뒤 suffix 행 전체에 대한
  attention 열 합(H2O), head 평균. 구현은 `mmimpress/impress_mt.py`: 방금 쓴 store(FP16)를 전량 읽어 suffix만 한 번 forward해 집계한다
  (`serve.calibrate_image`의 질문 1개 경로와 같은 `Calibrator`, suffix만 실제 턴 1 프롬프트의 것을 쓴다). 순서는 layer별 stable argsort 내림차순
  (`scripts/02_reorder.py --order importance`의 `rewrite_layer`+`commit`과 같은 결과; separator도 다른 토큰처럼 중요도 순서에 놓인다).
  K, V, probe key 파일을 같은 순서로 다시 쓰고 separator sidecar를 다시 만든다. **이후 대화나 별도 보정 질문으로 갱신하지 않는다.**
- **읽기(턴 2·3)**: probe head 3개, α = 0.6, layer마다 25% 유지, chunk 64 토큰. query 행은 suffix 전체(이전 턴의 질문·답 + 현재 질문 + 템플릿)다.
  separator는 기존처럼 예산 밖에서 항상 읽는다(선택된 chunk에 separator가 든 chunk를 더한다).
- **턴 1 쪽 비용**(중요도 집계, 재배치 쓰기, probe key 쓰기)은 followup TTFT에 넣지 않고 따로 잰다. probe key 최초 쓰기 시간은 writer가 파일별로 재지 않으므로
  `Path.write_bytes`를 감싸 파일 이름별로 관측한다(writer 동작은 바꾸지 않음). 재배치는 K / V / probe 파일별로 읽기·치환·쓰기 시간을 잰다.
  중요도 집계는 별도 forward로 재므로 턴 1 추론에 얹어 집계하는 구현보다 크게 잡힐 수 있다는 점을 보고서에 적는다.

## 7. 코드 변경 범위

- 신규: `mmimpress/readahead.py`, `mmimpress/impress_mt.py`, `tests/test_mt_gqa_async.py`, `scripts/187_eval_mt_gqa_async.py`(실행),
  `scripts/188_validate_mt_gqa_async.py`(gate), `scripts/189_report_mt_gqa_async.py`(보고; 측정에 관여하지 않으므로 소스 freeze에서 제외).
- 수정(기본 동작 불변, 추가만):
  - `mmimpress/serve.py`: `PREFETCH_POLICIES`에 `"layer_stream"` 추가; `LayerSelector`에 FullLoad layer stream(`begin_readahead` 등), `max_ahead`,
    요청 스레드가 읽기에 막힌 시간(`read_wait_stats()`, `stats()`의 키는 그대로), layer별 keep mask 참조 보관; `Server.request`에 `max_ahead`, `selector_out`,
    결과 키 `read_wait` 추가.
  - `mmimpress/packed_store.py`: `PackedReader.stream_plan()`, `read_range()` 추가.
  - `mmimpress/packed_serve.py`: `prepare`의 layer별 scatter를 `_scatter_layer`로 추출(연산 동일), `PackedStreamSelector`, `request_packed(readahead=...)` 추가.
- 건드리지 않는 것: QA selector, `sparsevlm_ssd_*`, MPIC, Qwen·GLM, 기존 스크립트(01–186), 기존 `results/`·`runs/`·store.

## 8. 검증 단계와 통과 기준 (사전 고정; 하나라도 실패하면 본실행 없이 실패 내용만 보고)

1. **CPU 테스트.** `tests/test_mt_gqa_async.py`(신규)를 포함해 전체를 `python -m unittest discover -s tests`로 돌린다. 변경 전에 기록한 환경별 기준 실패 외에
   새 실패가 없어야 한다. 기준 실패: 5.13.1 환경은 `test_qwen25_vision…stock_forward_capture…`, `test_visdial_turn1_piggyback_core…real_tiny_clip…` 2개.
   4.57.6 환경(실행 환경)은 GLM-4.6V API가 없어 실패하는 `test_glm46v_runner`(setUpClass), `test_glm46v_validation…partial_rope…`, `test_glm46v_vision…sdpa_hook…` 3개
   (변경 전 693개 중 3개 error; 기존 v2/pyrbeta 실행도 GLM 모듈은 별도 환경으로 기록). 두 환경 모두에서 실행해 기록한다.
2. **비동기 정합성** (gate phase, 49 대화): FullLoad, IMPRESS, Ours 각각에서 비동기 arm과 동기 arm의 생성 토큰열과 읽은 바이트 수(전체 및 종류별)가
   모든 요청(147건/arm)에서 같아야 한다.
3. **재배치 정합성** (gate phase): `impress_r100`의 생성 토큰열이 모든 요청에서 `fullload`와 같아야 한다.
4. **환경 일치** (gate phase): `recompute`, `fullload`, `fullload_async`, Ours 동기·비동기 arm의 생성 토큰열이 기존 HOLD 기록의 같은 (대화, 턴)과 모든 요청에서
   같아야 한다. 기준: ReComp·FullLoad는 `runs/layer_budget_kv25_v2_20261106T073502Z/hold/main`, Ours는 `runs/layer_budget_kv25_pyrbeta_20261110T073604Z/hold/main`의
   `ours_prefixstore_kv25_PYR-beta-raw_packed`(이 둘이 과제의 `results/…` 두 실행의 요청 단위 원본이다). 다르면 멈추고 원인을 보고한다.
5. **DEV 40장 파일럿** (dev phase, 7 arm): (a) 40장 모두 독립 감사 PASS로 commit, (b) 2번 기준이 DEV 전 요청에서 성립, (c) 4번 기준이 DEV 전 요청에서 성립,
   (d) 동기 대비 비동기의 효과: FullLoad, IMPRESS, Ours 각각에서 followup TTFT의 paired 차이(비동기 − 동기)의 이미지 단위 bootstrap(10,000회, seed 1234)
   평균 < 0 이고 95% CI 상한 < 0. (d)가 실패하면 비동기 구현이 의도대로 동작하지 않는 것이므로 본실행을 하지 않는다.
6. 1–5가 모두 PASS면 `gates/admission.json`을 PASS로 쓰고 398장 전체를 4 arm으로 이어서 실행한다. `main` stage는 admission 없이는 시작하지 않는다.

모든 phase에서 이미지별 독립 감사(완전성, ID, 질문·정답 동결, EM 재계산, history 계보와 프롬프트 재구성, TTFT 재계산, vision 호출 수, 실제 read syscall
trace와 바이트 집계 일치, cold 처리, 방법별 I/O 조건 — 5절, IMPRESS는 layer별 결정과 실제 읽은 chunk의 일치, 재배치 순서가 저장된 턴 1 중요도의 내림차순인지
재계산)가 실패하면 그 이미지는 commit되지 않고 실행이 멈춘다.

## 9. 보고 항목의 정의 (측정하지 않은 값은 "미실행")

- **본실행 표** (T2–T3 followup, arm별): 정답률(strict normalized EM), TTFT 평균·p50·p95, 같은 실행의 FullLoad 및 ReComp 대비 TTFT 평균 비율,
  SSD MB/followup(실제 read syscall이 돌려준 바이트). IMPRESS는 종류별 바이트(probe / K / V).
- **이미지 단위 paired bootstrap 95% CI** (10,000회, seed 1234, 요청 가중): Ours−IMPRESS, Ours−FullLoad, Ours−ReComp, IMPRESS−FullLoad의 followup 정답률과 TTFT.
- **DEV 표**: 방법별 동기·비동기 TTFT와 paired 차이, GPU가 읽기를 기다린 시간(= 요청 스레드가 reader 호출 안에 있었거나 워커의 읽기를 기다린 벽시계 시간
  `read_blocked_ms`; Ours 동기는 `packed_read_ms`). 읽기와 연산이 겹치므로 hook 시간과 디스크 시간의 합을 TTFT로 해석하지 않는다고 명시한다.
- **IMPRESS 내부 통계**: fallback 비율과 평균 Jaccard(전체 및 layer별), K·V chunk 비율, 그리고 같은 선택을 재배치 없는 순서에서 읽었다면 건드렸을 chunk 비율
  (요청의 keep mask를 원래 위치로 되돌려 계산한 값; 읽지 않은 계산값이며 실측과 구분해 적는다).
- **저장·턴 1 비용**: 방법별 이미지당 store 크기(구성별), 쓰기 시간, IMPRESS의 중요도 집계·재배치 시간, 기존 보고서(165)와 같은 방식의 3턴 세션 시간
  (shared: 이미지의 첫 대화에만 store 구축+activation을 더함 / standalone: 모든 대화에 더함, DERIVED).
- 검증 단계별 통과 여부, 예상과 다른 점, 하지 않은 것, 최대 디스크 사용량, 실행 환경의 transformers 버전.

## 10. 저장 계획·삭제 범위·보호

- 디스크 여유 실측(계약 작성 시점): 444.4 GiB (477,182,754,816 bytes).
- FRESH_IMAGE_STREAMING: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch 삭제. 한 이미지의 묶음 상한(최대 N = 2880):
  전체 KV store 2개(FullLoad용, IMPRESS용; 각 K/V 1.535 GB + probe 0.072 GB + sidecar·hidden) + packed 0.41 GB + 재배치 중 임시 layer 파일 ≈ 3.8 GB.
  최소 reserve 30 GiB + 작업 묶음 + margin 4 GiB + raw·evidence 증가 8 GiB가 없으면 `BLOCKED_STORAGE`로 멈춘다(`storage_plan.json`).
  이미지별 scratch 최대 사용량을 `disk_usage.json`에 기록하고 전체 최댓값을 보고한다.
- 삭제 범위: `runs/<RUN>/scratch/**`(감사+commit 후 정확한 allowlist)뿐이다. 기존 `results/`, `runs/`, store는 수정·삭제하지 않는다.
  freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 소스: freeze 때 `mmimpress/**/*.py`, `scripts/*.py`(보고 스크립트 189 제외), `tests/test_*.py`의 SHA256을 `config.json`에 넣고 각 phase 시작 때 대조한다.
- 커밋과 푸시는 하지 않는다.

## 11. 하지 않는 것

- Ours의 다른 변형, 25% 외 유지 비율, warm cache, 반복 실행(실행 간 변동), Qwen·GLM, 다른 데이터셋.
- IMPRESS의 캐시 계층, 여러 질문에 걸친 중요도 갱신, 선택된 K·V chunk나 fallback K 읽기의 비동기화, 두 layer 이상 앞선 probe 선행 읽기.
- O_DIRECT·io_uring 등 다른 I/O 인터페이스, 워커 여러 개.
