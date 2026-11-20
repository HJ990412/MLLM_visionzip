# LLaVA-1.5-7B × MT-VQA-v2-reconstructed-25k — ReComp / FullLoad / IMPRESS / Ours 실행 계약 (v2)

2026-11-17 UTC, 이 run의 어떤 GPU 측정보다 먼저 고정. 첫 계약(`docs/llava15_mtvqa_impress_contract.md`, run `llava15_mtvqa_impress_20261117T074736Z`)에서
검증 4의 기준만 바꾼 새 계약이다(사용자 결정, 13절). 나머지 조건·arm·표본·보고 항목은 첫 계약과 같다. 저장소 `/home/dblab/hj/mllm_v2` (`.git` 없음; 소스 보호는 SHA256 manifest).
Run root `runs/llava15_mtvqa_impress_20261117T085444Z` (요청 단위 원본·scratch), 결과 `results/llava15_mtvqa_impress_20261117T085444Z`.
이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다. 결과를 본 뒤 기준·표본·arm을 바꾸지 않는다.

## 1. 목적

논문의 비교 baseline인 IMPRESS(FAST'25)와 제안 방법 Ours를 LLaVA-1.5-7B, MT-VQA-v2-reconstructed-25k 조건에서 잰다.
측정 보고이며 GO/NO-GO 판정 규칙은 두지 않는다. 사전 고정하는 것은 검증 단계의 통과 기준(9절)과 보고 항목의 정의(10절)다.
표와 보고서의 방법 이름은 "ReComp", "FullLoad", "IMPRESS", "Ours"만 쓴다. 옵션값은 3절에만 적는다.

## 2. 환경과 모델

- 인터프리터 `/home/dblab/anaconda3/envs/mllm_ft/bin/python` — Python 3.10.20, torch 2.5.1+cu121, **transformers 4.57.6**
  (README 34·35·38절 실행과 같은 환경). 실행 시점 버전과 pip freeze를 `config.json`·`environment_before.json`에 기록하고,
  각 phase 시작 때 일치하지 않으면 중단한다. `.venv-glm46v`(transformers 5.13.1)는 단위 테스트 교차 확인에만 쓴다.
- 모델 `llava-hf/llava-1.5-7b-hf` revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` (34절과 같은 `Llava15Runner`):
  NF4 double quant / BF16 compute / eager attention, FP16 SSD KV, greedy max 16, seed 1234, batch 1, CPU threads 2, RTX 4090 1장 단독 사용.
  visual token 576개 고정, separator 없음, HF 기본 336×336 전처리.

## 3. Arm과 설정값

| 표의 이름 | method id (동기) | method id (비동기) | store 종류 |
|---|---|---|---|
| ReComp | `recompute` | — | 없음 (매 턴 픽셀부터) |
| FullLoad | `fullload` | `fullload_async` | `raster`: 전체 Visual KV, raster 순서 |
| IMPRESS | `impress` | `impress_async` | `impress`: 전체 Visual KV + probe key 사본, 턴 1 중요도 순서로 재배치 |
| Ours | `llava15_kv_PYR-b20_packed` | `llava15_kv_PYR-b20_packed_async` | `packed_PYR-b20`: packed PrefixStore |

- **본실행 arm (이 순서로 고정, 4개)**: `recompute`, `fullload_async`, `impress_async`, `llava15_kv_PYR-b20_packed_async`.
- **IMPRESS** (README 38절 본실행의 구현 그대로, `mmimpress.serve.LayerSelector(mode="impress")`): `importance_metric="h2o"`, `fallback_v="selected"`,
  probe head 3개, α = 0.6, layer마다 25% 유지(k = ceil(0.25·576) = 144), chunk 64 토큰(layer당 9 chunk).
  유사도 임계값 t = j^α, j = r/(2−r) (r = 144/576) → t ≈ 0.3112. 비동기 arm은 여기에 `prefetch="next_layer"`.
  `mmimpress/impress_official.py`는 쓰지 않는다. 6절 참고.
- **Ours**: 34절의 `PYR-b20_packed` 하나. 벡터 문서 `runs/llava15_mt_gqa_20261112T062708Z/vectors/layer_budget_llava15_PYR-b20.json`을 그대로 복사
  (파일 SHA256 `7cc93f8a52646e337568e9a2d1341a204f4089fadc6a82639ffdb71acaea117c`, ratio SHA256 `51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724`,
  `r_min = 0.0`, `r_max = 1.0`, total ratio 0.25). image-only CLIP CLS→patch attention 순서(질문 미사용), k_l = 281, 272, 263, …, 34, 25, 16, 7 (합 4,608 = 32·144),
  packed 파일 1개 75,497,472 bytes. 저장 형식 `llava15_prefixstore_ratio_packed_v1` 그대로. 다른 Ours 변형은 넣지 않는다.
- **FullLoad**: `Server.request(mode="fullload")`. 비동기 arm은 `prefetch="layer_stream"`.
- 검증 전용 arm 두 개는 gate phase에만 있고 어떤 표에도 넣지 않는다: `impress_r100`(IMPRESS 경로, 유지 비율 100%, 동기)과
  `fullload_reordered`(IMPRESS의 재배치 store를 FullLoad loader — `Server.request(mode="fullload")`, 선택·마스크 없음, 동기 — 로 읽음).
  두 arm은 `impress`·`impress_async`와 같은 store(`impress`)를 쓴다.

## 4. ReComp·FullLoad 기록 확인 (2026-11-17 수행)

작업 환경(`/home/dblab/hj/mllm_v2`의 `runs/`·`results/`·`docs/`·`scripts/`, `/home/dblab/hj` 아래 다른 디렉터리)에서 LLaVA-1.5 × MT-VQA 25k 기록을 찾았다.
- LLaVA-1.5 기록은 MT-GQA뿐이다: `runs|results/llava15_mt_gqa_20261112T062708Z`, `…_dev_20261112T0613Z`.
- MT-VQA 25k 기록은 LLaVA-NeXT(v1.6-vicuna)·Qwen·GLM뿐이다: `runs/mt_vqa_v2_reconstructed_25k{,_v2,_v3}_*`, `runs/llava_next_mtvqa_pyr_20261114T182331Z`.
- 첫 run(`runs/llava15_mtvqa_impress_20261117T074736Z`)은 본실행 없이 끝났고, 거기 있는 것은 gate 44장·pilot 40장의 요청뿐이다(25,000장 기록이 아니다).
- **LLaVA-1.5 × MT-VQA 25k의 ReComp·FullLoad 기록은 없다.** 따라서 ReComp와 FullLoad(비동기)를 본실행 arm에 넣어 네 arm으로 돌린다.
  기록 재사용이 없으므로 모든 비교는 같은 실행 안의 값이다. 1,000장 구간(10절)은 본실행의 부분 구간으로 그대로 보고한다.

## 5. 데이터와 절차

- MT-VQA-v2-reconstructed-25k 전체: 25,000 images, 이미지당 dialogue 1개, 3턴 → arm당 75,000요청(followup 50,000).
  `data/mt_vqa_v2_reconstructed_25k/dialogues.json`에서 읽고 ordered workload SHA256 `eb3c5e812fd6937cc0741cac6dd85d6839a7d00288dffbfa7ea8294272269cc8`,
  v3 run(`runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z/llava/main/manifest.json`)의 dialogue 목록·순서와 같은지 freeze에서 확인한다. 25,000개 이미지 파일 SHA256 전부 확인.
- 채점: `mmimpress/mtvqa25k/scorer.py`(GT-Vision-Lab/VQA `a013f0043c1e2cdc995922dfe257f7149aa9af06`), ten-answer leave-one-out 공식 VQA score(%).
  `adapter.enrich`를 그대로 쓴다(공식 evaluator 값과 독립 계산값이 1e-10 이내로 같지 않으면 중단). followup(T2–T3)은 method-local generated history(gold 미사용).
- 프롬프트·history 형식은 34절 LLaVA-1.5 경로(`llava15_support.prompt_factory`)와 같다: `USER: <image>\n[Q1…/A1…]\n\nCurrent question Q{t}: …\nAnswer the current question with a single word or short phrase. ASSISTANT:`.
  LLaVA-NeXT MT-VQA 실행(108/156 `prompt_factory`)과 문구가 같다.
- arm 순서는 dialogue 단위 회전(`method_order(methods, global dialogue ordinal)`), 세 턴 동안 고정. image-atomic attempt, 이미지별 독립 감사 후 commit,
  commit 뒤 run-local scratch store만 정확한 SHA allowlist로 삭제(image-major 수명주기), 요청 원본(`*.jsonl`)은 lossless gzip으로 보존, 중단 시 `--resume`.
- **턴 1은 모든 방법이 픽셀부터 계산한다**(모든 arm). KV·page cache 예열은 없다. 프로세스 시작 때 합성 이미지 1건의 CUDA warm-up 요청이 있으며(34·38절과 동일) 어떤 지표에도 넣지 않는다.
  저장 KV 읽기는 요청마다 직전에 그 store의 모든 `.bin`을 `posix_fadvise(DONTNEED)`로 page cache에서 비운 cold 조건이고, 이 비우기는 타이머 밖이다(파일별 성공을 기록).
- TTFT: 요청 시작(프롬프트 조립, 픽셀 요청이면 이미지 디코드·전처리 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`. 34절 LLaVA-1.5 경로와 같은 경계다. 비동기화로 경계와 cold 처리를 바꾸지 않는다.
- store는 이미지의 (유일한) dialogue 턴 1에서 만든다. store 종류마다, 회전 순서에서 그 종류를 쓰는 첫 arm의 턴 1 캡처로 만든다.
  본실행에서는 store 종류와 arm이 1:1이다. gate phase에서는 동기 arm과 비동기 arm(및 `impress_r100`, `fullload_reordered`)이 **같은 store**를 쓴다.
- 다른 GPU compute 프로세스가 있으면 멈춘다(종료시키지 않는다). 확인은 이미지마다 시작·끝에서 한다.

### Phase

| phase | 이미지 | arm | 요청 수 | 용도 |
|---|---|---|---|---|
| `preserve` | MT-GQA manifest 순서 앞쪽 이미지 2장(`n229548`, `n181355`)의 모든 대화 33개 | 34절의 5 arm | 495 | 검증 2 |
| `gate` | v3 smoke 4장 + main manifest 순서 앞쪽 40장 = 44장 | 9 (7 + `impress_r100` + `fullload_reordered`) | 1,188 | 검증 3·4·5 |
| `pilot` | v3 pilot40의 40장 | 4 (본실행 arm) | 480 | 검증 6 |
| `main` | 25,000장 전체 (manifest 순서) | 4 | 300,000 | 본실행 |

preserve·gate·pilot 요청은 main 분모에 넣지 않는다. gate·pilot 이미지는 main에 다시 포함된다(main은 25,000장 전체가 모집단).
gate·pilot은 구현 확인용이며 main의 arm이나 설정을 고르는 데 쓰지 않는다(이 문서에서 이미 고정).
**1,000장 구간**: main manifest 순서의 앞쪽 1,000장(`global_dialog_ordinal` 0–999). main 실행 안에서 측정된 값을 그대로 쓴다.

## 6. IMPRESS arm의 정의 (38절 구현 그대로, 모델만 LLaVA-1.5)

- **저장**: 턴 1(픽셀부터 계산한 그 요청의 KV 캡처)에서 Visual KV 전체와 probe key 사본(각 layer의 첫 3개 head의 key)을 SSD에 쓴다. 토큰을 버리지 않는다.
  writer는 FullLoad의 raster store와 같은 `mmimpress.llava15.raster.persist_captured_llava15_raster_prefix(probe_heads=3)`
  (`mmimpress.store.write_image_store`를 LLaVA-NeXT raster writer와 같은 인자로 호출; separator가 없어 `sep_kv.bin`은 0 bytes, rater용 `v_hidden.pt`는 쓰지 않는다).
  FullLoad store에도 쓰이지 않는 probe 사본이 들어 있다(38절과 동일; 보고서에 크기를 나눠 적는다).
- **재배치**: 턴 1 질문에 대한 중요도의 내림차순으로 layer별로 정한다. 중요도 = 턴 1 프롬프트의 `<image>` 뒤 suffix 행 전체에 대한 attention 열 합(H2O), head 평균.
  `mmimpress/impress_mt.py`(수정 없음): 방금 쓴 store(FP16)를 전량 읽어 suffix만 한 번 forward해 집계하고, layer별 stable argsort 내림차순으로 K·V·probe key 파일을 다시 쓴다.
  **이후 갱신하지 않는다.**
- **읽기(턴 2·3)**: probe head 3개, layer마다 25% 유지, chunk 64 토큰. query 행은 suffix 전체(이전 턴의 질문·답 + 현재 질문 + 템플릿)다.
  Fallback(probe head 간 평균 Jaccard ≤ t)은 모든 head의 key를 읽고 head별로 고른 뒤, 선택된 토큰(head별 선택의 합집합)을 덮는 V chunk만 읽는다.
  다음 layer의 probe key를 선행 로딩한다(비동기 arm).
- **턴 1 쪽 비용**(중요도 집계, 재배치 쓰기, probe key 쓰기)은 followup TTFT에 넣지 않고 따로 잰다. 중요도 집계는 별도 forward로 재므로 턴 1 추론에 얹어 집계하는 구현보다 크게 잡힐 수 있다.

## 7. 비동기 읽기 (38절과 같은 규칙)

공통: 해당 layer의 질문 연산 결과 없이도 위치가 정해지는 읽기는 백그라운드 워커 1개(`Server._prefetch_pool()`의 단일 스레드, 요청 전에 미리 시작)가 layer 순서대로 미리 읽고,
각 layer는 자기 데이터가 도착하면 적재하고 연산한다. 워커는 디스크 읽기(와 host 버퍼 변환)만 하고 GPU 적재·scatter·마스크는 요청 스레드가 한다.
선행 읽기는 요청의 TTFT 타이머가 시작된 뒤에만 걸린다. 앞서 읽는 깊이 상한은 두지 않는다(`max_ahead = None`; IMPRESS probe는 구현상 한 layer 앞).

- **FullLoad** (`prefetch="layer_stream"`): 워커가 layer 0의 K 전체, V 전체, layer 1의 K, V, … 순서로 읽는다(동기와 같은 64회 pread, 같은 범위).
- **IMPRESS** (`prefetch="next_layer"`): layer 0의 probe key는 hook 안에서 읽고, layer i의 hook이 끝날 때 layer i+1의 probe key 읽기를 워커에 넘긴다.
  선택된 K·V chunk와 fallback의 K 전량·V chunk는 선택 뒤에 hook 안에서 동기로 읽는다.
- **Ours** (`request_llava15_packed(readahead="layer_stream")`, 신규 `Llava15StreamSelector`): 워커가 packed 파일을 [layer 0의 K|V 구간] → … → [layer 31의 K|V 구간] 순서로
  `preadv`해 pinned buffer의 제자리에 채운다(separator가 없으므로 32회; LLaVA-NeXT의 33회와 다르다). layer i의 hook은 자기 구간이 도착하기를 기다렸다가
  동기 경로와 같은 `_scatter_layer`로 GPU cache에 쓰고 마스크를 건 뒤 연산한다. 구간들은 파일 전체를 빈틈·겹침 없이 한 번 덮는다.
- **동기 경로는 기본값으로 남는다.** `request_llava15_packed`의 `readahead` 기본값은 `"none"`(hit당 preadv 1회, 파일 전체)이고 기존 method id의 동작·감사 조건은 그대로다.

비동기 arm의 감사 조건:
- `fullload_async`: pread 64회, 범위·순서가 동기와 같음, 모든 읽기가 워커 스레드에서 타이머 시작 뒤에 발생, 선행 읽기 항목 32개 모두 소비.
- `impress_async`: probe key 읽기 32회 중 layer 0은 요청 스레드, layer 1–31은 워커 스레드. K·V 읽기는 모두 요청 스레드. 종류별 바이트 합 = 전체.
- `…_packed_async`: `preadv`만(pread 0회), 32회, 범위가 [layer 0, …, layer 31] 순서로 파일을 빈틈·겹침 없이 덮음, 반환 바이트 합 = 파일 크기 = 75,497,472, 모든 읽기가 워커 스레드에서 타이머 시작 뒤.
- 공통: 요청이 끝난 뒤 hook·마스크가 남지 않음, 읽은 파일이 모두 직전에 cold 처리됨, vision tower 호출 0회.

## 8. 코드 변경 범위

- 첫 계약 때 만든 코드를 그대로 쓴다. 첫 run 뒤에 바꾼 것은 검증 4의 새 기준에 필요한 것뿐이다: `scripts/193`에 gate 전용 arm `fullload_reordered`와 schema 이름·계약 경로,
  `scripts/194`의 검증 4 비교(`compare_tokens`; 이전 기준의 분류·진단 stage는 삭제), `tests/test_llava15_mtvqa_impress.py`의 해당 테스트, 보고 스크립트 195. `mmimpress/**`는 첫 run과 같다.
- 신규(첫 계약 기준): `mmimpress/llava15/raster.py`, `scripts/193_eval_llava15_mtvqa_impress.py`(freeze·gate·pilot·main 실행), `scripts/194_validate_llava15_mtvqa_impress.py`(검증 단계),
  `scripts/195_report_llava15_mtvqa_impress.py`(보고; 측정에 관여하지 않으므로 소스 freeze에서 제외), `tests/test_llava15_mtvqa_impress.py`, 이 문서.
- 수정(기본 동작 불변, 추가만) `mmimpress/llava15/serve.py`: `Llava15PrefixSelector.prepare`의 layer별 scatter를 `_scatter_layer`로 추출(연산·순서 동일),
  `Llava15StreamSelector`, `request_llava15_packed(readahead=…, max_ahead=…)` 추가. `mmimpress/llava15/__init__.py`는 설명 문자열만.
- 그대로 쓰는 것(수정 없음): `mmimpress/serve.py`(`Server`, `LayerSelector`, `ImageContext`), `mmimpress/impress_mt.py`, `mmimpress/readahead.py`, `mmimpress/store.py`,
  `mmimpress/llava15/{runner,store}.py`, `scripts/llava15_support.py`, `scripts/156…`(수명주기 helper), `mmimpress/mtvqa25k/*`(데이터·채점·gzip 보관), `scripts/142…`의 `bootstrap`.
- 건드리지 않는 것: 기존 `results/`, `runs/`, 기존 store, 기존 스크립트(01–192), LLaVA-NeXT·Qwen·GLM 경로.

## 9. 검증 단계와 통과 기준 (사전 고정; 하나라도 실패하면 본실행 없이 실패 내용만 보고)

1. **CPU 테스트.** `tests/test_llava15_mtvqa_impress.py`(신규)를 포함해 전체를 실행 환경(4.57.6)에서 `python -m unittest discover -s tests`로 돌린다.
   변경 전 기준(2026-11-17 변경 전 실행: 736개 중 error 3개 — `test_glm46v_runner`(setUpClass), `test_glm46v_validation…partial_rope…`, `test_glm46v_vision…sdpa_hook…`, GLM API가 없는 환경 실패) 외에 새 실패가 없어야 한다.
   5.13.1 환경에서도 돌려 기록한다(알려진 환경 실패: `test_qwen25_vision…stock_forward_capture…`, `test_visdial_turn1_piggyback_core…real_tiny_clip…`; 교차 확인이며 그 외 실패가 있으면 실패로 본다).
2. **경로 보존** (`preserve`): 수정된 코드로 34절 실행 경로(`scripts/179`의 `run_image`, 동기 packed + recompute, 5 arm)를 MT-GQA 이미지 2장의 모든 대화에 돌려,
   34절 본실행 기록(`runs/llava15_mt_gqa_20261112T062708Z/main`)의 같은 (대화, arm, 턴)과 생성 토큰열이 모든 요청(495건)에서 같아야 한다.
3. **비동기 정합성** (`gate`): FullLoad, IMPRESS, Ours 각각에서 비동기 arm과 동기 arm의 생성 토큰열과 읽은 바이트 수(전체 및 종류별)가 모든 요청(132건/arm)에서 같아야 한다.
4. **IMPRESS 정합성** (`gate`): 유지 비율 100% IMPRESS(`impress_r100`)의 생성 토큰열이, **같은 재배치 store를 FullLoad loader(선택·마스크 없음)로 읽은 결과**
   (`fullload_reordered`)와 모든 요청(132건)에서 같아야 한다. 두 arm의 입력(프롬프트)도 모든 요청에서 같아야 한다(앞 턴의 토큰이 같으면 저절로 같다).
   하나라도 다르면 FAIL.
   - 참고(판정에 쓰지 않는다): 재배치 store FullLoad(`fullload_reordered`)와 raster store FullLoad(`fullload`)의 생성 토큰이 다른 요청 수를 함께 보고한다.
     그중 두 요청의 입력이 같은 것과, 앞 턴의 답이 달라져 history가 달라진 것을 나눠 적고, 입력이 같은 요청에는 raster FullLoad의 그 위치 상위 두 후보 logit을 붙인다
     (이를 위해 gate phase의 `fullload` 요청에서만 생성 단계마다 logit 최댓값·상위 후보를 기록한다).
5. **턴 1 일치**: gate(9 arm)와 pilot(4 arm)의 모든 이미지에서 모든 arm의 턴 1 생성 토큰열이 같아야 한다. main에서는 일치 건수를 보고한다.
6. **Pilot 40장** (`pilot`, 본조건 4 arm): 40장 모두 독립 감사 PASS로 commit되고 10절의 본실행 표가 pilot에 대해 만들어져야 한다.
   pilot 실측(이미지당 wall time, 보존 원본 크기, scratch 최대 크기)으로 main 예상 소요 시간과 필요 디스크를 계산해 `gates/pilot.json`에 남긴다(첫 run의 pilot 실측: 이미지당 5.4초, 약 37.5시간).
   필요 디스크(30 GiB reserve + 작업 묶음 + pilot 실측 원본 크기 × 625 × 1.35)가 여유보다 크면 FAIL(BLOCKED_STORAGE). 소요 시간에는 기준을 두지 않는다.
7. 1–6이 모두 PASS면 `gates/admission.json`을 PASS로 쓰고 25,000장 전체를 이어서 실행한다. `main` stage는 admission 없이는 시작하지 않는다.

모든 phase에서 이미지별 독립 감사(완전성, ID, 질문·annotation 동결, 공식 VQA score 재계산, history 계보와 프롬프트 재구성, TTFT 재계산, vision 호출 수,
실제 read syscall trace와 바이트 집계 일치, cold 처리, 방법별 I/O 조건 — 7절, IMPRESS는 layer별 결정과 실제 읽은 chunk의 일치, 재배치 순서가 저장된 턴 1 중요도의 내림차순인지 재계산,
Ours는 k_l 독립 재계산·파일 길이)가 실패하면 그 이미지는 commit되지 않고 실행이 멈춘다.

## 10. 보고 항목의 정의 (측정하지 않은 값은 "미실행")

- **본실행 표** (arm별): VQA score(T1, T2, T3, 전체, followup), followup TTFT 평균·p50·p95, SSD MB/followup(실제 read syscall이 돌려준 바이트, MB = 1e6 bytes).
  IMPRESS는 종류별 바이트(probe / K / V).
- **같은 실행 1,000장 구간 표**: 네 방법의 followup TTFT(평균·p50·p95)와 FullLoad 및 ReComp 대비 평균 비율. TTFT 비율 비교는 이 구간의 값으로 한다.
  (전체 25,000장의 같은 비율도 참고로 적는다.)
- **이미지 단위 paired bootstrap 95% CI** (10,000회, seed 1234, `scripts/142`의 `bootstrap` 그대로, pooled sums/counts): Ours−IMPRESS의 followup score와 TTFT.
  Ours−ReComp, Ours−FullLoad, IMPRESS−FullLoad, IMPRESS−ReComp, FullLoad−ReComp도 같은 방식으로 제시하고 "같은 실행, 기록 재사용 없음"을 명시한다.
  CI가 0을 포함하면 "차이 미입증"으로 쓴다(동등성 아님).
- **IMPRESS 내부 통계**: fallback 비율과 평균 Jaccard(전체 및 layer별), K·V chunk 비율(읽은 chunk / layer당 9 chunk), GPU가 읽기를 기다린 시간
  (= 요청 스레드가 reader 호출 안에 있었거나 워커의 읽기를 기다린 벽시계 시간 `read_blocked_ms`; FullLoad·Ours도 같은 정의로 적는다).
  읽기와 연산이 겹치므로 hook 시간과 디스크 시간의 합을 TTFT로 해석하지 않는다고 명시한다.
- **저장·턴 1 비용**: 방법별 이미지당 store 크기(구성별), 쓰기 시간(writer의 store-ready 시점까지; 무결성 해시는 따로), IMPRESS의 중요도 집계·재배치 시간.
- ReComp·FullLoad 기록을 찾았는지와 그 경로(4절), 검증 단계별 통과 여부, 예상과 다른 점, 하지 않은 것, 최대 디스크 사용량, 실행 환경의 transformers 버전, 실제 요청 수·실패·소요 시간.

## 11. 저장 계획·삭제 범위·보호

- 디스크 여유 실측(계약 작성 시점): 441.7 GiB (474,252,947,456 bytes), `/dev/nvme0n1p2`.
- image-major: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch 삭제. 한 이미지의 묶음:
  전체 KV store 2개(FullLoad용, IMPRESS용; 각 K/V 301,989,888 + probe 14,155,776 bytes + sys/meta) + packed 75,497,472 bytes ≈ 0.74 GB (상한 1 GiB로 잡는다).
  gate phase도 store 종류는 3개다. 최소 reserve 30 GiB + 작업 묶음 4 GiB + 원본·evidence 증가분 16 GiB가 없으면 `BLOCKED_STORAGE`로 멈춘다.
- SSD 쓰기량(예상): 이미지당 약 1.03 GB(전체 KV store 2개 + IMPRESS 재배치 재기록 + packed) × 25,084장 ≈ 26 TB. 실행의 성격상 불가피하며 보고서에 적는다.
- 삭제 범위: `runs/<RUN>/scratch/**`(감사+commit 후 정확한 allowlist)와 `runs/<RUN>/preserve/scratch/**`뿐이다. 기존 `results/`, `runs/`, store는 수정·삭제하지 않는다.
  freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 소스: freeze 때 `mmimpress/**/*.py`, `scripts/*.py`(보고 스크립트 195 제외), `tests/test_*.py`의 SHA256을 `config.json`에 넣고 각 phase 시작 때 대조한다.
  변경 전 전체 SHA256 목록과 수정 전 `mmimpress/llava15/serve.py` 사본을 `runs/<RUN>/source_before/`에 둔다.
- 커밋과 푸시는 하지 않는다.

## 12. 하지 않는 것

- Ours의 다른 변형(U25 등), 25% 외 유지 비율, warm cache, 반복 실행(실행 간 변동), LLaVA-NeXT·Qwen·GLM, 다른 데이터셋.
- IMPRESS의 캐시 계층, 여러 질문에 걸친 중요도 갱신, 선택된 K·V chunk나 fallback K 읽기의 비동기화, 두 layer 이상 앞선 probe 선행 읽기, `impress_official.py`의 새 구현.
- 본실행에서의 동기 arm(동기·비동기 TTFT 차이는 gate 44장의 값만 참고로 적는다), O_DIRECT·io_uring, 워커 여러 개, SMART·NAND 수준 측정.

## 13. 첫 계약과의 관계 (공개)

- 첫 계약의 run `runs/llava15_mtvqa_impress_20261117T074736Z`: 검증 1·2·3·5·6 PASS, 검증 4 FAIL → 본실행 미실행. 첫 계약의 검증 4는
  "`impress_r100`의 토큰이 raster store의 `fullload`와 같아야 하고, 불일치는 FullLoad의 상위 두 후보 logit이 정확히 동점인 경우만 허용"이었다.
  132건 중 1건(image 23351, 턴 3)이 동점이 아닌 불일치였다(FullLoad의 두 후보 19.375 / 19.25). 진단에서 재배치 store는 raster 행을 순서만 바꿔 bit 단위로 담고 있었고,
  FullLoad loader를 재배치 store에 써도 `impress_r100`과 같은 토큰이 나왔다(key 순서에 따른 BF16 반올림 차이).
- 이 결과를 보고받은 사용자가 검증 4의 기준을 이 문서 9.4로 바꾸기로 결정했다(2026-11-17). 바뀐 것은 9.4와, 그 비교에 필요한 gate 전용 arm `fullload_reordered` 하나(3절, 5절 phase 표)뿐이다.
  나머지 조건·arm·표본·보고 항목은 첫 계약과 같다. 첫 계약 11절의 probe 사본 크기 오기(28,311,552 → 14,155,776 bytes)는 이 문서에서 바로잡았다.
- 첫 계약 때의 개발 리허설 `runs/llava15_mtvqa_impress_dev_20261117T0900Z`(측정 아님)과 첫 run의 gate·pilot은 이 run과 같은 이미지에서 돌았다.
  그 값은 이 run의 판정과 표에 쓰지 않는다. 이 run은 freeze 뒤 검증 1부터 다시 측정한다. 이 계약에 대한 별도의 개발 리허설은 하지 않는다.
- 첫 run과 개발 리허설의 `runs/`·`results/` 디렉터리는 수정·삭제하지 않는다.
