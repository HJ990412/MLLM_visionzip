# 실험 B — LLaVA-1.5-7B × MT-VQA-v2-reconstructed-25k: ReComp / FullLoad / IMPRESS / Ours (IMPRESS = 공개 구현 방식) 실행 계약

2026-11-17 UTC, 이 run의 어떤 GPU 측정보다 먼저 고정. 저장소 `/home/dblab/hj/mllm_v2`.
Run root `runs/llava15_mtvqa_impress_code_<UTC>` (요청 단위 원본·scratch), 결과 `results/llava15_mtvqa_impress_code_<UTC>` (같은 `<UTC>`).
이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다. 결과를 본 뒤 기준·표본·arm을 바꾸지 않는다.
조건은 `docs/llava15_mtvqa_impress_contract.md`(및 그 v2)와 같되 IMPRESS만 공개 구현 방식으로 바꾼다. 바뀐 곳은 3·6·8·9절이다.

## 1. 목적과 배경

IMPRESS(FAST'25)와 HyperInfer(IEEE TC'26) 논문은 중요도를 "attention 행렬의 열 합"이라고만 적고 어느 행을 더하는지 밝히지 않는다.
그래서 baseline의 정의를 저자들의 공개 구현(https://github.com/ISCS-ZJU/HyperInfer, 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`)으로 정한다.
이 실험의 IMPRESS는 공개 구현 방식 하나뿐이다. 질문 토큰의 attention만 쓰는 이전 구현(`mmimpress.serve.LayerSelector(mode="impress")`)은 arm에 넣지 않는다.
측정 보고이며 GO/NO-GO 판정 규칙은 없다. 사전 고정하는 것은 검증 단계의 통과 기준(9절)과 보고 항목의 정의(10절)다.
표와 보고서의 방법 이름은 "ReComp", "FullLoad", "IMPRESS", "Ours"만 쓴다. 옵션값은 3절에만 적는다.

## 2. 환경과 모델

- 인터프리터 `/home/dblab/anaconda3/envs/mllm_ft/bin/python` — Python 3.10.20, torch 2.5.1+cu121, **transformers 4.57.6**.
  실행 시점 버전과 pip freeze를 `config.json`·`environment_before.json`에 기록하고 각 phase 시작 때 일치하지 않으면 중단한다. `.venv-glm46v`(5.13.1)는 단위 테스트 교차 확인에만 쓴다.
- 모델 `llava-hf/llava-1.5-7b-hf` revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` (README 34절과 같은 `Llava15Runner`):
  NF4 double quant / BF16 compute / eager attention, FP16 SSD KV, greedy max 16, seed 1234, batch 1, CPU threads 2, RTX 4090 1장 단독 사용.
  visual token 576개 고정, separator 없음, HF 기본 336×336 전처리.
- 공개 구현 체크아웃: 레포 바깥 `/home/dblab/hj/_external_ref/HyperInfer` (읽기 전용, `IMPRESS_OFFICIAL_REPO`). freeze가 HEAD가 위 커밋인지 확인하고 기록한다.

## 3. Arm과 설정값

| 표의 이름 | method id (본실행, 비동기 읽기) | 동기 쌍 (gate 전용) | store 종류 |
|---|---|---|---|
| ReComp | `recompute` | — | 없음 (매 턴 픽셀부터) |
| FullLoad | `fullload_async` | `fullload` | `raster`: 전체 Visual KV, raster 순서 |
| IMPRESS | `impress_official_async` | `impress_official` | `impress_official`: 전체 Visual KV + probe key 사본, 턴 1 통계로 재배치 |
| Ours | `llava15_kv_PYR-b20_packed_async` | `llava15_kv_PYR-b20_packed` | `packed_PYR-b20`: packed PrefixStore |

- **본실행 arm (이 순서로 고정, 4개)**: `recompute`, `fullload_async`, `impress_official_async`, `llava15_kv_PYR-b20_packed_async`.
- **IMPRESS**: `mmimpress/impress_official.py` — 실험 A(LLaVA-NeXT)와 **같은 코드**다(`request_official`, `OfficialImpressSelector`, `select_tokens`, `Turn1AttentionStats`, `write_official_layout`).
  probe head 0·1·2, layer마다 25% 유지(k = ceil(0.25·576) = 144), 유사도 임계값 0.3(`sim >= 0.3`이면 합의), chunk 64 토큰(layer당 9 chunk),
  비동기 arm은 다음 layer probe sidecar 선행 읽기(`next_layer`). separator가 없으므로 visual token 576개 전부가 선택 후보다. 정의는 6절.
- **Ours**: README 34절의 `PYR-b20_packed` 하나. 벡터 문서 `runs/llava15_mt_gqa_20261112T062708Z/vectors/layer_budget_llava15_PYR-b20.json`을 그대로 복사
  (파일 SHA256 `7cc93f8a52646e337568e9a2d1341a204f4089fadc6a82639ffdb71acaea117c`, ratio SHA256 `51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724`).
  packed 파일 1개 75,497,472 bytes, 저장 형식 `llava15_prefixstore_ratio_packed_v1`. 비동기 arm은 `request_llava15_packed(readahead="layer_stream")`.
- **FullLoad**: `Server.request(mode="fullload")`. 비동기 arm은 `prefetch="layer_stream"`.
- ReComp, FullLoad, Ours의 코드·설정은 `scripts/193`(중지한 실행)과 같다. `mmimpress/**`는 수정하지 않는다.
- **검증 전용 arm** (gate phase에만, 어떤 표에도 넣지 않는다): 동기 쌍 세 개, `impress_official_r100`(IMPRESS 경로, 유지 비율 100%, 동기),
  `fullload_reordered`(IMPRESS의 재배치 store를 FullLoad loader — 선택·mask 없음, 동기 — 로 읽음). IMPRESS 계열 넷은 같은 store를 쓴다.

## 4. ReComp·FullLoad 기록

LLaVA-1.5 × MT-VQA 25k의 완결된 ReComp·FullLoad 기록은 없다(중지한 실행 `runs/llava15_mtvqa_impress_20261117T085444Z`에 5,956장의 부분 기록만 있다; 13절).
따라서 ReComp와 FullLoad(비동기)를 본실행 arm에 넣어 네 arm으로 돌리고, 모든 비교는 같은 실행 안의 값이다. 부분 기록은 이 run의 판정과 표에 쓰지 않는다.

## 5. 데이터와 절차 (첫 계약과 동일)

- MT-VQA-v2-reconstructed-25k 전체: 25,000 images, 이미지당 dialogue 1개, 3턴 → arm당 75,000요청(followup 50,000).
  ordered workload SHA256 `eb3c5e812fd6937cc0741cac6dd85d6839a7d00288dffbfa7ea8294272269cc8`, v3 run의 dialogue 목록·순서와 같은지와 25,000개 이미지 파일 SHA256을 freeze에서 확인한다.
- 채점: `mmimpress/mtvqa25k/scorer.py`, ten-answer leave-one-out 공식 VQA score(%). followup(T2–T3)은 method-local generated history(gold 미사용).
- 프롬프트·history 형식은 README 34절 LLaVA-1.5 경로(`llava15_support.prompt_factory`)와 같다.
- arm 순서는 dialogue 단위 회전, 세 턴 동안 고정. image-atomic attempt, 이미지별 독립 감사 후 commit, commit 뒤 run-local scratch store만 정확한 SHA allowlist로 삭제,
  요청 원본은 lossless gzip으로 보존, 중단 시 `--resume`.
- **턴 1은 모든 방법이 픽셀부터 계산한다.** 저장 KV 읽기는 요청마다 직전에 그 store의 모든 `.bin`을 `posix_fadvise(DONTNEED)`로 비운 cold 조건이고 비우기는 타이머 밖이다. 캐시 계층은 없다.
- TTFT: 요청 시작(프롬프트 조립, 픽셀 요청이면 이미지 디코드·전처리 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`.
- store는 이미지의 (유일한) dialogue 턴 1에서, store 종류마다 회전 순서에서 그 종류를 쓰는 첫 arm의 턴 1 캡처로 만든다.
- 다른 GPU compute 프로세스가 있으면 멈춘다(종료시키지 않는다).

| phase | 이미지 | arm | 요청 수 | 용도 |
|---|---|---|---|---|
| `gate` | v3 smoke 4장 + main manifest 순서 앞쪽 40장 = 44장 | 9 (본실행 4 + 동기 쌍 3 + `impress_official_r100` + `fullload_reordered`) | 1,188 | 검증 2·3·4·5 |
| `pilot` | v3 pilot40의 40장 | 4 (본실행 arm, 본조건) | 480 | 검증 7 |
| `main` | 25,000장 전체 (manifest 순서) | 4 | 300,000 | 본실행 |

gate·pilot 요청은 main 분모에 넣지 않는다. gate·pilot 이미지는 main에 다시 포함된다. gate·pilot은 구현 확인용이며 main의 arm이나 설정을 고르는 데 쓰지 않는다.
**1,000장 구간**: main manifest 순서의 앞쪽 1,000장. main 실행 안에서 측정된 값을 참고로 함께 적는다.

## 6. IMPRESS의 정의 (실험 A와 같은 코드, 모델만 LLaVA-1.5)

- **중요도**: probe head 0·1·2에 대해 전체 행(prefix + suffix)의 attention 열 합.
  prefix 행(시스템 프롬프트 + 이미지 위치의 행)의 기여분은 이미지의 턴 1 요청의 prefill에서 모델이 계산한 attention 확률을 읽어 구해 저장한다
  (별도 forward 없음, float32 합; probe key 파일 끝에 저장 순서로, little-endian float32 3 × 576). 요청 시 suffix 행(이전 턴의 질문·답 + 현재 질문 + 템플릿 전체)의 기여분을
  probe head의 query와 key의 causal softmax(float32)로 계산해 더한다.
- **선택**: head별 상위 144개 → 세 head 쌍의 평균 pairwise Jaccard → 0.3 이상이면 득표수 상위 144개(합의), 미만이면 fallback.
  선택 대상은 이미지 토큰(576개 전부)이고 시스템 프롬프트는 항상 유지한다(LLaVA-1.5에는 separator가 없다). 득표 동점은 저장 위치가 앞선 토큰 우선.
- **Fallback**: 그 layer의 Visual K·V를 전부 읽고 mask 없이 모두 쓴다.
- **저장·로드**: Visual KV 전체와 probe key 사본(head 0·1·2의 key)을 FullLoad와 같은 `mmimpress.llava15.raster` writer로 쓴 뒤, layer별로 이미지 토큰을 재배치 통계의 내림차순으로
  K, V, probe key 파일을 다시 쓴다(K·V 분리 저장). 재배치 통계 = 턴 1 요청 하나의 prefill에서 모든 head·모든 행의 열 합. 이후 갱신하지 않는다.
  합의 layer는 선택된 토큰이 속한 64 토큰 chunk의 K와 V를 읽고, 선택 안 된 토큰은 mask한다.
- **선행 로딩**: 비동기 arm은 layer 0의 probe sidecar는 hook 안에서, layer i+1의 것은 layer i의 hook이 끝날 때 워커 1개가 읽는다. 선택된 K·V chunk와 fallback 읽기는 hook 안에서 동기로 읽는다.
- 캡처 시간은 그 턴 1 요청의 TTFT·E2E 안에 들어간다. 재배치 쓰기는 요청 밖이며 따로 잰다. 둘 다 followup TTFT에는 들어가지 않는다.

### 원본과 다른 점 (보고서에 같은 표를 싣는다)

"원본"은 공개 구현(커밋 91df7a2)의 동작이다. 논문이 밝히지 않은 부분은 공개 구현을 원본으로 본다. "방향"은 IMPRESS(baseline)의 TTFT·score에 유리한지, 불리한지, 중립인지다.

| # | 항목 | 원본 | 이 실험 | 방향 |
|---|---|---|---|---|
| 1 | 선택 대상 | prefix 토큰 전부 | 이미지 토큰 576개. 시스템 프롬프트는 항상 유지, 예산 밖 | score에 유리 / 읽기량에 소폭 불리 |
| 2 | prefix 행 기여분 | 정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태) | 턴 1(mask 없음)의 값을 저장해 요청 때 더함 | 유리 (요청 때 prefix 행 연산 없음) |
| 3 | 캐시 계층 | 논문·공개 구현: GPU·DRAM·SSD 다층 캐시 | 없음. 모든 방법이 cold SSD | 불리 (모든 방법 공통 조건) |
| 4 | 다음 layer probe 선행 읽기 | 공개 구현의 IMPRESS 설정은 `--no-prefetch` | 켬 | 유리 |
| 5 | 득표 동점 | 순서 미정의 | 저장 위치가 앞선 토큰 우선 | 유리 또는 중립 |
| 6 | 재배치 통계의 추론 상태 | head별 25% mask 추론, 같은 prefix의 마지막 요청 | mask 없는 턴 1 요청 하나 | 중립 |
| 7 | 재배치 적용 | `key[reorder]` (token→rank 인덱스를 그대로 적용해 내용이 어긋남) | 위치 r = rank r 토큰, K·V·probe 같은 순서 | 중립 (접근 위치는 같다) |
| 8 | 중요도 수치 정밀도 | float16 softmax·합 | BF16 확률 + float32 합 / float32 softmax | 중립 |
| 9 | Fallback 읽기 단위 | chunk 파일별 | layer의 K 파일 1회, V 파일 1회 pread | 유리 (호출 수 적음, 바이트 같음) |
| 10 | 선택 chunk 읽기 단위 | chunk 파일별 | 인접한 chunk는 pread 하나로 합침 | 유리 |
| 11 | prefix 행 기여분 읽기 | 없음 | probe sidecar 끝 6,912 bytes/layer를 같은 read로 | 불리 (probe 읽기 +1.6%) |
| 12 | mask | 정확도 경로에만 있음(성능 경로는 0으로 채우고 답을 내지 않음) | mask (한 경로에서 score와 시간을 함께 잼) | 중립 |
| 13 | 모델·생성 | OPT(FP16) 텍스트 LLM, 1토큰 logits | LLaVA-1.5(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정 | 중립 |
| 14 | HyperInfer의 KV 선행 로딩 | HyperInfer 확장 | 없음 (IMPRESS의 probe 선행 읽기만) | 불리 (HyperInfer 대비; IMPRESS 정의에는 해당 없음) |

## 7. 비동기 읽기 (첫 계약 7절과 같은 규칙)

해당 layer의 질문 연산 결과 없이도 위치가 정해지는 읽기는 백그라운드 워커 1개가 layer 순서대로 미리 읽고, GPU 적재·scatter·mask는 요청 스레드가 한다.
선행 읽기는 요청의 TTFT 타이머가 시작된 뒤에만 걸린다.

- `fullload_async`: pread 64회, 범위·순서가 동기와 같음, 모든 읽기가 워커 스레드에서 타이머 시작 뒤에 발생.
- `impress_official_async`: probe sidecar 읽기 32회 중 layer 0은 요청 스레드, layer 1–31은 워커 스레드. K·V 읽기는 모두 요청 스레드. 종류별 바이트 합 = 전체.
- `…_packed_async`: `preadv` 32회, 범위가 [layer 0, …, layer 31] 순서로 파일을 빈틈·겹침 없이 덮음, 반환 바이트 합 = 75,497,472, 모든 읽기가 워커 스레드.
- 공통: 요청이 끝난 뒤 hook·mask가 남지 않음, 읽은 파일이 모두 직전에 cold 처리됨, vision tower 호출 0회.

## 8. 코드 변경 범위

- 신규: `scripts/199_eval_llava15_mtvqa_impress_code.py`(freeze·gate·pilot·main; `scripts/193`에서 IMPRESS arm의 요청·저장·감사 분기를 공개 구현 경로로 바꾼 것),
  `scripts/200_validate_llava15_mtvqa_impress_code.py`(검증), `scripts/201_report_llava15_mtvqa_impress_code.py`(보고; 측정에 관여하지 않으므로 소스 freeze에서 제외),
  `tests/test_impress_code_runs.py`(실험 A와 공용), 이 문서.
- 수정 없음: `mmimpress/**`(`impress_official.py`, `llava15/*` 포함), 기존 스크립트 01–195, 기존 테스트. `mmimpress/impress_official.py`는 separator가 없는 store에서도 그대로 동작한다(테스트로 고정).
- 건드리지 않는 것: 기존 `results/`, `runs/`, 기존 store, LLaVA-NeXT·Qwen·GLM 경로.

## 9. 검증 단계와 통과 기준 (사전 고정; 하나라도 실패하면 본실행 없이 실패 내용만 보고)

1. **공개 구현과의 동등성** (CPU; `tests/test_impress_code_runs.py`와 `tests/test_impress_official.py`의 `PublicEquivalenceTests`). 같은 텐서를 공개 구현의 함수와 우리 구현에 넣어
   head별 상위 k, 평균 Jaccard, fallback 판정, 선택 집합이 같아야 한다. LLaVA-1.5 구조(후보 576개, k = 144, separator 없음)의 case를 포함한다.
   통과 기준: 두 파일의 모든 `PublicEquivalenceTests` case가 skip 없이 통과(득표 동점이 경계에 걸린 입력은 득표수 조건으로 비교).
2. **prefix 행 기여분** (GPU, gate 뒤). gate 이미지 44장 각각에 대해, 저장한 값과 같은 턴 1 프롬프트로 픽셀부터 다시 전체 계산해 캡처한 값이
   **모든 원소에서 같아야 한다**(prefix 행 기여분, 재배치 통계 둘 다).
3. **정합성** (gate). `impress_official_r100`의 생성 토큰열이, 같은 재배치 store를 FullLoad loader(선택·mask 없음)로 읽은 `fullload_reordered`와 모든 요청(132건)에서 같아야 한다
   (프롬프트도 같아야 한다). 참고(판정에 쓰지 않는다): `fullload_reordered`와 raster store `fullload`의 생성 토큰이 다른 요청 수를 보고한다.
4. **비동기 정합성** (gate). FullLoad, IMPRESS, Ours 각각에서 비동기 arm과 동기 arm의 생성 토큰열과 읽은 바이트 수(전체 및 종류별)가 모든 요청(132건/방법)에서 같아야 한다.
5. **재현성 — 턴 1 일치**: gate(9 arm)와 pilot(4 arm)의 모든 이미지에서 모든 arm의 턴 1 생성 토큰열이 같아야 한다. main에서는 일치 건수를 보고한다.
   (ReComp·FullLoad·Ours를 대조할 완결된 이전 기록은 이 실험에 없다 — 4절.)
6. **기존 테스트.** `python -m unittest discover -s tests`를 두 환경에서 돌린다. 환경 문제로 실패하던 것(4.57.6: GLM 3개, 5.13.1: 2개) 외에 새 실패가 없어야 한다.
7. **Pilot 40장** (pilot, 본조건 4 arm): 40장 모두 독립 감사 PASS로 commit되고 5번이 성립해야 한다. pilot 실측으로 main 예상 소요 시간과 필요 디스크를 계산해 `gates/pilot.json`에 남긴다.
   필요 디스크(30 GiB reserve + 작업 묶음 4 GiB + pilot 실측 원본 크기 × 625 × 1.35)가 여유보다 크면 FAIL(BLOCKED_STORAGE). 소요 시간에는 기준을 두지 않는다.
8. 1–7이 모두 PASS면 `gates/admission.json`을 PASS로 쓰고 25,000장 전체를 이어서 실행한다. `main` stage는 admission 없이는 시작하지 않는다.

모든 phase에서 이미지별 독립 감사(완전성, ID, 질문·annotation 동결, 공식 VQA score 재계산, history 계보와 프롬프트 재구성, TTFT 재계산, vision 호출 수,
실제 read syscall trace와 바이트 집계 일치, cold 처리, 방법별 I/O 조건 — 7절; IMPRESS는 선택 파라미터, layer별 결정과 실제 읽은 chunk의 일치, probe sidecar read 크기,
store 순서가 증거 파일의 재배치 통계로 다시 계산한 순서와 같은지, 저장된 prefix 행 기여분의 SHA, 통계가 턴 1 요청에서 나왔는지; Ours는 k_l 독립 재계산·파일 길이)가
실패하면 그 이미지는 commit되지 않고 실행이 멈춘다.

## 10. 보고 항목의 정의 (측정하지 않은 값은 "미실행")

- **본실행 표** (followup 기준, arm별): VQA score(T1, T2, T3, 전체, followup), followup TTFT 평균·p50·p95, 같은 실행의 FullLoad 및 ReComp 대비 TTFT 평균 비율,
  SSD MB/followup(실제 read syscall이 돌려준 바이트, MB = 1e6 bytes). IMPRESS는 종류별 바이트(probe / K / V).
- **이미지 단위 paired bootstrap 95% CI** (10,000회, seed 1234, `scripts/142`의 `bootstrap`): Ours−IMPRESS, Ours−FullLoad, Ours−ReComp, IMPRESS−FullLoad의 followup score와 TTFT.
  CI가 0을 포함하면 "차이 미입증"으로 쓴다(동등성 아님).
- **IMPRESS 내부 통계**: fallback 비율과 평균 Jaccard(전체 및 layer별), K·V chunk 비율(읽은 chunk / layer당 9 chunk),
  중요도 합에서 prefix 행이 차지하는 비중(layer마다 Σ prefix 행 기여분 / Σ(prefix + suffix), probe head 3개와 이미지 토큰 전체, 요청 평균),
  GPU가 읽기를 기다린 시간(`read_blocked_ms`; FullLoad·Ours도 같은 정의로). 읽기와 연산이 겹치므로 hook 시간과 디스크 시간의 합을 TTFT로 해석하지 않는다.
- **저장·턴 1 비용**: 방법별 이미지당 store 크기(구성별), 쓰기 시간, IMPRESS의 턴 1 추가 비용(턴 1 요청 안의 attention 통계 캡처, 요청 밖의 순서 계산·재배치 쓰기).
- 6절의 "원본과 다른 점" 표, 중지한 실행의 상태, 검증 단계별 통과 여부, 예상과 다른 점, 하지 않은 것, 최대 디스크 사용량, 디스크 여유와 SSD 쓰기량(예상·실측),
  실행 환경의 transformers 버전과 공개 구현의 커밋, 실제 요청 수·실패·소요 시간.

## 11. 저장 계획·SSD 쓰기량·삭제 범위·보호

- 디스크 여유 실측(계약 작성 시점): 439.4 GiB (471,748,730,880 bytes), `/dev/nvme0n1p2`. 실험 A가 먼저 끝난 뒤의 여유는 freeze가 `storage_plan.json`에 기록한다.
- image-major: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch 삭제. 한 이미지의 묶음:
  전체 KV store 2개(FullLoad용, IMPRESS용; 각 K/V 301,989,888 + probe key 14,155,776 + prefix 행 기여분 221,184 bytes + sys/meta) + packed 75,497,472 bytes ≈ 0.71 GB (상한 1 GiB).
  최소 reserve 30 GiB + 작업 묶음 4 GiB + 원본·evidence 증가분 16 GiB가 없으면 `BLOCKED_STORAGE`로 멈춘다.
- **예상 SSD 쓰기량**: 이미지당 약 1.03 GB(전체 KV store 2개 + IMPRESS 재배치 재기록 + packed) × 25,084장(gate 44 + pilot 40 + main 25,000) ≈ **25.9 TB** (요청 원본·증거 수십 GB는 별도).
  본실행의 성격상 불가피하다. 실측은 `/sys/block/nvme0n1/stat`의 쓰기 섹터 수 차이(장치 전체)로 보고한다.
  (중지한 실행은 5,956장까지 같은 방식으로 약 6.1 TB를 이미 썼다.)
- 삭제 범위: `runs/<RUN>/scratch/**`(감사+commit 후 정확한 allowlist)뿐이다. 기존 `results/`, `runs/`, store는 수정·삭제하지 않는다.
  freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 소스: freeze 때 `mmimpress/**/*.py`, `scripts/*.py`(보고 스크립트 198·201 제외), `tests/test_*.py`의 SHA256을 `config.json`에 넣고 각 phase 시작 때 대조한다.
- 커밋과 푸시는 하지 않는다.

## 12. 하지 않는 것

- 이전 IMPRESS 구현의 arm, Ours의 다른 변형, 25% 외 유지 비율, 다른 임계값·probe head, warm cache, 반복 실행, LLaVA-NeXT(실험 A는 별도 계약)·Qwen·GLM, 다른 데이터셋.
- 공개 구현의 캐시 계층, HyperInfer의 KV 선행 로딩, 여러 요청에 걸친 재배치 갱신, 선택된 K·V chunk나 fallback 읽기의 비동기화, 두 layer 이상 앞선 probe 선행 읽기.
- 본실행에서의 동기 arm(동기·비동기 TTFT 차이는 gate 44장의 값만 참고로 적는다), O_DIRECT·io_uring, 워커 여러 개, SMART·NAND 수준 측정.

## 13. 이전 실행과의 관계 (공개)

- **중지한 실행**: `runs/llava15_mtvqa_impress_20261117T085444Z`(계약 `docs/llava15_mtvqa_impress_v2_contract.md`, 이전 IMPRESS 구현)의 본실행을 이 작업의 지시에 따라
  2026-11-17 18:15:20 UTC에 이미지 경계에서 중지했다(`STOP_REQUESTED`). 완료 5,956 / 25,000장, 71,472 / 300,000요청. 부분 결과는 삭제하지 않았고 `STATUS_STOPPED.json`으로 "중단됨" 표시를 남겼다.
  scratch store는 그 실행의 수명주기대로 이미지마다 삭제되어 남아 있지 않다. 그 실행의 pilot·부분 본실행 값(예: pilot followup TTFT ReComp 122.0 / FullLoad 116.5 / IMPRESS 236.7 / Ours 73.8 ms)은
  이 계약을 쓰기 전에 알려져 있었다. 이 run의 판정과 표에 쓰지 않는다.
- 첫 run `runs/llava15_mtvqa_impress_20261117T074736Z`와 개발 리허설 `…_dev_20261117T0900Z`도 수정·삭제하지 않는다.
- 이 계약의 arm, 설정값, 검증 기준은 사용자가 지정한 것이다(2026-11-18 KST 작업 지시). 검증 3번은 v2 계약의 검증 4와 같은 형태다.
- 이 계약에 대한 개발 리허설(`runs/llava15_mtvqa_impress_code_dev_<UTC>`, 측정 아님)을 freeze 전에 한 번 돌려 스크립트 오류를 확인한다. 그 값은 판정과 표에 쓰지 않는다.
- 실험 A(`docs/mt_gqa_impress_code_contract.md`)를 먼저 끝내고 이 실험을 시작한다. A가 실패해도 이 실험은 진행한다.
