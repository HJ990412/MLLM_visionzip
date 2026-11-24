# ConvBench 기준 점수 점검 보고 — ReComp vs MetaCompress 기본 모델

결과 디렉터리 `results/convbench_alignment_20261121T083134Z` · 작성 2026-11-21T23:40:00Z · 드라이버 상태 `done`

> 실행하지 않았거나 실패한 단계: D1_fp16_gen (GPU 메모리 부족으로 건너뜀)

## 1. 최종 대조표

목표(실행 전 고정): 단계별 차이 ±3.0점, 평균 차이 ±2.0점. 분모 577, 판정 불가는 0승. 판정 불가는 S1 / S2 / S3 건수.

| 모델 | 행 | S1 | S2 | S3 | 평균 | 판정 불가 |
|---|---|---:|---:|---:|---:|---|
| LLaVA-NeXT-7b | MetaCompress 기본 모델 | 7.95 | 11.46 | 7.58 | 9.00 | 알 수 없음 |
| LLaVA-NeXT-7b | 9월 ReComp (9월 판정기) | 7.28 | 7.28 | 4.33 | 6.30 | 12 / 46 / 81 |
| LLaVA-NeXT-7b | **최종 ReComp** (3B 생성 + 확정 판정기) | 9.19 | 8.49 | 7.63 | 8.43 | 9 / 17 / 5 |
| LLaVA-1.5-7b | MetaCompress 기본 모델 | 4.33 | 5.72 | 5.55 | 5.20 | 알 수 없음 |
| LLaVA-1.5-7b | 9월 ReComp | 실행한 적 없음 | | | | |
| LLaVA-1.5-7b | **최종 ReComp** (4단계) | 4.85 | 4.16 | 4.51 | 4.51 | 24 / 20 / 6 |

- LLaVA-NeXT-7b: **충족** — MetaCompress 대비 차이 S1 +1.24, S2 -2.97, S3 +0.05, 평균 -0.57
- LLaVA-1.5-7b: **충족** — MetaCompress 대비 차이 S1 +0.52, S2 -1.56, S3 -1.04, 평균 -0.69
- 참고(판정 기준에는 쓰지 않음): 승률 8% 근처에서 577대화의 이항 표준오차는 약 1.13점이다. 우리 쪽과 MetaCompress 쪽 모두 단일 실행이다.

## 2. 1단계 결과 (저장된 자료 분석)

### 2.1 판정 불가 543건 분류 (9월 실행, 네 방법)

| 방법 | 단계 | 판정 불가 | (a) 상한에서 잘림 | (b) 결론 있음·해석 실패 | (c) 승자 미선택 | (d) 기타 |
|---|---|---:|---:|---:|---:|---:|
| ReComp | S1 | 12 | 3 | 9 | 0 | 0 |
| ReComp | S2 | 46 | 22 | 24 | 0 | 0 |
| ReComp | S3 | 81 | 59 | 22 | 0 | 0 |
| ReComp | 합계 | 139 | 84 | 55 | 0 | 0 |
| FullLoad | S1 | 8 | 3 | 5 | 0 | 0 |
| FullLoad | S2 | 36 | 14 | 22 | 0 | 0 |
| FullLoad | S3 | 87 | 64 | 23 | 0 | 0 |
| FullLoad | 합계 | 131 | 81 | 50 | 0 | 0 |
| Prefix25 | S1 | 14 | 4 | 10 | 0 | 0 |
| Prefix25 | S2 | 35 | 14 | 21 | 0 | 0 |
| Prefix25 | S3 | 82 | 58 | 24 | 0 | 0 |
| Prefix25 | 합계 | 131 | 76 | 55 | 0 | 0 |
| Prefix45 | S1 | 15 | 4 | 11 | 0 | 0 |
| Prefix45 | S2 | 40 | 19 | 21 | 0 | 0 |
| Prefix45 | S3 | 87 | 68 | 19 | 0 | 0 |
| Prefix45 | 합계 | 142 | 91 | 51 | 0 | 0 |
| 전체 | 합계 | 543 | 332 | 211 | 0 | 0 |

- (a) 판정문이 512토큰 상한을 다 쓰고 결론 없이 끝남. (b) 결론 문장은 있으나 9월 해석기가 읽지 못함. (c) 승자를 고르지 않음, (d) 기타는 0건.
- 상한에 닿은 판정 불가는 428건이고, 그중 96건은 잘리기 전에 결론처럼 보이는 문장이 있어 (b)로 셌다(규칙 기반 분류라 항목별 중간 결론이 섞여 있다). 상한에 닿지 않은 115건은 전부 (b)다.
- (b)의 원인: Llama는 "Assistant A's response is better"라고 써서 공식 1차 문구에 걸리지 않고, 9월의 2차 추출은 공식 추출이 아닌 자체 프롬프트에 출력 16토큰·한 글자 완전 일치 규칙이었다. 2차 출력이 A/B로 시작했는데도 탈락한 (b)가 144건이다.
- 방법별·단계별·분류별 예시 3건씩: `stage1/unresolved_examples.md`. 건별 분류: `stage1/unresolved_classification.csv`.

### 2.2 9월 ReComp 생성 답 점검

| 턴 | 요청 | 답 길이(토큰) 평균 / p50 / p90 / p95 / p99 / 최대 | 기준 답 길이(토큰) 평균 / p50 / p90 / p95 / p99 / 최대 | 상한 도달(1024) | 상한 도달(문맥으로 낮춘 상한) | 문맥으로 상한이 낮아진 요청 | 반복 표시 | 숫자 반복 검토 표시 | 입력 > 4096 | 입력 p50 / 최대 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| T1 | 577 | 137 / 98 / 264 / 310 / 1024 / 1024 | 177 / 96 / 396 / 480 / 657 / 9306 | 12 | 1 | 2 | 5 | 3 | 0 | 2287 / 3222 |
| T2 | 577 | 171 / 116 / 361 / 472 / 1024 / 1024 | 185 / 134 / 416 / 477 / 606 / 926 | 10 | 3 | 37 | 2 | 1 | 1 | 2434 / 4137 |
| T3 | 577 | 346 / 322 / 620 / 848 / 1024 / 1024 | 486 / 471 / 853 / 973 / 1198 / 1534 | 20 | 11 | 73 | 4 | 0 | 6 | 2600 / 4945 |

합계: 상한 도달 57/1731 (3.29%), 그중 문맥으로 낮춘 상한 15건; 반복 표시 11건; 4096 초과 입력 7건; 상한에 한 번이라도 도달한 대화 45개.

- "문맥으로 낮춘 상한"은 9월 규칙 `min(1024, 4096 - 입력 + 1)` 때문에 1024보다 작아진 상한이다. 반복 표시는 9월 `per_request_diagnostics.csv`의 표시를 그대로 셌다.

### 2.3 설정 대조표


- **공식**: ConvBench 공식 저장소 `data/convbench_source` (commit `cbaaa30c6a4dec1f4789d391778972d3cb8b19ca`).
  줄 번호는 `ConvBenchEval/convbencheval.py`(이하 `eval.py`), `ConvBenchEval/prompts.py`,
  `VLMEvalKit/vlmeval/vlm/llava.py`(이하 `llava.py`), `VLMEvalKit/run_multi_turn.py` 기준.
- **MetaCompress 저장소**: https://github.com/MArSha1147/MetaCompress, commit `70a620e` (2026-03-21), 2026-11-21 확인.
  브랜치는 `main` 하나, 커밋 3개, 파일은 `README.md` 하나(538바이트)뿐이다. README에는
  "The code implementation is currently being organized and will be released as soon as possible."라고만 적혀 있다.
  따라서 저장소 열은 전부 **없음**이다. 논문(arXiv 2603.21701)에 적힌 내용만 괄호 안에 따로 적는다.
- **9월 실행**: `runs/convbench_context/full_577_native`(생성), `runs/convbench_context/judge_full_577_stage_only`(판정).
- 표시: **≠공식** = 공식 코드와 다름, **?** = MetaCompress 쪽을 알 수 없어 대조 불가.

## 판정 쪽

| 항목 | ConvBench 공식 | MetaCompress 저장소 (논문) | 9월 실행 | 표시 |
|---|---|---|---|---|
| 판정 모델 | `gpt-3.5-turbo` (`eval.py:253-285`); 400 오류 시 `gpt-4` (`eval.py:170`) | 없음 (논문 §6.1 각주 2: GPT-3.5-turbo 대신 Llama-3.1-8B-Instruct) | Llama-3.1-8B-Instruct, revision `0e9e39f`, BF16, 비양자화, SDPA (`mmimpress/convbench_judge.py:239-254`) | ≠공식 (MetaCompress 논문과는 같은 모델), 정밀도·로딩 방식은 **?** |
| 판정 프롬프트 | system/user/assistant/user 4개 메시지 (`eval.py:108-152`, `prompts.py:326-428`), 이미지는 넣지 않음(`eval.py:124` 주석 처리) | 없음 | 공식 `generate_request`를 그대로 불러 씀 (`convbench_judge.py:41-62, 108-128`) | 같음 |
| A/B 배치 | `pairwise.npy[i]`; 0이면 A=모델 답 (`eval.py:301-303, 238-241`) | 없음 | 같은 파일·같은 규칙 (`convbench_judge.py:65-75, 115`) | 같음 |
| 판정 출력 상한 | `max_tokens=512` (`eval.py:160`) | 없음 | 512 (`scripts/42_judge_convbench.py:325`) | 같음. 단 Llama 판정문은 6,924건 중 506건이 512에 닿음 → **결함 (가)** |
| 판정 디코딩 | temperature를 넘기지 않음 → API 기본값(1.0) 샘플링. `temp_gpt=0.0` 인자는 쓰이지 않음 (`eval.py:155-160`) | 없음 | greedy, `do_sample=False` (`convbench_judge.py:268-274`) | ≠공식, **?** |
| 문맥 초과 처리 | `context_length_exceeded`이면 마지막 user 메시지만 보냄 (`eval.py:177-180`) | 없음 | 해당 없음 (최대 17,019토큰, Llama 문맥 131,072 안) | — |
| 승자 해석 1차 | 소문자화한 판정문에 `response a is better` / `response b is better` 중 정확히 하나 (`eval.py:214-219`) | 없음 | 같은 규칙 (`convbench_judge.py:131-135`) | 같음 |
| 승자 해석 2차 | 공식 추출 프롬프트 + 예시 3개(`prompts.py:4-40`, `eval.py:192-211`)를 `gpt-3.5-turbo`에 보내고 출력에 `Final Answer: Response A` / `B` 중 정확히 하나 (`eval.py:221-229`), 출력 상한 512 | 없음 | **자체 작성 프롬프트**(`convbench_judge.py:152-165`), 출력 상한 **16토큰**, 출력 전체가 `A`/`B` 한 글자일 때만 인정(`convbench_judge.py:138-142`, `scripts/42_judge_convbench.py:235-240`) | **≠공식 (나)** |
| 승자 해석 3차(집계 시) | 등급이 A/B가 아니면 판정문에서 `Rating:(.*)` → A/B, 아니면 C (`eval.py:67-77, 397-404`) | 없음 | 없음(2차 실패 시 바로 판정 불가). 결과는 같음: Llama 판정문에 `Rating:`이 없으면 C | 사실상 같음 |
| 점수 집계 | `TEST_SUM = 577`, ID 1..578을 돌며 결과 파일이 없으면 0승, 승수/577 (`eval.py:376-435`) | 없음 (논문 Table 1의 LLaVA-NeXT-7b Base 7.95 / 11.46 / 7.58은 k/577로 표현되지 않음. 세 값을 동시에 만드는 분모는 400..578 중 **541뿐**: 43/541, 62/541, 41/541. LLaVA-1.5-7b·13b, LLaVA-NeXT-13b Base는 577로만 맞음) | 분모 577 고정, 판정 불가 0승 (`convbench_judge.py:168-236`) | 공식과 같음. MetaCompress의 LLaVA-NeXT-7b 행은 **?** |
| 판정 범위 | 1·2·3턴 + overall 4회 (`eval.py:250-267`) | 없음 | 1·2·3턴만(`--stage-only`), overall은 Avg에 들어가지 않음 | S1·S2·S3에는 영향 없음 |

## 생성 쪽

| 항목 | ConvBench 공식 (LLaVA-1.5 래퍼) | MetaCompress 저장소 (논문) | 9월 실행 (LLaVA-NeXT-7b) | 표시 |
|---|---|---|---|---|
| 대상 모델 | `liuhaotian/llava-v1.5-7b`, `-13b` (`vlmeval/config.py:23-24`). LLaVA-NeXT 래퍼는 없음 | 없음 (논문: LLaVA-1.5-7b/13b, LLaVA-NeXT-7b/13b, XComposer-2.5-7b) | `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6c`, transformers 4.57.6 | 체크포인트 출처 **?** |
| 생성 길이 상한 | `max_new_tokens=1024` (`llava.py:53`), 문맥에 따른 조정 없음 | 없음 | 1024 (`scripts/41_eval_convbench_answers.py:961-962`). 단 입력 ≤ 4096이면 `min(1024, 4096-입력+1)`로 낮춤, 입력 > 4096이면 1024 (`41:355-382`) | **≠공식 (나)** + 낮춘 상한에 닿아 잘린 답 15건 **(가)** |
| 프롬프트 형식 | `llava_v1` 대화 템플릿 (`llava.py:51, 134-137`): `SYSTEM USER: <image>\nQ1 ASSISTANT:` | 없음 | 같은 문자열 (`41:198-211`) | 문자열은 같음. 토큰열은 아래 항목 |
| 프롬프트 토큰열 | 원본 LLaVA `tokenizer_image_token` (`llava.py:139`): `<image>` 뒤에 `▁`(29871) + `\n`(13), 이력의 `</s>` 뒤에 `▁US`,`ER`(3148, 1001) | 없음 | HF `LlavaNextProcessor` 토크나이저: `<image>` 뒤 `▁` 없음, `</s>` 뒤가 `USER`(11889) 한 토큰. 3턴 프롬프트에서 공식보다 3토큰 짧음. (LLaVA-1.5 HF 토크나이저는 같은 문자열에서 공식 토큰열과 **일치**) | **≠공식 (나)**, LLaVA-NeXT만 해당, 이력 경계 차이는 T2·T3에만 영향 |
| 대화 이력 | 모델이 생성한 A1, A2를 다음 턴 이력으로 사용 (`llava.py:148-179`, `run_multi_turn.py:178-188`의 `i == 0`), 이미지는 첫 턴에만 | 없음 | 같은 방법으로 생성한 A1, A2 (`41:877`), 이미지는 첫 턴에만 | 같음, MetaCompress는 **?** |
| 이력 구분자 | 답 뒤 `</s>` (`llava_v1`, sep2) | 없음 | 답 뒤 `</s>` (`41:210`) | 같음 |
| 답 후처리 | `.strip().split("</s>")[0]` (`llava.py:145`) | 없음 | 특수 토큰 제거 후 `.strip()` (`mmimpress/serve.py:2734`) | 같음 |
| 디코딩 | `do_sample=True, temperature=0.2, top_p=None, num_beams=1` (`llava.py:53`) | 없음 | greedy (`serve.py:2722-2726`) | ≠공식, **?** (논문 전체가 greedy — 채택 대상 아님, 진단으로만 확인) |
| 모델 정밀도 | FP16 (`llava.py:126`, `load_pretrained_model` 기본값) | 없음 | NF4 4-bit(double quant) + BF16 계산 + eager (`mmimpress/model.py:31-39`) | ≠공식, **?** (논문 전체 설정 — 바꾸지 않음, 진단으로만 확인) |
| 이미지 전처리 | `image_aspect_ratio='pad'` (`llava.py:125`): 정사각형으로 채운 뒤 336 | 없음 | LLaVA-NeXT: HF AnyRes. LLaVA-1.5(4단계): HF 기본값 = 짧은 변 336 리사이즈 + **가운데 자르기** (`mmimpress/llava15/runner.py:10-12`) | LLaVA-1.5에서 ≠공식, **?** (논문 전체의 LLaVA-1.5 전처리 — 바꾸지 않음, 범위를 벗어나면 진단으로만 확인) |
| 4096 초과 입력 | 처리 없음(그대로 실행) | 없음 | 자르지 않고 그대로 실행 (`native_overflow_no_truncation_v1`) | 같음 |
| 생성 실패 처리 | 예외 시 10회까지 재시도, 실패하면 그 대화는 결과 없음 → 0승 (`run_multi_turn.py:186-192`) | 없음 | 실패 0건 | — |


## 3. 시도한 설정 전체의 점수표

판정기 열: `9월` = 9월 판정기(상한 512, 자체 2차 추출), `J1` = 상한 512 + 공식 2차 추출, `J1+J2` = 상한 2048 + 공식 2차 추출(**확정 판정기**).

| # | 모델 | 생성 설정 (바꾼 항목) | 판정기 | S1 | S2 | S3 | 평균 | 판정 불가 | 채택 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 0 | NeXT | 9월 답 (바꾼 것 없음) | 9월 | 7.28 | 7.28 | 4.33 | 6.30 | 12 / 46 / 81 | 기준 |
| 1 | NeXT | 9월 답 | J1 | 7.28 | 7.11 | 4.33 | 6.24 | 18 / 50 / 92 | 아니오 (참고: 공식 상한 유지) |
| 2 | NeXT | 9월 답 | J1+J2 | 7.28 | 7.63 | 5.20 | 6.70 | 18 / 27 / 5 | 판정기 확정 |
| 3 | NeXT | 3A: 생성 상한 항상 1024 (G1) | J1+J2 | 7.28 | 7.63 | 5.20 | 6.70 | 17 / 25 / 5 | 아니오 (항목별 효과용) |
| 4 | NeXT | 3B: G1 + 공식 토큰열 (G2) | J1+J2 | 9.19 | 8.49 | 7.63 | 8.43 | 9 / 17 / 5 | **예 (확정 생성 설정)** |
| 5 | NeXT | D1: 3B + FP16 가중치 (건너뜀: GPU 메모리 부족으로 건너뜀) | J1+J2 | — | — | — | — | — | 아니오 (진단) |
| 6 | NeXT | D2: 3B + temperature 0.2 샘플링 (조건에 해당하지 않아 실행하지 않음) | J1+J2 | — | — | — | — | — | 아니오 (진단) |
| 7 | 1.5 | 4단계: 3B와 같은 설정 | J1+J2 | 4.85 | 4.16 | 4.51 | 4.51 | 24 / 20 / 6 | **예** |
| 8 | 1.5 | D3: 4단계 + 공식 pad 전처리 (조건에 해당하지 않아 실행하지 않음) | J1+J2 | — | — | — | — | — | 아니오 (진단) |

같은 답을 공식 상한 512 보기(J1)로 읽은 점수 (참고):

| 실행 | S1 | S2 | S3 | 평균 | 판정 불가 |
|---|---:|---:|---:|---:|---|
| 9월 답 | 7.28 | 7.11 | 4.33 | 6.24 | 18 / 50 / 92 |
| 3A | 7.28 | 7.11 | 4.33 | 6.24 | 18 / 47 / 91 |
| 3B | 9.19 | 8.49 | 5.89 | 7.86 | 12 / 32 / 107 |
| LLaVA-1.5 4단계 | 4.85 | 4.16 | 3.64 | 4.22 | 28 / 33 / 101 |

생성 실행별 점검 (T1 / T2 / T3):

| 실행 | 새로 생성 / 9월 답 재사용 (대화) | 상한 도달 | 문맥으로 상한이 낮아진 요청 | 입력 > 4096 | 최대 입력 토큰 | 공식 토큰열 불일치 요청 |
|---|---|---|---|---|---:|---:|
| `next_3A_cap` | 15 / 562 | 13 / 12 / 22 | 0 / 0 / 0 | 0 / 1 / 6 | 5138 | 45 |
| `next_3B_cap_tok` | 577 / 0 | 15 / 10 / 19 | 0 / 0 / 0 | 0 / 0 / 8 | 4613 | 0 |
| `llava15_final` | 577 / 0 | 4 / 6 / 17 | 0 / 0 / 0 | 0 / 0 / 0 | 2736 | 0 |
| `next_D1_fp16` | — | — | — | — | — | — |

- 3A의 "공식 토큰열 불일치"는 9월 토큰열을 일부러 유지한 실행이므로 새로 생성한 요청 전부가 불일치로 세어진다(9월 답을 재사용한 대화는 세지 않는다).
- 사전 검사(replay): 새 생성 스크립트를 9월 설정으로 12대화에 돌린 답이 9월 답과 글자 단위로 같은 요청 36/36.

2단계 확인 — 고친 판정기가 9월에 판정 가능했던 건의 승패를 바꾸었는가 (ReComp 1,731건):

- 앞 512토큰 판정문이 9월 원문과 글자 단위로 같은 건: 1731/1731 (다른 건 0건).
- 9월에 판정 가능했던 건: 1592건.
- J1 (상한 512): 승자 같음 1565건, **승자 바뀜 0건**, 판정 불가로 바뀜 27건. 9월 판정 불가 중 이번에 판정된 건 6건(그중 모델 승 1건), 여전히 판정 불가 133건.
  - 판정 불가로 바뀐 건: `convbench:26` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:54` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:75` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:85` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:92` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:112` _second_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:134` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:174` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:192` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:206` _second_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:213` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:215` _third_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:240` _third_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:277` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:304` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:308` _second_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:311` _third_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:348` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:380` _second_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:383` _first_turn (9월 B, second_pass)
- J1+J2 (상한 2048, 확정): 승자 같음 1580건, **승자 바뀜 0건**, 판정 불가로 바뀜 12건. 9월 판정 불가 중 이번에 판정된 건 101건(그중 모델 승 8건), 여전히 판정 불가 38건.
  - 판정 불가로 바뀐 건: `convbench:75` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:92` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:112` _second_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:174` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:206` _second_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:308` _second_turn (9월 A, second_pass)
  - 판정 불가로 바뀐 건: `convbench:348` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:380` _second_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:383` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:473` _second_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:507` _first_turn (9월 B, second_pass)
  - 판정 불가로 바뀐 건: `convbench:531` _third_turn (9월 A, second_pass)
- 판정문이 상한 2048에도 닿은 건(S1 / S2 / S3): 3 / 2 / 3. 사전에 정한 대로 상한을 더 올리지 않았다.

## 4. 바꾼 항목별 효과 (점, LLaVA-NeXT-7b ReComp)

| 항목 | 비교 | ΔS1 | ΔS2 | ΔS3 | Δ평균 |
|---|---|---:|---:|---:|---:|
| J1 판정 2차 추출을 공식대로 | (9월 답, J1) − (9월 답, 9월 판정기) | +0.00 | -0.17 | +0.00 | -0.06 |
| J2 판정 출력 상한 512 → 2048 | (9월 답, J1+J2) − (9월 답, J1) | +0.00 | +0.52 | +0.87 | +0.46 |
| G1 생성 상한 항상 1024 | 3A − (9월 답, J1+J2) | +0.00 | +0.00 | +0.00 | +0.00 |
| G2 공식 토큰열 | 3B − 3A | +1.91 | +0.87 | +2.43 | +1.73 |
| 전체 (J1+J2+G1+G2) | 3B − (9월 답, 9월 판정기) | +1.91 | +1.21 | +3.29 | +2.14 |

- G1과 G2는 순차 분해다(G2만 고친 전체 실행은 돌리지 않았다). 한 대화의 승패가 한 번 바뀌면 0.17점이다.
- 모델 승수 9월 답 (J1+J2): S1 42, S2 44, S3 30 / 577.
- 모델 승수 3A (J1+J2): S1 42, S2 44, S3 30 / 577.
- 모델 승수 3B (J1+J2): S1 53, S2 49, S3 44 / 577.
- 모델 승수 9월(9월 판정기): S1 42, S2 42, S3 25 / 577.

## 5. 양자화 진단 (D1, 채택하지 않음)

- 결과 없음. 상태: `skipped` GPU 메모리 부족으로 건너뜀

## 6. 남은 차이에 대한 설명

**확인된 원인(고친 것)**

- J1·J2(판정 쪽): 판정 불가가 S1/S2/S3 12/46/81건에서 18/27/5건으로 줄었다. 점수 효과는 4절.
- G1(생성 상한의 문맥 조정)과 G2(LLaVA-NeXT 프롬프트 토큰열이 공식 LLaVA와 다름): 효과는 4절.

**추정이지만 확인하지 못한 원인**

- 생성 디코딩: 공식 래퍼는 temperature 0.2 샘플링, 우리는 greedy. LLaVA-NeXT가 범위 안이라 진단 D2는 돌리지 않았다.
- LLaVA-1.5 이미지 전처리: 공식은 pad, HF 기본값은 가운데 자르기. LLaVA-1.5가 범위 안이라 진단 D3는 돌리지 않았다.
- 양자화(NF4 vs FP16): 5절의 진단 값. 논문 전체 설정이라 바꾸지 않았다.
- 판정 모델 디코딩: 공식 호출은 temperature를 넘기지 않아 API 기본 샘플링이고 우리는 greedy다. MetaCompress가 Llama를 어떻게 디코딩했는지는 알 수 없다. 모델 답이 이기는 비율이 10% 미만인 상황에서 판정이 샘플링되면 승패가 뒤집히는 건이 생기므로 점수가 달라질 수 있다는 것은 추정일 뿐, 재지 않았다.
- 반복 출력: greedy에서 1024토큰 상한까지 반복하는 답이 남아 있다(3절 표의 상한 도달 수). 모델의 실제 출력이라 고치지 않았다.

**MetaCompress 쪽 설정을 알 수 없어 대조하지 못한 항목**

- 공개 저장소(commit `70a620e`)에는 README만 있다. 생성 길이 상한, 프롬프트 형식, 대화 이력 구성(생성 답인지 기준 답인지), 디코딩, 모델 정밀도, 체크포인트 출처, 판정 모델 로딩·디코딩·출력 상한, 2차 추출 방식 전부 확인할 수 없다.
- **분모**: Table 1의 LLaVA-NeXT-7b Base 7.95 / 11.46 / 7.58은 k/577로 만들 수 없고, 세 값을 동시에 만드는 분모는 400..578 중 541뿐이다(43, 62, 41승). LLaVA-1.5-7b·13b와 LLaVA-NeXT-13b의 Base는 577로만 맞는다. 즉 MetaCompress의 LLaVA-NeXT-7b 기본 모델 행은 36개 대화가 빠진 채 계산된 것으로 보이지만, 어떤 대화인지는 알 수 없다.
  - 참고 환산(판정에는 쓰지 않음): MetaCompress 쪽을 분모 577로 바꾸면 43/62/41승 = 7.45 / 10.75 / 7.11 (평균 8.44). 우리 최종 승수는 53 / 49 / 44승이다.
- 판정 모델이 같은 Llama-3.1-8B-Instruct라도 revision, 정밀도, 채팅 템플릿이 다르면 판정문이 달라진다. 확인할 수 없다.

## 7. 동결한 설정과 근거

`docs/convbench_baseline_alignment.md`에 적었다.
요약: 생성 = NF4/BF16/eager, greedy, 상한 항상 1024, 공식 LLaVA 토큰열, 생성 답 이력, 자르지 않음. 판정 = Llama-3.1-8B-Instruct BF16 greedy, 공식 프롬프트·A/B 배치, 1차 출력 상한 2048, 공식 2차 추출(상한 512), 공식 `Rating:` 3차 규칙, 분모 577, 판정 불가 0승.

## 8. 코드, 지시와 다르게 한 것, 한계

**추가한 코드** (기존 파일은 고치지 않았다)

- `scripts/212_convbench_alignment_stage1.py` — 1단계 분석(판정 불가 분류, 생성 답 점검).
- `scripts/213_convbench_recomp_generate.py` — ReComp 전용 생성(SSD store 없음). 상한 정책·토큰열·디코딩·정밀도·pad 옵션.
- `scripts/214_convbench_alignment_judge.py` — 고친 판정기(공식 2차 추출, 상한 2048, 프롬프트 해시 캐시, 상한 512 보기 동시 산출).
- `scripts/215_convbench_alignment_driver.py`, `scripts/run_convbench_alignment.sh` — 분리 실행 드라이버, status.json, 보고서.
- `docs/convbench_baseline_alignment.md` — 동결 설정(5단계).
- 방법 코드(`mmimpress/`), 9월 스크립트(`scripts/41`, `42` 등), 기존 store·결과 디렉터리·동결 파일은 수정하지 않았다. `mmimpress/convbench_judge.py`의 함수는 불러 쓰기만 했다.

**지시와 다르게 했거나 지시에 없던 것**

- 생성 디코딩(greedy)과 LLaVA-1.5 이미지 전처리(HF 기본값)는 공식 코드와 다르지만, 정밀도처럼 논문 전체가 공유하는 설정으로 보고 채택 대상에서 빼고 조건부 진단(D2, D3)으로만 두었다. 지시문이 명시한 것은 정밀도뿐이므로 이것은 내 판단이다.
- 판정 디코딩(greedy)도 공식(API 기본 샘플링)과 다르지만 바꾸지 않았다. 재현 가능한 판정을 유지하기 위해서다.
- 3단계에서 "항목별로 나눠 돌린다"를 G2만 고친 전체 실행 대신 순차 분해(3A = G1만, 3B = G1+G2)로 대신했다. 3A는 범위 충족 여부와 관계없이 항상 돌렸다(9월 답 재사용으로 비용이 작다).
- 3A는 577대화를 모두 새로 생성하지 않았다. greedy에서 문맥으로 낮춘 상한에 닿지 않은 대화는 답이 9월과 같으므로 9월 답을 그대로 썼다(근거: replay 검사). 3B·4단계·진단은 577대화 × 3턴을 모두 새로 생성했다.
- 판정 캐시: 같은 판정 프롬프트는 다시 판정하지 않고 저장된 출력을 썼다(같은 판정기·greedy이므로 출력이 같다. 2단계에서 9월 원문과의 일치 수로 검증).
- 2단계의 J1 점수는 상한 512로 따로 돌린 것이 아니라 상한 2048 출력의 앞 512토큰으로 계산했다.
- 양자화 진단은 지시대로 FP16(`torch.float16`)으로 했다. 계산 dtype도 BF16에서 FP16으로 함께 바뀐다.
- MetaCompress 논문 PDF와 저장소는 대조 목적으로만 내려받았고 프로젝트에 복사하지 않았다.

**한계**

- 모든 점수는 단일 실행이다. 577대화에서 한 대화가 0.17점이고, 승률 5–10%의 표준오차가 1점 안팎이므로 1점 이하의 항목별 효과는 방향을 단정하기 어렵다.
- 1단계의 (a)/(b) 분류는 규칙 기반이다. 상한에 닿은 건 중 (b)로 센 96건에는 항목별 중간 결론이 섞여 있다.
- 판정 모델은 MetaCompress를 따라 Llama-3.1-8B-Instruct를 썼다. 공식 ConvBench의 GPT-3.5 판정 점수와는 비교할 수 없다.
- TTFT 등 시스템 지표는 이 작업에서 재지 않았다. `scripts/213`은 품질 점검용이며 시간 측정 경계가 없다.
- 본실험의 네 방법 실행기(`scripts/41` 계열)는 9월 정책(문맥 조정 상한, 9월 토큰열)을 코드에 고정하고 있다. 동결 설정을 본실험에 쓰려면 그 실행기에 같은 두 항목을 반영해야 한다(이 작업에서는 방법 코드를 바꾸지 않았다).

## 단계 실행 기록

| 단계 | 상태 | 시작 (UTC) | 종료 (UTC) | 소요 | 비고 |
|---|---|---|---|---:|---|
| stage2_judge_sept | done | 2026-11-21T08:56:12Z | 2026-11-21T12:28:23Z | 3.59 h |  |
| replay_check | done | 2026-11-21T12:28:23Z | 2026-11-21T12:32:03Z | 0.06 h |  |
| stage3A_gen | done | 2026-11-21T12:32:03Z | 2026-11-21T12:39:44Z | 0.13 h |  |
| stage3A_judge | done | 2026-11-21T12:39:44Z | 2026-11-21T12:46:44Z | 0.12 h |  |
| stage3B_gen | done | 2026-11-21T12:46:44Z | 2026-11-21T14:53:51Z | 2.12 h |  |
| stage3B_judge | done | 2026-11-21T14:53:51Z | 2026-11-21T18:29:03Z | 3.59 h |  |
| stage4_gen | done | 2026-11-21T18:29:03Z | 2026-11-21T19:46:47Z | 1.30 h |  |
| stage4_judge | done | 2026-11-21T19:46:47Z | 2026-11-21T23:22:39Z | 3.60 h |  |
| stage5_freeze | done | 2026-11-21T23:22:39Z | 2026-11-21T23:22:39Z | 0.00 h |  |
| D1_fp16_gen | skipped | 2026-11-21T23:22:39Z | 2026-11-21T23:40:00Z | 0.29 h | GPU 메모리 부족으로 건너뜀 |
| D1_fp16_judge | not_needed |  |  |  | 조건에 해당하지 않음 |
| D2_sample_gen | not_needed |  |  |  | 조건에 해당하지 않음 |
| D2_sample_judge | not_needed |  |  |  | 조건에 해당하지 않음 |
| D3_pad_gen | not_needed |  |  |  | 조건에 해당하지 않음 |
| D3_pad_judge | not_needed |  |  |  | 조건에 해당하지 않음 |
| final_report | done | 2026-11-21T23:40:00Z | 2026-11-21T23:40:00Z | 0.00 h |  |
