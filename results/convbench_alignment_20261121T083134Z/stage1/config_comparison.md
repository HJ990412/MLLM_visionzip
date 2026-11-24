# 설정 대조표 (1단계-3)

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
