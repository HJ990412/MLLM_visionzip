# ConvBench 기준 설정 동결 (baseline alignment)

작성 2026-11-21T23:22:39Z · 근거 실행 `results/convbench_alignment_20261121T083134Z` (`report.md`, `notes.md`)

이 문서의 생성 설정과 판정 설정을 이후 ConvBench 본실험(네 방법 × 두 모델)에 그대로 쓴다. 9월 실행(`results/convbench_full/full_577`)에서 바꾼 항목에는 근거를 적었다.

## 1. 생성 설정

| 항목 | 값 | 9월 대비 | 근거 |
|---|---|---|---|
| 모델 | LLaVA-NeXT-7b `llava-hf/llava-v1.6-vicuna-7b-hf` rev `c916e6c`; LLaVA-1.5-7b `llava-hf/llava-1.5-7b-hf` rev `b234b80` | 같음 / 신규 | |
| 정밀도 | NF4 4-bit(double quant), BF16 계산, eager attention | 같음 | 논문 전체 설정 |
| 디코딩 | greedy (`do_sample=False`) | 같음 | 논문 전체 설정. 공식 래퍼는 temperature 0.2 샘플링 |
| 생성 길이 상한 | `max_new_tokens = 1024`, 문맥 길이에 따른 조정 없음 | **바뀜** (9월: 입력 ≤ 4096이면 `min(1024, 4096 − 입력 + 1)`) | 결함: 낮춘 상한에 닿아 잘린 답 15건(9월 ReComp). 공식 `llava.py:53`에는 조정이 없음 |
| 프롬프트 | `llava_v1` 템플릿: `{system} USER: <image>\n{Q1} ASSISTANT: {A1}</s>USER: {Q2} ASSISTANT: …` | 문자열 같음 | 공식 `llava.py:51, 134-179` |
| 프롬프트 토큰열 | 원본 LLaVA `tokenizer_image_token`(느린 legacy 토크나이저)과 같은 토큰 ID. LLaVA-NeXT HF 토크나이저에서는 `<image> \n`, `</s> USER`로 렌더링해야 같아진다. LLaVA-1.5 HF 토크나이저는 템플릿 문자열 그대로 같다. 요청마다 일치를 검사한다 | **바뀜** (LLaVA-NeXT만: 9월은 `<image>` 뒤 `▁` 없음, `</s>` 뒤 `USER` 한 토큰) | 공식 코드와의 불일치. `scripts/213_convbench_recomp_generate.py`의 `OFFICIAL_RENDER`, `official_token_ids` |
| 대화 이력 | 같은 방법이 생성한 A1, A2. 이미지는 첫 턴에만 | 같음 | 공식 `run_multi_turn.py:178-188` |
| 답 후처리 | 특수 토큰 제거 후 `.strip()` | 같음 | 공식 `llava.py:145` |
| 문맥 한계 | 프롬프트·이력을 자르지 않는다. 입력이 4096을 넘어도 그대로 실행한다 | 같음 | 9월 `native_overflow_no_truncation_v1` |
| 이미지 전처리 | HF 프로세서 기본값(LLaVA-NeXT AnyRes, LLaVA-1.5 가운데 자르기) | 같음 | 논문 전체 설정. 공식 래퍼는 pad |

## 2. 판정 설정

| 항목 | 값 | 9월 대비 | 근거 |
|---|---|---|---|
| 판정 모델 | `meta-llama/Meta-Llama-3.1-8B-Instruct` rev `0e9e39f`, BF16, 비양자화, SDPA, greedy | 같음 | MetaCompress §6.1 각주 2 |
| 판정 프롬프트 | 공식 `generate_request` (S1·S2·S3), 이미지는 넣지 않음 | 같음 | 공식 `convbencheval.py:108-152` |
| A/B 배치 | 공식 `pairwise.npy` | 같음 | |
| 1차 판정 출력 상한 | 2048토큰 | **바뀜** (9월·공식: 512) | 결함: Llama 판정문 6,924건 중 506건이 512에 닿았고 판정 불가 543건 중 428건이 그 경우 |
| 승자 해석 1차 | 소문자 판정문에 `response a is better` / `response b is better` 중 정확히 하나 | 같음 | 공식 `convbencheval.py:214-219` |
| 승자 해석 2차 | 공식 `generate_parsing_answer_request`(공식 system 프롬프트 + 예시 3개)를 같은 Llama에 보내고, 출력에 `Final Answer: Response A` / `B` 중 정확히 하나. 출력 상한 512 | **바뀜** (9월: 자체 프롬프트, 16토큰, 한 글자 완전 일치) | 공식 코드와의 불일치. 공식 `convbencheval.py:192-229` |
| 승자 해석 3차 | 판정문의 `Rating:` → A/B, 아니면 판정 불가 | 공식 규칙 추가(결과 영향 없음) | 공식 `convbencheval.py:67-77, 397-404` |
| 점수 | 단계별 모델 승수 / 577, 평균 = (S1+S2+S3)/3, 판정 불가는 0승 | 같음 | 공식 `convbencheval.py:376-435` |

## 3. 이 설정으로 얻은 ReComp 점수

| 모델 | | S1 | S2 | S3 | 평균 | 판정 불가 |
|---|---|---:|---:|---:|---:|---|
| LLaVA-NeXT-7b | MetaCompress 기본 모델 | 7.95 | 11.46 | 7.58 | 9.00 | |
| LLaVA-NeXT-7b | ReComp (동결 설정) | 9.19 | 8.49 | 7.63 | 8.43 | 9 / 17 / 5 |
| LLaVA-NeXT-7b | 목표 범위(단계 ±3.0, 평균 ±2.0) | **충족** — MetaCompress 대비 차이 S1 +1.24, S2 -2.97, S3 +0.05, 평균 -0.57 | | | | |
| LLaVA-1.5-7b | MetaCompress 기본 모델 | 4.33 | 5.72 | 5.55 | 5.20 | |
| LLaVA-1.5-7b | ReComp (동결 설정) | 4.85 | 4.16 | 4.51 | 4.51 | 24 / 20 / 6 |
| LLaVA-1.5-7b | 목표 범위(단계 ±3.0, 평균 ±2.0) | **충족** — MetaCompress 대비 차이 S1 +0.52, S2 -1.56, S3 -1.04, 평균 -0.69 | | | | |
| LLaVA-NeXT-7b | 9월 ReComp (9월 설정) | 7.28 | 7.28 | 4.33 | 6.30 | 12 / 46 / 81 |

## 4. 실행 방법

```
# ReComp 생성 (SSD store 없음)
python scripts/213_convbench_recomp_generate.py --out <dir> --model next|llava15 --cap-policy official_1024 --tokenization official
# 판정
python scripts/214_convbench_alignment_judge.py --answers <dir> --out <dir> --cache <cache dir> --cap 2048
```

## 5. 본실험에 옮길 때 주의

- 네 방법 실행기(`scripts/41_eval_convbench_answers.py` 계열)는 9월 정책을 코드에 고정하고 있다(`_effective_cap`, `render_prompt`). 본실험 실행기에 ① 상한 항상 1024, ② 공식 토큰열 렌더링(LLaVA-NeXT: `<image> \n`, `</s> USER`)과 요청별 일치 검사를 반영해야 한다. 이 작업에서는 방법 코드를 바꾸지 않았다.
- ② 는 이미지 뒤 접미부 토큰을 바꾼다. 이미지 prefix KV(시스템 문장 ~ 이미지 토큰)는 그대로다.
- 판정은 `scripts/214`를 방법별 답에 그대로 쓰면 된다(현재는 `method == "ReComp"` 행만 읽으므로 방법 이름 선택 인자를 추가해야 한다).
- 채택하지 않은 설정과 진단 결과는 근거 실행의 `report.md` 3–6절에 있다.
