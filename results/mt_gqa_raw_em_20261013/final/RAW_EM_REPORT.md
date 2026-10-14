# MT-GQA Raw Exact Match 사후 재채점

분석 기준일: 2026-10-13 (Asia/Seoul). 기존 저장 prediction과 gold[0]만 비교했다.
`raw_em = int(prediction == gold[0])`; 문자열을 strip/lower/정규화하지 않았다.
Normalized EM은 lower → 구두점 공백 치환 → a/an/the 제거 → 공백 정리 후 exact 비교다.
추론, prompt, 데이터, history, KV, GPU 및 기존 평가 파일을 변경하지 않았다.
Raw EM은 decoder가 이미 처리하여 저장한 prediction 기준이다. 생성 token stream의 무가공 일치율이나 공식 GQA/MetaCompress 재현을 뜻하지 않는다.

## LLaVA: mt_gqa_B_llava5_qwen3_20261001T093921Z

398 images / 4,061 dialogues / 방법당 12,183 main 요청, 8,122 followups. 총 60,915요청.
무결성·membership·기존 Normalized EM 재계산 PASS. 요청별 저장 score 대조 60,915개, 없는 score 0개.
Frozen native tokenizer.decode(skip_special_tokens=True).strip(); tokenizer default cleanup is not independently disabled. serve.py / mpic.py / sparsevlm_ssd_attention.py. Raw EM compares stored prediction after this existing decoder processing.

| Method | T1 Raw EM | T2 Raw EM | T3 Raw EM | All Raw EM | Followup Raw EM |
|---|---:|---:|---:|---:|---:|
| ReComp | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/12183) | 0.000% (0/8122) |
| FullLoad | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/12183) | 0.000% (0/8122) |
| MPIC-32 | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/12183) | 0.000% (0/8122) |
| AllHead | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/12183) | 0.000% (0/8122) |
| Ours | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/4061) | 0.000% (0/12183) | 0.000% (0/8122) |

| Method | All Normalized / Raw % | 차이 pp | Followup Normalized / Raw % | 차이 pp | normalized=1/raw=0 followups |
|---|---:|---:|---:|---:|---:|
| ReComp | 66.716 / 0.000 | 66.716 | 68.481 / 0.000 | 68.481 | 5562/8122 (68.481%) |
| FullLoad | 66.650 / 0.000 | 66.650 | 68.382 / 0.000 | 68.382 | 5554/8122 (68.382%) |
| MPIC-32 | 66.765 / 0.000 | 66.765 | 68.555 / 0.000 | 68.555 | 5568/8122 (68.555%) |
| AllHead | 66.585 / 0.000 | 66.585 | 68.284 / 0.000 | 68.284 | 5546/8122 (68.284%) |
| Ours | 65.181 / 0.000 | 65.181 | 66.178 / 0.000 | 66.178 | 5375/8122 (66.178%) |

### Ours−baseline: Followup Raw EM

| Baseline | Raw Δ pp [95% CI] | Normalized Δ pp [95% CI] | Raw−Normalized gap change pp |
|---|---:|---:|---:|
| ReComp | +0.000 [+0.000, +0.000] | -2.302 [-3.052, -1.540] | +2.302 |
| FullLoad | +0.000 [+0.000, +0.000] | -2.204 [-2.951, -1.451] | +2.204 |
| MPIC-32 | +0.000 [+0.000, +0.000] | -2.376 [-3.135, -1.607] | +2.376 |
| AllHead | +0.000 [+0.000, +0.000] | -2.105 [-2.868, -1.352] | +2.105 |

## Qwen: qwen_mt_gqa_prefixstore3_20261002T074856Z

398 images / 4,061 dialogues / 방법당 12,183 main 요청, 8,122 followups. 총 36,549요청.
무결성·membership·기존 Normalized EM 재계산 PASS. 요청별 저장 score 대조 36,549개, 없는 score 0개.
Frozen tokenizer.decode(skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(). Raw EM compares stored prediction after this existing decoder processing.

| Method | T1 Raw EM | T2 Raw EM | T3 Raw EM | All Raw EM | Followup Raw EM |
|---|---:|---:|---:|---:|---:|
| ReComp | 53.903% (2189/4061) | 56.218% (2283/4061) | 56.168% (2281/4061) | 55.430% (6753/12183) | 56.193% (4564/8122) |
| FullLoad | 53.903% (2189/4061) | 56.218% (2283/4061) | 56.193% (2282/4061) | 55.438% (6754/12183) | 56.205% (4565/8122) |
| Ours | 53.903% (2189/4061) | 52.524% (2133/4061) | 52.721% (2141/4061) | 53.049% (6463/12183) | 52.623% (4274/8122) |

| Method | All Normalized / Raw % | 차이 pp | Followup Normalized / Raw % | 차이 pp | normalized=1/raw=0 followups |
|---|---:|---:|---:|---:|---:|
| ReComp | 61.159 / 55.430 | 5.729 | 61.364 / 56.193 | 5.171 | 420/8122 (5.171%) |
| FullLoad | 61.184 / 55.438 | 5.746 | 61.401 / 56.205 | 5.196 | 422/8122 (5.196%) |
| Ours | 58.828 / 53.049 | 5.779 | 57.868 / 52.623 | 5.245 | 426/8122 (5.245%) |

### Ours−baseline: Followup Raw EM

| Baseline | Raw Δ pp [95% CI] | Normalized Δ pp [95% CI] | Raw−Normalized gap change pp |
|---|---:|---:|---:|
| ReComp | -3.571 [-4.315, -2.830] | -3.497 [-4.260, -2.743] | -0.074 |
| FullLoad | -3.583 [-4.323, -2.856] | -3.534 [-4.302, -2.784] | -0.049 |

## GLM: glm46v_prefixstore3_v6_20261007T104505Z

398 images / 4,061 dialogues / 방법당 12,183 main 요청, 8,122 followups. 총 36,549요청.
무결성·membership·기존 Normalized EM 재계산 PASS. 요청별 저장 score 대조 0개, 없는 score 36,549개.
Frozen native-final-answer-box-v1 extracts a closed native answer span; malformed/plain output uses literal decode. skip_special_tokens=True, clean_up_tokenization_spaces=False, then strip(). Actual native assistant body is a separate saved history field. Raw EM compares the stored scoring prediction, not the token stream or history body.

| Method | T1 Raw EM | T2 Raw EM | T3 Raw EM | All Raw EM | Followup Raw EM |
|---|---:|---:|---:|---:|---:|
| ReComp | 21.029% (854/4061) | 18.197% (739/4061) | 19.010% (772/4061) | 19.412% (2365/12183) | 18.604% (1511/8122) |
| FullLoad | 21.029% (854/4061) | 18.197% (739/4061) | 19.035% (773/4061) | 19.421% (2366/12183) | 18.616% (1512/8122) |
| Ours | 21.029% (854/4061) | 17.065% (693/4061) | 17.927% (728/4061) | 18.674% (2275/12183) | 17.496% (1421/8122) |

| Method | All Normalized / Raw % | 차이 pp | Followup Normalized / Raw % | 차이 pp | normalized=1/raw=0 followups |
|---|---:|---:|---:|---:|---:|
| ReComp | 61.216 / 19.412 | 41.804 | 61.635 / 18.604 | 43.031 | 3495/8122 (43.031%) |
| FullLoad | 61.159 / 19.421 | 41.738 | 61.549 / 18.616 | 42.933 | 3487/8122 (42.933%) |
| Ours | 58.852 / 18.674 | 40.179 | 58.089 / 17.496 | 40.593 | 3297/8122 (40.593%) |

### Ours−baseline: Followup Raw EM

| Baseline | Raw Δ pp [95% CI] | Normalized Δ pp [95% CI] | Raw−Normalized gap change pp |
|---|---:|---:|---:|
| ReComp | -1.108 [-1.682, -0.521] | -3.546 [-4.416, -2.692] | +2.438 |
| FullLoad | -1.120 [-1.706, -0.512] | -3.460 [-4.328, -2.597] | +2.339 |

## 판정이 달라진 사례

최대20개 deterministic 사례. 원본 문자열을 JSON 표기로 표시해 공백·구두점·대소문자를 보존했다. 두 판정은 Normalized=1, Raw=0이다.

| Model / method | Image / dialogue / turn | Prediction | Gold | 원인 |
|---|---|---|---|---|
| LLaVA / FullLoad | n100552 / mtgqa_002992 / T1 | "Fence" | "fence" | case_only |
| LLaVA / MPIC-32 | n100552 / mtgqa_002992 / T1 | "Fence" | "fence" | case_only |
| LLaVA / AllHead | n100552 / mtgqa_002992 / T1 | "Fence" | "fence" | case_only |
| LLaVA / Ours | n100552 / mtgqa_002992 / T1 | "Fence" | "fence" | case_only |
| LLaVA / ReComp | n100552 / mtgqa_002992 / T1 | "Fence" | "fence" | case_only |
| Qwen / Ours | n100552 / mtgqa_003000 / T2 | "Yes" | "yes" | case_only |
| Qwen / ReComp | n100552 / mtgqa_003000 / T2 | "Yes" | "yes" | case_only |
| Qwen / FullLoad | n100552 / mtgqa_003000 / T2 | "Yes" | "yes" | case_only |
| GLM / ReComp | n100552 / mtgqa_002995 / T3 | "No" | "no" | case_only |
| GLM / FullLoad | n100552 / mtgqa_002995 / T3 | "No" | "no" | case_only |
| GLM / Ours | n100552 / mtgqa_002995 / T3 | "No" | "no" | case_only |
| GLM / Ours | n111390 / mtgqa_002709 / T2 | "A table" | "table" | case_articles_or_spacing |
| GLM / ReComp | n111390 / mtgqa_002709 / T2 | "A table" | "table" | case_articles_or_spacing |
| GLM / FullLoad | n111390 / mtgqa_002709 / T2 | "A table" | "table" | case_articles_or_spacing |
| GLM / Ours | n117888 / mtgqa_003066 / T1 | "Yes." | "yes" | case_punctuation_or_spacing |
| GLM / ReComp | n117888 / mtgqa_003066 / T1 | "Yes." | "yes" | case_punctuation_or_spacing |
| GLM / FullLoad | n117888 / mtgqa_003066 / T1 | "Yes." | "yes" | case_punctuation_or_spacing |
| GLM / FullLoad | n167552 / mtgqa_001385 / T3 | "A chair." | "chair" | combined_normalization |
| GLM / ReComp | n167552 / mtgqa_001385 / T3 | "A chair." | "chair" | combined_normalization |
| GLM / Ours | n179136 / mtgqa_003479 / T1 | "The surfer." | "surfer" | combined_normalization |

## 해석·기존 결과 처리

기존 Normalized EM 수치가 모든 방법·범위에서 재계산과 일치하므로 그 수치를 잘못된 결과로 수정할 이유는 없다. 기존 표의 지표명을 Normalized EM으로 유지하고 Raw EM을 보조 지표로 병기한다.
Raw EM은 답변의 표기 형식에도 민감하다. 원인 빈도는 raw_em_difference_categories.csv, 원본 예시는 raw_em_examples.csv를 참조한다. Ours−FullLoad의 gap 변화는 위 표와 raw_em_paired.csv에서 확인한다.
공식 GQA와 동일 evaluator 또는 MetaCompress의 정확한 dialogue/normalization/history 재현을 주장하지 않는다. 외부 논문의 발표 수치를 이 분석으로 수정할 수 없다.
CI는 seed1234/10,000회/398paired image clusters의 percentile95%다. 동일 이미지의 모든 요청과 방법을 함께 resample하고 pooled sum/count로 요청 가중치를 유지한다. CI가0을 포함해도 동등성을 뜻하지 않는다.
GLM은 원래 raw에 score가 없으므로 요청별 기존 score 일치를 주장하지 않는다. 저장 prediction을 직접 재채점하고 기존15개 Normalized EM 집계 및10개 paired accuracy CI를 확인했다.

## 산출물·재현

- raw_em_summary.csv: 모든 scope의 정답수/요청수/Raw·Normalized EM.
- raw_em_paired.csv: Raw·Normalized EM Δ와95%CI, Ours−baseline gap 변화.
- raw_em_sensitivity.csv: 정규화로 추가 정답이 된 count/rate.
- raw_em_examples.csv: 전체20개 이내 원본 사례.
- raw_em_requests.csv: adopted 요청별 두 score와 원본prediction/gold.
- raw_em_validation.json / raw_em_evidence_index.json / source_files_sha256.json: full hashes·무결성·읽기 범위.
- analyze_raw_em.py: numpy와표준라이브러리만 사용. 모델/토크나이저/추론 코드를 import하지 않는다.

기존 결과 디렉터리를 overwrite하지 않고 없는 새 경로로 재현한다.

```bash
cd /home/dblab/hj/mllm_v2
.venv-glm46v/bin/python results/mt_gqa_raw_em_20261013/analyze_raw_em.py \
  --root /home/dblab/hj/mllm_v2 --output results/mt_gqa_raw_em_REPLAY_NEW
```

## 추가 독립 검산과 핵심 해석

별도 verify_and_finalize.py가 exported prediction/gold 문자열134,013개를 다시 직접 비교하고, regex scorer를 import하지 않는 문자 기반 정규화로55개 summary를 검산했다. Followup Raw EM paired CI8개도 별도 choice-resampling으로 재계산해 모두 일치했다.
LLaVA는 T1/T2/T3/All/Followup에서 모든 방법 Raw EM0이다. Normalized 정답40,435개는 전부 대소문자만 다르다. 이것은 저장 답변의 표기 때문에 Raw EM이 바닥값에 머문 결과다. Δ0/CI[0,0]는 품질 동등성 증거가 아니며 압축 품질을 구분하지 못한다.
Qwen도 정규화로 추가 정답이 된 모든 요청은 대소문자 차이다. GLM은 대소문자 외 관사·구두점 등의 차이도 존재한다. 진단 변환은 원인 분류에만 사용했고 Raw EM의 비교 문자열은 바꾸지 않았다.
LLaVA의 pinned tokenizer_config는 clean_up_tokenization_spaces=false다. LLaVA decoder는 skip_special_tokens 후 strip을 이미 적용했다. Qwen은 cleanup=false를 decode 호출에 명시하고 strip을 적용했다. GLM은 native final-answer span 추출 후 strip한다. 추가 metadata는 decoder_preprocessing_metadata.json에 hash와 함께 보존했다. 이 Raw EM은 이 처리 이후의 저장 prediction을 평가한다.

| Model | Ours−FullLoad Followup Normalized pp | Raw pp | Gap 변화 pp [95% CI] |
|---|---:|---:|---:|
| LLaVA | -2.204 | +0.000 | +2.204 [+1.451, +2.951] |
| Qwen | -3.534 | -3.583 | -0.049 [-0.360, +0.258] |
| GLM | -3.460 | -1.120 | +2.339 [+1.596, +3.118] |

기존 연구 표가 Normalized EM이라고 명시돼 있다면 기존 정확도 숫자를 교체할 필요는 없다. Raw EM을 함께 보고해 표기 민감도를 공개한다. 만약 manuscript에서 그 숫자를 Raw EM 또는 공식 GQA/MetaCompress의 동일 평가 결과라고 설명했다면 해당 지표명과 재현 주장은 수정해야 한다. 기존 run/report/code는 이번 작업에서 수정하지 않았다.

전체 재현에서는 앞의 분석 명령에 이어 다음 검산·보충 명령을 실행한다.

```bash
.venv-glm46v/bin/python results/mt_gqa_raw_em_20261013/verify_and_finalize.py \
  --analysis results/mt_gqa_raw_em_REPLAY_NEW
```
