# LLaVA-1.5-7B 포팅 + MT-GQA HOLD 계약 (ReComp / U100 / U50 / U25 / PYR-b20 packed)

작성 2026-11-12. 프롬프트 `papers/llava15_mt_gqa_prompt.md`. 이 문서는 gate·smoke·main 전에 SHA256으로 freeze한다
(`runs/llava15_mt_gqa_<ts>/config.json`의 `contract_sha256`). freeze 뒤의 변경은 `freeze_amendment_<n>.json`에 원인·수정을
남기고 `177 --stage refreeze`로 다시 freeze한다(이전 기록 보존, 모든 gate 재실행).

## 1. 목적

- 이미지당 KV 크기 축의 중간 점을 실제 모델로 얻는다: Qwen2.5-VL 약 22 MB < **LLaVA-1.5-7B 약 302 MB** < LLaVA-NeXT-7B 약 1,174 MB.
- LLaVA-NeXT의 방법(Turn-1 piggyback 저장, image-only VisionZip 순서, 그 순서의 앞 k_l행만 layer l에서 사용, 이미지당 packed 파일 1개·
  hit당 preadv 1회, dense 버퍼 + layer별 mask)을 LLaVA-1.5-7B로 옮겨 MT-GQA HOLD에서 측정한다.
- PyramidKV(β=20) layer 배분이 세 번째 모델에서도 통하는지 같은 규칙으로 확인한다.
- 질문은 선택에 쓰지 않는다. 데이터로 무엇을 맞추거나 고르는 과정은 없다. HOLD를 보고 비율·판정 기준을 바꾸지 않는다.

## 2. 모델과 조건

- `llava-hf/llava-1.5-7b-hf`, revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`(로컬 HF cache snapshot, refs/main과 동일),
  `LlavaForConditionalGeneration`, Vicuna-7B v1.5(MHA, 32 layers × 32 heads × 128), CLIP ViT-L/14-336.
- 이미지당 시각 토큰 576개 고정(24 × 24 patch, `vision_feature_layer = -2`, `vision_feature_select_strategy = "default"`가 CLS를 제외),
  연속 1구간, separator 0개, `image_sizes` 없음. `image_token_index = 32000`.
- NF4 double quant / BF16 compute / eager attention / FP16 SSD KV / seed 1234 / batch 1 / greedy max 16 / CPU thread 2.
- 환경: `/home/dblab/anaconda3/envs/mllm_ft/bin/python`(conda env mllm_ft), transformers 4.57.6, bitsandbytes 0.49.2, torch 2.5.1+cu121.
  실제 값은 Phase 0의 `phase0/model_record.json`에 기록한다.
- 이미지 전처리는 HF 기본 processor 그대로이며 바꾸지 않는다: `CLIPImageProcessor`(RGB 변환, 짧은 변 336 bicubic resize, 336 × 336 중앙 자르기,
  1/255 rescale, CLIP mean/std normalize). 설정 전체를 `model_record.json`에 기록한다. 원 LLaVA-1.5 저장소는 정사각형 패딩이므로 공개 GQA 수치(61.9)와
  조건이 다르다. 모든 arm이 같은 전처리를 쓴다.
- 프롬프트: runner의 단일 턴 템플릿은 `USER: <image>\n{질문} Answer the question using a single word or phrase. ASSISTANT:`이다.
  HOLD의 모든 요청은 기존 멀티턴 형식(LLaVA-NeXT HOLD와 동일)을 쓴다:
  `USER: <image>\n[Q1: ..\nA1: ..\n..\n\n]Current question Q{t}: {질문}\nAnswer the current question with a single word or short phrase. ASSISTANT:`.

## 3. Arm과 벡터 (이 순서로 freeze, 전부 필수)

| # | arm | 내용 | layer당 행 (N = 576) | payload |
|---|---|---|---|---:|
| 1 | `recompute` | 매 턴 full pixel | – | 0 |
| 2 | `llava15_kv_U100_packed` | 균등 100% | 576 | 301,989,888 B |
| 3 | `llava15_kv_U50_packed` | 균등 50% | 288 | 150,994,944 B |
| 4 | `llava15_kv_U25_packed` | 균등 25% — **주 운영점(사전 고정)** | 144 | 75,497,472 B |
| 5 | `llava15_kv_PYR-b20_packed` | 총 25%, PyramidKV β=20 원식 layer 배분, 하한 없음 | 281, 272, 263, …, 34, 25, 16, 7 (합 4,608) | 75,497,472 B |

실행할 수 없는 arm이 있으면 BLOCKED로 멈춘다. 회전은 `ordinal % 5`.

벡터 ratio SHA256(불일치 시 멈춤):

| 벡터 | ratio | total_ratio | ratio SHA256 |
|---|---|---|---|
| U100 | [1.0] × 32 | 1.0 | `04c01bedee671a22789fcaedf3348f178446db6f914bfdb98bb1699ab64bc114` |
| U50 | [0.5] × 32 | 0.5 | `673c2550205b17ed2a56548d2c9524811150da3c49cc4b370f80d05dec1201b0` |
| U25 | [0.25] × 32 | 0.25 | `54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba` |
| PYR-b20 | PyramidKV run의 PYR-beta-raw ratio | 0.25 | `51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724` |

- PYR-b20은 `results/layer_budget_kv25_pyrbeta_20261110T073604Z/layer_budget_llava_PYR-beta-raw.json`(파일 SHA `8bfc4cbc…`)을
  run의 `vectors/`에 그대로 복사하고 ratio만 가져온다. 새로 계산하지 않는다.
- `mmimpress.layer_budget`의 모든 호출은 `num_layers=32, total_ratio=해당 비율, r_min=0.0, r_max=1.0`을 keyword로 명시한다.
  모듈 기본값(r_min 0.10, r_max 0.60, total_ratio 0.25)에 기대는 호출이 있으면 LG8에서 걸린다. 복사한 PYR-beta-raw 문서의 clip 범위는
  [0.0, 0.6]이지만 최대 ratio가 0.4875이므로 k_l은 [0.0, 1.0]에서도 같다(Phase 0에서 N = 576으로 확인). 이 포팅의 budget 문서·store meta는
  clip 범위 [0.0, 1.0]만 허용하고 다른 값은 fail closed한다.
- k_l = Hamilton largest-remainder(합 = 32·ceil(total_ratio·N)). 독립 구현(정수 분수, 라이브러리 미사용)과 비교한다.
- separator가 없으므로 packed 파일 크기 = Σ_l 2·k_l·8,192 B이다(separator sidecar 0 B).

## 4. 데이터

MT-GQA reconstructed: 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm, main 60,915 요청.
INDEX_SHA `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`,
WORKLOAD_SHA `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`.
manifest는 `runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/llava/manifest.json`(dialogues, image identity)이며 smoke dialogue id는
LLaVA-NeXT HOLD와 같다(`mtgqa_000001, 000012, 000034, 000038`; 4 dialogues × 3 turns × 5 arm = 60 요청).
gate fixture: manifest의 앞 10개 이미지와 각 이미지의 첫 대화(30 요청). 정합성 fixture이며 품질 holdout이 아니다.

## 5. 재사용한 함수와 새로 구현한 함수

기존 소스(`mmimpress/*.py`, `mmimpress/qwen25`, 기존 scripts·tests)는 수정하지 않는다. 새 파일만 추가한다:
`mmimpress/llava15/{__init__,runner,store,serve}.py`, `scripts/llava15_support.py`, `scripts/177–180_*llava15_mt_gqa*.py`,
`tests/test_llava15_{store,harness}.py`, 이 문서. 산출물은 `runs/llava15_mt_gqa_<ts>/`, `results/llava15_mt_gqa_<ts>/`에만 쓴다.

**그대로 import해 재사용:**
- `mmimpress.model.LlavaRunner`(상속: prompt, encode, to_device, visual_spans, layers 등), `cache_layers`
- `mmimpress.cvpr25.clip_cls_patch_saliency`(gate의 vision-only 확인), `visionzip_repack_order`, `permutation_sha256`
- `mmimpress.piggyback.VisionForwardCapture`(Turn-1 saliency capture), `_fsync_staging_tree`, `_rename_noreplace`, `_fsync_directory`,
  `_store_file_sizes`, `_sampled_store_sha256`, `sha256_file`
- `mmimpress.layer_budget.{validate_ratios, layer_keep_counts, total_row_budget, budget_sha256}`(모든 인자 명시)
- `mmimpress.packed_store.{packed_layout, PackedReader, PACKED_NAME}`(preadv 1회, pinned buffer), `mmimpress.packed_serve.PackedReadTrace`
- `mmimpress.serve.{Server.recompute, Server._decode, PrefixCache, bias_from_keep, BIAS, IOCounter, suffix_ids_from_prompt}`
- `mmimpress.store.{load_meta, META_NAME}`, `mmimpress.prefix25.{seal_integrity, verify_integrity, inventory, process_io}`
- `mmimpress.recompute_kv25.{paired_bootstrap, sha_manifest, stat_inventory, guarded_rmtree}`, `mmimpress.mt_gqa.workload_sha256`
- `scripts/156`의 lifecycle helper: `frozen_dialogues, method_order, strict_score, assert_gpu_exclusive, conditioning_trace, PilotCounts,
  file_hashes, Checks, capacity_guard, cleanup_committed, atomic_json, append, load_jsonl, canonical_hash, safe, sha`
- `scripts/49`의 `_combined_processor, _suffix_from_tokenized, _NoVisionForward`, `scripts/main_b_support`의 `persist_llava, source_hashes`

**LLaVA-NeXT에 묶여 있어 새로 구현(같은 동작, 다른 모델 제약):**
- `Llava15Runner.load`(`LlavaForConditionalGeneration`), `image_inputs`(`image_sizes` 없음), `visual_span`(576 강제), `flat_layout`(separator 없음),
  `prefix_forward`; `anyres_layout`은 호출 시 예외
- `mmimpress.llava15.store`: `make_budget / validate_budget / keep_counts`, `write_llava15_packed_store`(기존 `write_packed_store`의 `(N + 3) // 4` assert 대신
  문서의 total_ratio), `validate_llava15_packed_meta`(기존 validator는 0.25 고정), `token_scores_from_capture`(AnyRes 매핑 없음),
  `persist_captured_llava15_prefix`(기존 `persist_captured_visual_prefix`는 `anyres_layout`·separator sidecar 전제)
- `mmimpress.llava15.serve`: `Llava15PackedContext`, `Llava15PrefixSelector`(기존 `PackedPrefixSelector`는 budget ≠ 0.25 거부; separator scatter 제거),
  `request_llava15_packed`(timing 경계는 `request_packed`와 동일)
- `scripts/llava15_support`: `prompt_factory`·`request_id`(모델 id), `normal_request`·`image_input_hash`(`image_sizes` 없음), `stored_hit`·`stored_request`,
  `persist`, `activate`, `warmup`, 독립 참조(`dense_reference_cache`, `full_length_reference_cache`, `MaskHooks`, `reference_decode`),
  독립 검사(`independent_counts`, `check_packed_slices`, `check_masks`, `check_cache_bits`)
- `scripts/179`: 164의 `make_row / audit_image_rows / run_image / run_phase / verify_freeze`를 arm 5개·비율 축에 맞게 옮긴 것
  (요청 생성·회전·history·채점·image-atomic commit·감사·cleanup 규칙은 동일)

## 6. Gate (모두 PASS여야 main. 실패 gate 우회·임계값 변경·표본 교체 금지)

검증 표본은 4절의 gate fixture(10 이미지 × 첫 대화 × 3턴 = 30 요청)이다.

- **LG1 runner 동치(GPU)**: 새 runner의 recompute 경로(`Server.recompute`)가 같은 입력에 대한 HF `model.generate(do_sample=False, max_new_tokens=16)`와
  생성 token ID 전부 동일(30 요청, history는 recompute의 generated history).
- **LG2 geometry·순서(GPU)**: 시각 span이 576개 연속 1구간, separator 0개, `image_sizes` 없음; CLS saliency가 (1, 576) 유한값; 순서가 0..575의 순열이고
  stable 내림차순; 같은 이미지에서 질문을 바꾼 T2·T3 요청의 순열 SHA와 vision tower만 호출한 재계산(`clip_cls_patch_saliency`, 2회)의 순열 SHA가 모두 동일;
  vision-only 계산 중 decoder layer 호출 0.
- **LG3 capture 정확성(GPU)**: T1에서 캡처한 prefix KV(`[system | 576 시각 토큰]`)가 독립 forward의 prefix KV와 32 layer × K/V 모두 bitwise 동일.
  독립 forward는 **T1과 같은 길이의 입력에서 질문 자리(prefix 뒤 전체)를 filler 토큰으로 바꾼 것**(pad 토큰 / token id 0 / seed 1234 무작위 id, 3종)을
  generate·hook 없이 `model(...)`로 한 번 실행한 것이다. 질문 행의 KV가 T1과 다름도 함께 확인한다.
  - 프롬프트 문구의 "prefix-only forward"를 문자 그대로(길이 P로 자른 짧은 forward) 구현하면, BF16 GEMM의 반올림이 tensor shape에 의존하므로
    causal하게 같은 행이라도 layer ≥ 1에서 bitwise가 성립하지 않는다(이 저장소의 선례: `scripts/30_validate_visdial_turn1_piggyback.py`의
    `bitwise_identity_required: False`, `docs/recompute_kv25_ablation_contract.md`의 noise floor; 이번 개발 probe에서 31 / 32 layer 불일치, rel-L2 약 0.009 확인).
    그래서 이 계약은 freeze 전에 참조를 "같은 shape의 질문 제거 forward"로 정의한다. **기준(bitwise, 전 layer, K/V)은 그대로다.**
    문자 그대로의 짧은 forward는 LG3-literal 진단으로 실행해 layer 0 bitwise 여부와 layer별 rel-L2를 기록한다(gate 아님). 이 차이는 보고서의
    "프롬프트와 다르게 한 점"에 적는다.
- **LG4 저장 정확성(GPU)**: 각 벡터의 packed 파일에서 layer l slice가 캡처 KV를 FP16으로 바꾼 뒤 순서표 앞 k_l행을 뽑은 것과 bitwise 동일
  (offset은 검사 쪽에서 다시 계산), 파일 크기 = 3절의 기대 바이트(separator 0 B), packed SHA256 재계산 일치. validator·activation이 변조된 meta
  (k_l, r_min, total_ratio, ratio SHA), 다른 벡터, 잘린 파일, payload SHA 불일치를 거부.
- **LG5 hit 경로 구현(GPU, 벡터 4개)**: 실제 SSD 경로 / capture만으로 독립적으로 만든 같은 shape의 in-memory 참조(BIAS 표를 쓰지 않는 자체 mask hook,
  자체 greedy loop) / SSD 경로 반복 실행의 첫 logits bitwise 동일, 생성 ID 전부 동일. forward 직전의 production cache 비트가 capture의 FP16 왕복과 같고
  (k_l 밖 행은 0), 설치된 layer별 mask가 독립 계산한 "순서표 앞 k_l개" 가시 집합과 bit 단위로 같다.
- **LG6 의미 검증(GPU)**: 독립 구현으로 "576행이 모두 실제 값인 전체 길이 cache에 layer별 가시 집합을 mask로만 적용한 참조"를 만들어 production과
  FP32 첫 토큰 logits `atol = rtol = 1e-4`, 첫 토큰·생성 ID 동일(기존 LLaVA G2와 같은 기준·같은 저장 순서 배치). 같은 가시 집합을 원래 token 순서로
  배치한 참조는 key 합산 순서가 달라 BF16 반올림 차이가 생기므로 최대 차·토큰 일치만 진단으로 기록한다(gate 아님).
  U100은 추가로 recompute와 첫 토큰 일치율을 fixture 30 요청에서 보고하고 95% 이상이어야 한다(불일치는 top-2 logit 차와 함께 원인을 기록).
- **LG7 I/O(GPU)**: hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기, vision 호출 0·score 호출 0, 질문만 바꿔도(T1·T2·T3) read range와 선택 ID 불변.
- **LG8 CPU**: 벡터 SHA·합; N = 576에서 Σ k_l = 32·ceil(r·576)(라이브러리 = 독립 구현 = 3절 표); 새 소스의 `layer_budget` 호출이 모두 인자를 명시
  (AST 검사), LLaVA-NeXT 0.25 경로 함수 미호출; meta의 r_min·r_max·total_ratio를 기본값으로 바꾸거나 빼면 거부; negative fixture
  (한 layer k_l ± 1, 두 layer 행 뒤바꿈 — meta의 count와 디스크의 block 각각, 가려야 할 행 1개 노출, 다른 벡터 SHA).
- **LG9 회귀**: 기존 CPU 테스트 전체 통과(GLM 테스트 모듈은 분리 실행해 기존 실패를 따로 보고). 기존 소스 무수정(protection before의 SHA와 비교,
  직전 run의 protection after와도 비교).
- **LG10 통합**: smoke dialogue에서 5 arm generated-history 3턴 실행(`179 --stage integration`, 60 요청) + 독립 재감사 PASS.

비 gate 진단: H2D 분해(FP16→BF16 host 변환 / H2D / GPU index write / mask)를 fixture에서 계측용 복제 경로로 잰다. HOLD 측정이 아니다.

## 7. 실행 (Phase 2)

smoke(60 요청) 후 main(398 images, 60,915 요청). 각 arm은 자기 full-pixel T1을 실행하고, 이미지의 첫 source dialogue의 T1에서만
capture·store 생성(arm별 store 4개), method-local generated history(gold 미사용), strict normalized EM, image-atomic attempt,
이미지별 독립 감사 후 run-local scratch store만 정확한 SHA allowlist로 삭제. timed 요청 중 다른 GPU 작업·무거운 감사 금지
(요청마다 GPU 독점 확인). 중단 시 `--resume`. 요청 수 완주·실패 0·감사 PASS여야 final.

timing 경계(LLaVA-NeXT HOLD와 동일, true TTFT): 요청 시작은 프롬프트 구성 전. pixel 요청(T1, recompute)은 이미지 디코딩 + HF processor + vision +
전체 prefill을 포함한다. hit은 토큰화, dense cache 할당, preadv 1회, scatter + mask, suffix prefill을 포함한다. `posix_fadvise(DONTNEED)`는 timer 밖이다
(성공 여부를 기록·감사). persistence·activation은 요청 timer 밖에서 따로 잰다.

## 8. 보고와 판정 (실행 전 고정. HOLD를 본 뒤 바꾸지 않는다)

공통: HOLD followup(T2–T3), 이미지 단위 paired bootstrap 95% CI(10,000회, seed 1234). CI가 0을 포함하면 "차이 미입증"으로 쓴다.

- **(A) 주 운영점**: U25_packed vs recompute — 정확도 Δ·CI, 원래 성능 대비 %, TTFT 평균/p50/p95와 비율, SSD MB/followup.
- **(B) KV 크기 축 지표(보고, 판정 아님)**: R100 = TTFT(U100_packed) / TTFT(recompute), S25 = TTFT(U25_packed) / TTFT(U100_packed),
  S50 = TTFT(U50_packed) / TTFT(U100_packed) (평균 TTFT의 비). 참고값: Qwen2.5-VL R100 0.462·S25 0.892(같은 packed 경로),
  LLaVA-NeXT-7B R100 1.39·S25 0.34(v1 run, 둘 다 파일 분리 경로라 직접 비교가 아님).
  사전 기록 가설: LLaVA-1.5의 R100은 0.462와 1.39 사이, S25는 0.34와 0.892 사이. 맞았는지 틀렸는지를 그대로 적는다.
- **(C) layer 배분 판정**: Δ = PYR-b20_packed − U25_packed.
  GO: Δ ≥ +1.0 pp & CI 0 제외 & SSD MB ≤ U25 × 1.02 & TTFT 평균 ≤ U25 × 1.05.
  PARTIAL: Δ ≥ +0.5 pp이고 CI 0 포함, 또는 GO에서 MB/TTFT 중 정확히 하나 위반. NO-GO: 그 외(숫자는 그대로 함께 적는다).
- **(D) 곡선**: 100/50/25%의 정확도·TTFT·MB 표와 그림(가로 TTFT, 세로 정확도, recompute 포함, 오차 막대 = 95% CI).
- **(E) 구간별 시간**: read, H2D·scatter(합산; 분해는 fixture 진단), suffix prefill ms(arm별), persistence ms/이미지, store MB/이미지,
  session(shared-stream 실측 / standalone-equivalent DERIVED).
- **(F) 정상성 참고**: T1 정확도(모든 arm 동일해야 함). 공개 수치(61.9)와는 직접 비교가 아니며 판정에 쓰지 않는다.
- **정합성 사후 확인(보고 전 필수)**: 모든 arm의 T1 생성 토큰이 같은 요청에서 서로 일치해야 한다. 아니면 불일치를 덤프하고 상태를 PENDING_REVIEW로 둔다.

판정·기준값은 frozen `scripts/180`이 `summary.json`과 `decision.json`에 기계적으로 기록한다. 산출물: REPORT.md, CONTRACT.md, summary.json,
decision.json, 벡터 json 4개 + SHA, quality / latency_io / sessions / storage / paired_analysis CSV·JSON, gate 결과, independent_audit, REPRODUCE.md,
protection before/after, 삭제 로그, 그림, RESULT_RECORD.md.

## 9. 저장 계획과 보호

- 디스크 여유는 freeze 시점에 실측해 `storage_plan.json`에 기록한다(이 문서 작성 시 `df` 기준 약 257 GB 여유).
- 이미지당 scratch payload 603,979,776 B(약 604 MB) + 작은 파일(< 2 MB). 한 번에 이미지 1장만 유지한다: store 4개 생성 → hit → 감사 → commit →
  정확한 SHA allowlist로 삭제. 필요 여유 64 GiB(reserve 30 + 작업 4 + 실패 attempt 보존 20 + raw·evidence 8 + gate 2) 미만이면 BLOCKED_STORAGE로 멈춘다.
  기존 산출물은 어떤 경우에도 삭제하지 않는다.
- 삭제 범위: `runs/<RUN>/scratch/**`(감사 + commit 후 allowlist), `runs/<RUN>/gates/stores/**`(fixture 이미지별), `runs/<RUN>/gates/cpu_fixture`.
- 소스 보호 before inventory: results / docs / mmimpress / scripts / tests / README.md 전체 SHA256(새 경로 제외), kvstore*와 다른 runs/*의
  (size, mtime_ns, inode). 기존 소스는 직전 run(Qwen sweep)의 protection after와도 비교한다. 보고 시 after와 대조한다.
- GitHub에 없는 로컬 변경을 되돌리지 않는다. reset / clean / upgrade / commit / push를 하지 않는다.

## 10. 예상 소요 시간 (smoke 이후)

main 60,915 요청: 약 2.5–4.5 시간. 근거는 개발 rehearsal(측정 아님; fixture·smoke dialogue로 코드 동작만 확인)의 요청 간격이다:
dialogue 1개(15 요청)에 약 2.4 s, 이미지의 첫 dialogue는 store 4개 생성·activation·해시·감사로 약 3 s 추가 → 4,061 × 2.4 s + 398 × 3 s ≈ 3.0 h.
추정 범위는 freeze 시 `config.json`의 `expected_duration_after_smoke`에도 기록한다.
