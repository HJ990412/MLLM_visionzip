# LLaVA-NeXT-7B PyramidKV 배분(PYR-beta-raw packed) — MT-VQA-v2-reconstructed-25k 단일 arm 실행 계약

2026-11-15 (KST), 어떤 GPU 측정보다 먼저 freeze. 저장소 `/home/dblab/hj/mllm_v2` (사용 가능한 `.git` 없음; 소스 보호는 SHA256 manifest).
Run root `runs/llava_next_mtvqa_pyr_20261114T182331Z`, 결과 `results/llava_next_mtvqa_pyr_20261114T182331Z`.
참조 run(읽기 전용): v3 MT-VQA `runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z` / `results/…v3…`, PyramidKV MT-GQA `results/layer_budget_kv25_pyrbeta_20261110T073604Z`.
실행 프롬프트: `papers/llava_next_mtvqa_pyr_prompt.md`. 이 계약의 SHA256은 `runs/<RUN>/experiment_freeze.json`의 `contract_sha256`에 기록한다.

## 1. 목적

MT-GQA HOLD에서 GO였던 방법을 MT-VQA 25k에서 그대로 측정한다:
image-only VisionZip 순서(질문 미사용) + PyramidKV β=20 원식 layer 배분(PYR-beta-raw, 하한 없음) + 이미지당 packed 파일 1개·hit당 `preadv` 1회.
데이터로 벡터를 만들거나 고르는 과정은 없다. 기존 serving·store·packed reader 코드 경로를 바꾸지 않는다.

- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`. v3 MT-VQA run과 같은 설정:
  NF4 double quant / BF16 compute / eager attention / FP16 SSD KV / seed 1234 / batch 1 / greedy cap 16 / CPU Torch·OMP·MKL threads 2 / 기존 slow AnyRes 전처리.
- conda env `mllm_ft` (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`), GPU RTX 4090.

## 2. Arm

- **main arm 1개**: `ours_prefixstore_kv25_PYR-beta-raw_packed`.
- **gate 전용 arm**: `ours_prefixstore_kv25_U_packed` — 5절 VG5(G-det-U)의 pilot40 40장에서만 실행하며 어떤 main 분모에도 넣지 않는다.
- `recompute`, `fullload`, `sparsevlm_ssd_kv25_allhead`, `mpic32_ssd`, `ours_prefixstore_kv25`[B]는 실행하지 않는다.
  비교는 v3 run의 요청별 기록과 요청 단위(같은 image, 같은 turn)로 짝지어 한다. `mpic32_ssd`는 비교표에서 제외한다(기존 MPIC 결과는 지우지 않는다).
- arm은 이름으로 판별한다(`adapter.check_methods`, 164 `stored_methods`/`kind_of`). recompute가 첫 arm이라는 가정(`methods[0]`/`methods[1:]`)은 새 경로에 없다(VG3에서 확인).

## 3. 데이터·채점·벡터 (모두 v3/PyramidKV run에서 그대로; 불일치 시 멈춤)

- 데이터: v3 `llava/{smoke,pilot40,main}/manifest.json`을 복사(파일 SHA 일치). main 25,000 images, 이미지당 dialogue 1개, 3 turns → 75,000 요청.
  Ordered workload SHA256 `eb3c5e812fd6937cc0741cac6dd85d6839a7d00288dffbfa7ea8294272269cc8` (`data/mt_vqa_v2_reconstructed_25k/dialogues.json`에서 다시 계산, main manifest dialogues와 동일성 확인),
  Dataset config SHA256 `1b99478b0c0f253810453f169b91f9843973c05dd815b3cc46aeabcc2a07b933`. 25,000개 이미지 파일 SHA256 전부 확인.
  smoke 4장·pilot40 40장은 v3와 같은 이미지이며 main 25,000장의 부분집합이다(main에서 다시 실행되고, smoke·pilot40 요청은 main 분모에 넣지 않는다).
- 채점: v3와 같은 공식 evaluator(GT-Vision-Lab/VQA `a013f0043c1e2cdc995922dfe257f7149aa9af06`, `mmimpress/mtvqa25k/scorer.py` + vendor), ten-answer leave-one-out VQA score(%).
  v3 SCORER_VALIDATION의 fixture 18개를 다시 계산해 저장값과 일치해야 한다.
- 벡터: PyramidKV run의 json을 복사(새로 계산하지 않음).
  PYR-beta-raw ratio SHA256 `51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724` (문서 `r_min = 0.0`, `r_max = 0.6`, 합 8.0),
  U ratio SHA256 `54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba`.
- k_l: v3 raw에 기록된 25,000장의 실제 N(29개 값, 960–2880)으로 `layer_keep_counts(문서 r_min, r_max)`를 계산. Σ k_l = 32·ceil(N/4), k_l = 0인 layer 0개, 최소 k_l = 12행(layer 31, N = 960).
  사전 계산값과 일치: N=960 → 앞 468/453/439, 뒤 41/27/12 · N=1344 → 뒤 58/37/17 · N=2208 → 뒤 95/61/28 · N=2880 → 앞 1404/1360/1316, 뒤 124/80/36.
  `layer_budget` 호출은 항상 문서의 `r_min`·`r_max`를 인자로 넘긴다. 전역 기본값(`R_MIN = 0.10`)에 기대는 호출은 새 경로에 없다(VG3 정적 검사).
- 모델·환경: v3 `model_freeze.json`(snapshot 파일 14개 SHA·크기·경로)·`environment.json`(Python, torch, transformers, bitsandbytes, accelerate, numpy, Pillow, tokenizers 버전)과 현재가 같아야 한다.
  다르면 차이를 기록하고 멈춘다(정확도 재사용의 전제). 모든 GPU phase 시작 때 다시 확인한다.
- v3 요청별 raw: `runs/…v3…/llava/{smoke,pilot40,main}`의 이미지별 gzip journal을 전부 열어 commit의 raw SHA256과 대조하고 요약 index(`reference/v3_*_index.jsonl.gz`)를 만든다.
  main 25,000장 중 하나라도 읽을 수 없으면 BLOCKED_REFERENCE로 멈춘다. v3의 T1 생성 token이 5 arm에서 같은지도 함께 확인한다.

## 4. 코드: 재사용과 신규

기존 소스(`mmimpress/*.py`, `mmimpress/mtvqa25k`, 기존 scripts 전부)는 수정하지 않는다. 새 파일만 추가한다.

재사용(수정 없음, 호출만):
- 요청 경로: `scripts/164_eval_layer_budget_pyrbeta_hold.py`의 `run_image`, `prompt_factory`(156), `normal_request`(156; T1 full pixel + capture), `stored_request`/`stored_hit`(packed hit),
  `persist`/`activate`, `make_row`, `audit_image_rows`, `doc_counts`/`independent_counts`, `capacity_guard`; `mmimpress/{packed_store, packed_serve, layer_budget, prefix25, serve, piggyback}.py`; runner 로딩 `97.load_runner`, warm-up `49._warmup`.
- MT-VQA 덧씌우기: `mmimpress/mtvqa25k/adapter.py`의 `enrich`, `read_rows`, `score_from_strings`, `archive_before_commit`, `remove_archived_duplicates`, `dialogues_for_base`; `mmimpress/mtvqa25k/{scorer, dataset, common}.py`.
- 분석: `scripts/142_analyze_mtvqa25k.py`의 `bootstrap`(RandomState seed 1234, 10,000회 image-cluster, pooled sums/counts)을 그대로 불러 쓴다.
- gate: `scripts/166`의 `pyr_case`/`packed_traced`/`_fixture32`/`_expect_fail`/`_default_reliant_calls`, `scripts/154`의 `stage_cpu`·독립 참조 함수, `scripts/161`, `scripts/106`, `scripts/packed_support_v2.py`.

신규:
- `mmimpress/mtvqa25k_packed/adapter.py` — `bind_packed`: 164를 별도 module instance로 불러 `adapter.bind('llava')`가 108에 하는 것과 같은 것을 덧씌운다:
  SCHEMA(`mtvqa25k-llava-packed-layerbudget-v1`)·request_id, `make_row → enrich`(공식 VQA score, annotation, native decode 필드, `generated_token_ids`),
  감사의 공식 채점 재검(`independent_score` + `score_from_strings`, annotation·scorer_answer·생성 ID·dataset 일치), store meta `dataset = 'MT-VQA-v2-reconstructed-25k'`(writer 호출·옵션은 164와 동일),
  raw journal gzip 보관(`archive_before_commit`), `cleanup_owned`(이 run root에 묶인 fail-closed 삭제).
  164의 구조 감사는 score가 strict EM이라고 가정하므로, **EM으로 다시 채점한 사본**에 대해 실행하고(GLM `glm_official_auditor`와 같은 방식) 채택되는 행의 공식 VQA score는 별도로 독립 검사한다.
  164의 MT-GQA 전용 상수(INDEX_SHA, WORKLOAD_SHA, PRIOR, CONTRACT, `freeze`, `verify_freeze`, `run_phase`, `completed_images`, `frozen_dialogues`)는 호출하지 않는다.
- `mmimpress/mtvqa25k_packed/{checks, reference, analysis}.py` — CPU 동일성 검사, v3 raw index, 독립 감사·짝지은 분석·판정.
- `scripts/181_freeze_mtvqa_pyr.py`(Phase 0), `182_validate_mtvqa_pyr.py`(gate), `183_run_mtvqa_pyr.py`(138과 같은 수명주기: freeze·환경·admission 확인, 이미지 단위 실행, gzip 보관, `--resume`),
  `184_analyze_mtvqa_pyr.py`(독립 감사·분석·판정·보고), `tests/test_mtvqa_pyr_{adapter,analysis}.py`.

요청 단위 동일성(프롬프트 문구, 멀티턴 형식, history 구성, 디코딩, 채점)은 코드가 같다는 주장으로 끝내지 않고 5절 VG5와 7절 G-det-T1에서 v3 raw와 token 단위로 대조한다.

## 5. Timing 경계 (v3 MT-VQA 계약과 동일)

- TTFT: history/input/image decode 이전(`started`)에 시작해 첫 native token의 CUDA 동기화(`first_token_at_s`)에서 끝난다. E2E: runner 반환까지.
- persistence·activation: outer caller wall(`outer_helper_wall_ms`, `outer_wall_ms`). T1의 capture 비용은 T1 요청 E2E에 이미 들어 있고 다시 더하지 않는다.
- DONTNEED(`reader.drop_all`)는 timer 밖이며 요청마다 파일별 성공을 기록한다. cold NAND를 보장하지 않는다.
- 164와 108의 경계 비교: T1(`normal_request`)은 두 스크립트가 같은 순서·같은 경계다(이 arm의 T1은 v3 [B] arm의 T1처럼 KV·saliency를 capture한다).
  followup(`stored_hit` vs `legacy_hit`)도 시작·끝 지점이 같다. 다른 것은 timer 안에서 도는 내용뿐이다:
  (a) 읽기 경로가 `request_cvpr25`(파일 65개 pread) 대신 `request_packed`(preadv 1회), (b) 읽기 추적기가 `V.ReadTrace`(os.pread, os.open 감싸기) 대신 `PackedReadTrace`(os.pread, os.preadv 감싸기).
  따라서 TTFT 차이에는 packed 경로 효과가 섞여 있다.
- `torch.set_grad_enabled(False)`는 138처럼 전역으로 호출하지 않는다(요청 경로는 모두 `@torch.no_grad()`).

## 6. Gate (Phase 1 — 모두 PASS여야 main. 실패 gate 우회·임계값 변경·표본 교체 금지)

검증 표본은 v3 pilot40의 40장(최소/최대 N, 세로·가로 극단 포함)이며 품질 holdout이 아니다. 모든 gate 영수증은 현재 `experiment_freeze.json` SHA에 묶인다.

- **VG1** 데이터·채점 동일성(CPU): 3절 데이터·채점 항목.
- **VG2** 예산(CPU): 3절 벡터·k_l 항목 + 25,000장 각각에서 PYR-beta-raw packed 총 payload 바이트(Σ 2·k_l·8192 + separator 2·32·newline·8192)가
  v3 [B]의 같은 이미지 T2·T3 `ssd_read_bytes`와 같은지 대조. 다르면 원인을 기록하고, 설명되지 않으면 멈춘다.
- **VG3** 하한 전달(CPU): raw 벡터 fixture store(N = 960 / 2304 / 2880)가 모든 validator(prefix25, packed_store, ImageContext, PackedImageContext, 두 selector, 새 감사 경로의 `doc_counts`)를 통과하고,
  meta의 `r_min`만 0.10으로 바꾸면 모두 fail closed. 새 경로의 default 의존 호출 0건(정적 검사), 위치 기반 arm 가정 0건.
- **VG4** packed 경로(GPU): 166 `pyr_case`의 검사를 MT-VQA 고정 사례 5개에 적용. 사례는 pilot40에서 frozen geometry만으로 고른다:
  최소 N, 최대 N, 중간값 N(main 25,000장 N의 하위 중간값에 가장 가까운 것), 세로 극단, 가로 극단(동률은 image id 작은 것, 중복 없이 차례로).
  pilot40에서 가장 가로로 긴 이미지가 최소 N 이미지와 같으므로 가로 극단은 그다음으로 가로로 긴 이미지다.
  검사: packed slice == 같은 capture로 만든 unpacked layerwise 파일(bitwise), 파일 크기 = 기대 바이트, 복원 cache bit == 캡처 KV의 FP16 변환본에서 순서표 앞 k_l행,
  mask·첫 logits·생성 ID가 같은 shape의 독립 in-memory 참조와 동일, layer별 가시 행 수 == k_l(마지막 layer 포함), hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기,
  질문만 바꿔도 read range 불변, vision 호출 0·score 호출 0. 벡터는 PYR-beta-raw와 U 둘 다.
- **VG5** G-det-U(GPU, 필수): pilot40 40장에서 `ours_prefixstore_kv25_U_packed`를 3턴 실행(120요청, phase `gate_u_pilot40`)해 v3 pilot40의 `ours_prefixstore_kv25`[B] 기록과
  같은 요청(image, turn)의 입력 token ID(이미지 뒤 text 쪽 input id 전체와 prompt 문자열; 이미지 쪽은 image SHA로 묶임)·생성 token ID·예측 문자열·score·N·read bytes가 전부 일치해야 한다.
  100%가 아니면 불일치를 덤프하고 원인을 고친 뒤 재freeze한다.
- **VG6** 회귀: 기존 CPU 테스트 전체 + 새 테스트 통과(GLM 기존 실패는 분리 보고), 기존 소스 무수정(before inventory 대비 전체 SHA).
- 실패 시 원인·수정을 `freeze_amendment_*.json`으로 남기고 재freeze한다(실패 영수증은 `*.superseded_*`로 보존). 재freeze하면 모든 gate를 다시 실행한다.

## 7. Phase 2 — smoke → pilot40 → main

- smoke: v3 smoke 4장 × 3 turns × PYR arm = 12요청. 이미지별 독립 감사 PASS + T1 생성 token이 v3 smoke T1과 일치(`g_det_T1_smoke`).
- pilot40: 40장 × 3 turns = 120요청(PYR arm). 이미지당 wall time·보존 raw 크기·scratch store 최대 크기를 실측해 main 예상 소요 시간과 필요 디스크를 계산하고
  `llava/main_resource_admission.json`을 남긴다. T1 token이 v3 pilot40 T1과 일치해야 한다(`g_det_T1_pilot40`).
- main: 25,000 images × 3 turns = 75,000요청. T1 full pixel → capture → image-only 중요도 → packed store 생성(이미지당 1회) → T2·T3 packed hit.
  method-local generated history(gold 미사용), image-atomic attempt, 이미지별 독립 감사 후 run-local scratch store만 정확한 SHA allowlist로 삭제,
  raw journal은 lossless gzip으로 보존(원본·압축본 SHA를 commit에 기록). 요청마다 다른 GPU compute PID가 있으면 멈춘다(종료시키지 않는다).
  중단 시 `--resume`(freeze 불변일 때만; 실패한 attempt는 보존하고 그 이미지는 새 attempt로 처음부터 다시 실행). 요청 수 완주·실패 0·감사 PASS여야 final.
- **중간 보고 1회(사전 고정)**: commit된 main 이미지가 2,500장이 되면 그 2,500장(manifest 순서의 처음 2,500장; manifest 순서가 무작위 표본 순서)만으로 8절의 표를 계산해
  `results/…/INTERIM_2500.md`를 쓴다(제목에 INTERIM, "판정 아님" 명시). 분석은 runner가 **이미지 사이에서 동기적으로** 실행해 timed 요청과 겹치지 않게 한다.
  실행은 계속하며, 중간 결과를 보고 설정·arm·규칙을 바꾸지 않는다(완주가 목표).
- 2,500장마다 T1 TTFT 평균과 packed read 처리율(MB/s), 여유 디스크, SSD·GPU 온도를 `environment_log.jsonl`에 남긴다(환경 변동 추적, 판정 아님).

디스크 계획: 이미지당 scratch store(평균 약 310 MB, N = 2880에서 약 0.40 GB)를 이미지 단위로 만들고 감사·commit 후 삭제한다. 동시에 존재하는 store는 1개다.
raw gzip과 영수증은 보존한다. 여유 30 GiB 미만이 되면 BLOCKED_STORAGE로 멈춘다(`capacity_guard`: 30 GiB + 작업분 2 GiB; main admission: pilot40 실측 raw × 25,000 × 1.35 추가).
scratch는 v3 run과 같은 디스크·같은 파티션(`/dev/nvme0n1p2`, SK hynix SHPP41-2000GM)에 둔다. 기존 store·run·results는 수정·삭제하지 않는다.

## 8. 판정과 보고 (모두 실행 전에 고정. 결과를 본 뒤 바꾸지 않는다)

공통: Primary는 followup(T2–T3, 50,000요청)의 공식 VQA score(%). 비교는 v3 run의 같은 요청(image, turn)과 짝지은 이미지 단위 paired bootstrap 95% CI
(10,000회, seed 1234, 142의 `bootstrap` 함수 그대로). 모든 cross-run 비교에 "cross-run, 결정성 확인됨"을 표기한다. CI가 0을 포함하면 "차이 미입증"으로 쓴다(동등성 아님).
분석 코드의 자기 검사: v3 index만으로 v3 `paired_analysis.csv`의 [B] − baseline 행을 1e-6 이내로 재현해야 한다(`analysis_selfcheck.json`; freeze 전에 한 번, final에서 다시).

정합성(보고 전 필수) **G-det-T1**: PYR arm의 T1 생성 token ID(및 prompt·text 쪽 input id SHA)가 v3 main의 같은 요청 T1(v3는 모든 arm이 T1 일치)과 25,000/25,000 일치해야 한다.
100%가 아니면 불일치를 덤프하고 PENDING_REVIEW로 두며, cross-run 정확도 비교를 NOT_COMPARABLE로 표시한다.

- **(A) 주 판정 — PyramidKV 배분 효과**: D = PYR-beta-raw_packed − [B](v3 `ours_prefixstore_kv25`), followup VQA score.
  IMPROVED: CI 하한 > 0 · NOT_SHOWN: CI가 0 포함 · WORSE: CI 상한 < 0.
  참고값(판정 아님): D ≥ +1.0 pp인지(MT-GQA GO 기준), 손실 회복률 = D / (ReComp − [B]) (%). 조건 확인: SSD MB/followup이 [B] × 1.02 이하인지.
  MT-GQA의 "+1.0 pp" 규칙을 쓰지 않는 이유: MT-VQA에서는 [B]와 ReComp의 차이가 1.66 pp뿐이라 +1.0 pp는 손실의 60%를 메워야 통과하는 기준이 된다.
  이번 질문은 "MT-VQA에서도 좋아지는가"이므로 CI가 0을 제외하는지로 판정한다.
- **(B) 원래 성능 대비**: PYR − ReComp(v3)의 Δ·CI, ReComp 대비 %. PYR − FullLoad, PYR − SparseVLM-SSD-KV25-AllHead도 같은 방식.
- **(C)** turn별(T1/T2/T3/All/Followup) score, answer_type별(yes/no, number, other) followup Δ(탐색적, 다중비교 보정 없음), PYR과 [B]의 예측 일치율과 한쪽만 정답 건수
  ("정답" = 그 요청의 공식 VQA score > 0; score가 더 높은 쪽의 건수도 함께 적는다).
- **(D) 지연·I/O**: followup TTFT·E2E mean/p50/p95, read·scatter·suffix prefill ms, SSD MB/followup, pread/preadv 횟수, T1 TTFT.
  같은 실행 안 비율 R_in = followup TTFT 평균 / 같은 실행 T1 TTFT 평균(T1은 full pixel 재계산). 참고: v3에서 ReComp followup / T1 = 530.15 / 521.56 = 1.0165, [B] followup / T1 = 262.46 / 521.48 = 0.503.
  v3 ReComp 530.15 ms, FullLoad 755.79 ms, AllHead 844.59 ms, [B] 262.46 ms는 나란히 적되 "cross-run 참고치, 절대 TTFT는 실행 간 비교하지 않는다"를 명시한다.
  환경 변동 지표: 이번 T1 TTFT 평균 vs v3 T1(약 521.5 ms)의 비율, packed read 처리율(MB/s)과 MT-GQA PYR run의 값(308.05 MB / 97.5 ms).
- **(E) 저장·session**: store MB/이미지, persistence·activation mean/p50/p95, 3-turn session = 3 E2E + persistence + activation(이미지당 dialogue 1개라 분할 상각 없음).
  v3 ReComp 1679.79 ms, [B] 1954.00 ms와 나란히(cross-run 참고; v3 값은 `standalone_equivalent_session_ms`).
- **(F)** MT-GQA와의 일관성 한 줄: MT-GQA PYR − U = +1.244 pp와 MT-VQA의 D의 부호·크기를 나란히 적는다(지표가 EM과 VQA score로 다름을 명시).

판정·기준값은 frozen 스크립트(184)가 `summary.json`과 `decision.json`에 기계적으로 기록한다. 결과 묶음과 REPORT 필수 문구는 프롬프트 7절 목록을 따른다.
완료 보고에는 실제 요청 수, 실패·스킵, 소요 시간, 프롬프트와 다르게 한 점, NOT_MEASURED 항목을 적는다.

## 9. 소스·산출물 보호

- before inventory: 새 파일을 하나도 만들기 전(2026-11-14T18:23:31Z)의 `mmimpress/**/*.py`, `scripts/*.py|*.sh`, `tests/*.py`, `docs/*.md`, `papers/*.md`, vendor 파일 363개 전체 SHA256(`protection/source_before.json`).
  프롬프트 파일 끝의 "결과 기록" 칸은 실행 후 채우도록 되어 있으므로 그 파일 하나만 변경 검사에서 뺀다.
- 기존 `runs/`·`results/`(이 run 제외) 전체 파일의 크기·mtime_ns inventory(`protection/artifacts_before*.json`; 큰 파일의 전체 SHA는 아님)를 실행 전후로 비교한다.
- 새 산출물은 `runs/llava_next_mtvqa_pyr_20261114T182331Z/`와 `results/llava_next_mtvqa_pyr_20261114T182331Z/`에만 쓴다(단위 테스트의 임시 fixture `runs/llava_next_mtvqa_pyr_unittest_fixture`는 테스트가 만들고 지운다).
- reset/clean, upgrade/commit/push 없음.
