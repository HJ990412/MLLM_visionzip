# Layer-wise Budget — PyramidKV(β=20) MT-GQA HOLD 추가 실행 계약 (PYR-beta, PYR-beta-raw)

2026-11-10 UTC, 어떤 측정보다 먼저 freeze. 저장소 `/home/dblab/hj/mllm_v2` (사용 가능한 `.git` 없음; 소스 보호는 SHA256 manifest).
Run root `runs/layer_budget_kv25_pyrbeta_20261110T073604Z`, 결과 `results/layer_budget_kv25_pyrbeta_20261110T073604Z`.
부모 run: v2 `runs/layer_budget_kv25_v2_20261106T073502Z` / `results/layer_budget_kv25_v2_20261106T073502Z` (판정 GO, 읽기 전용 입력).
실행 프롬프트: `papers/layer_budget_kv25_pyrbeta_hold_prompt.md`. 이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다.

## 1. 목적

PyramidKV 공개 코드 식(β = 20, window 0; `Zefan-Cai/PyramidKV 68cd9551 pyramidkv/pyramidkv_utils.py L360-370`)의 layer 배분을
v2와 똑같은 packed PrefixStore 경로·HOLD 조건으로 MT-GQA에서 측정한다. 두 버전을 함께 측정한다.

- **PYR-beta**: v2 run에서 freeze한 벡터(원식 + 우리 계약 하한 0.10). ratio SHA256
  `e374616b5a028402f8c479159f458f83690c96c24728307e00113f206a4cd483`. 새로 만들지 않고 파일을 복사한다(파일 SHA `73457efbbcab…`).
- **PYR-beta-raw**: 같은 v2 json의 `source.raw_unclipped` 32개 값을 **그대로** 쓰는 신규 벡터(하한 없음, 문서 `r_min = 0.0`, `r_max = 0.60`).
  ratio SHA256 `51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724`, 파일 SHA256 `8bfc4cbc55eb778a7ef0433ade4bf13f7a24107b76515f23a72ac01e0740ea5d`.
- 같은 실행 안의 대조군으로 **U_packed**(ratio SHA256 `54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba`, v2 파일 복사)를 다시 측정한다.

데이터로 벡터를 만들거나 고르는 과정(CAL/DEV)은 없다. 따라서 holdout 겹침 문제가 없다. 기존 serving·store·packed reader 코드 경로를 바꾸지 않는다.
논문에서는 layer-wise 배분을 우리 기여가 아니라 **기존 방법(PyramidKV)을 적용한 부품**으로 쓰며, 이 실행은 그 원식 벡터의 HOLD 수치를 확보하기 위한 것이다.

알아 둘 사실:
- v2 HOLD의 `PYR`은 PyramidKV 원식이 아니라 v1 F-front(0.40 → 0.10 직선)를 이름만 바꾼 벡터이며 두 PYR-beta와 다르다.
- 원 PyramidKV는 layer별 예산과 함께 attention 기반 token 선택을 쓴다. 여기서는 image-only VisionZip 순서를 그대로 두고 **layer별 자르는 지점만** PyramidKV를 따른다.
- 모든 arm은 같은 바이트(Σ k_l = 32·ceil(0.25N)), 같은 packed I/O 경로(hit당 `preadv` 1회), 같은 token 순서를 쓰고 layer별 자르는 지점만 다르다.

## 2. Arm (이 순서로 freeze)

1. `ours_prefixstore_kv25_U_packed`
2. `ours_prefixstore_kv25_PYR-beta_packed`
3. `ours_prefixstore_kv25_PYR-beta-raw_packed`

선택 arm `recompute`는 "Phase 0 저장 공간·시간 계획이 허용하고 사용자가 실행 전에 허용한 경우에만" 포함한다. 실행 전에 사용자 허용이 없었으므로 **제외**한다(기본값).
따라서 ReComp 대비 같은 실행 안 TTFT 비교는 이 실행에서 NOT_MEASURED이다.

## 3. 데이터와 실행 조건 (v2 HOLD manifest 그대로)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm → 3 arm = 36,549 요청.
  INDEX_SHA `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`, WORKLOAD_SHA `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`.
  dialogues·image identity·smoke dialogue id(`mtgqa_000001`, `mtgqa_000012`, `mtgqa_000034`, `mtgqa_000038`; 4 images)는 v2 HOLD manifest와 동일함을 freeze에서 assert한다.
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, seed 1234, batch 1, CPU threads 2.
- v2 162와 동일: 각 arm 자기 full-pixel T1, method-local generated history, strict normalized EM, image-atomic attempt, 이미지별 독립 감사 후
  run-local scratch store만 삭제(정확한 SHA allowlist), DONTNEED는 timer 밖, true TTFT, 중단 시 `--resume`. dialog별 arm 순서 회전은 162의 `method_order` 규칙을 arm 수(3)에 그대로 적용한다.
- 코드: 기존 스크립트(162 포함)와 `mmimpress/*`는 수정하지 않는다. 새 스크립트만 추가한다.
  - `scripts/163_make_pyrbeta_raw_vector.py` (Phase 0: 벡터 복사·raw 생성·검산·k_l 표, freeze·저장 계획·protection before)
  - `scripts/164_eval_layer_budget_pyrbeta_hold.py` (162의 얇은 복사본. 차이: arm 목록을 `hold_arms.json`에서 받음, stored arm을 위치(`methods[1:]`)가 아니라 종류로 찾음,
    감사의 k_l 재계산이 문서의 `r_min`/`r_max`를 사용 + 독립 Hamilton 검산 + store meta의 clip range 일치 검사 추가, 벡터 경로 `<run>/vectors`, schema id. 그 외 요청 생성·회전·history·채점·감사·삭제 규칙은 162와 동일)
  - `scripts/165_report_layer_budget_pyrbeta.py` (감사·판정·보고)
  - `scripts/166_validate_layer_budget_pyrbeta.py` (gate: G3, G-rmin, CPU, GPU G6/G7, G-det)
- `mmimpress/layer_budget.py`의 전역 `R_MIN = 0.10`, `R_MAX = 0.60`은 바꾸지 않는다. raw 벡터는 문서의 `r_min = 0.0`으로만 표현한다.

## 4. 하한 0.10과 PYR-beta-raw

- v1부터 모든 벡터에 적용한 layer별 하한 0.10(상한 0.60)은 v1 설계 문서에서 **안전장치**로 넣은 값이며, 실험이나 선행 연구로 정한 근거가 없다.
- PyramidKV 원식(β = 20)은 min = 0.25/β = 0.0125, max = 0.5 − min = 0.4875, 32 layer 등간격 직선이고 합이 정확히 8.0이다.
  하한 0.10을 적용하면 layer 26–31(원식 0.0891 → 0.0125)이 모두 0.10으로 올라가고 그만큼 layer 0–25에서 비례해 덜어낸다.
- 독립 검산(`vectors/pyrbeta_raw_verification.json`): 유리수로 따로 계산한 원식과 `raw_unclipped`의 layer별 최대 차 = 5.551e-17 (< 1e-12),
  |합 − 8.0| = 8.882e-16 (< 1e-9). 값은 다시 계산하지 않고 v2 json의 리스트를 그대로 썼다.
- HOLD 398장(N 1344 – 2880, 산술 중간값 2256.0; 표는 실제 이미지 N인 하위 중간값 **N = 2208** 기준)에서
  `layer_keep_counts(..., r_min=0.0, r_max=0.60)`: PYR-beta-raw의 k_l = 0인 layer **0개**, 최소 k_l = **17행**(layer 31, N = 1344).
  (k_l = 0인 layer가 하나라도 있었다면 BLOCKED로 멈췄을 것이다.)

| layer | 원식 r_l (PYR-beta-raw) | 하한 0.10 적용 r_l (PYR-beta) | 차이 (하한 − 원식) | k_l raw (N=2208) | k_l PYR-beta (N=2208) | k_l U (N=2208) | HOLD 최소 k_l raw | HOLD 최소 k_l PYR-beta |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.487500 | 0.468801 | -0.018699 | 1076 | 1035 | 552 | 655 | 630 |
| 1 | 0.472177 | 0.454066 | -0.018111 | 1043 | 1003 | 552 | 635 | 610 |
| 2 | 0.456855 | 0.439331 | -0.017523 | 1009 | 970 | 552 | 614 | 591 |
| 3 | 0.441532 | 0.424597 | -0.016936 | 975 | 937 | 552 | 593 | 571 |
| 4 | 0.426210 | 0.409862 | -0.016348 | 941 | 905 | 552 | 573 | 551 |
| 5 | 0.410887 | 0.395127 | -0.015760 | 907 | 872 | 552 | 552 | 531 |
| 6 | 0.395565 | 0.380392 | -0.015173 | 873 | 840 | 552 | 532 | 511 |
| 7 | 0.380242 | 0.365657 | -0.014585 | 840 | 807 | 552 | 511 | 492 |
| 8 | 0.364919 | 0.350922 | -0.013997 | 806 | 775 | 552 | 490 | 472 |
| 9 | 0.349597 | 0.336187 | -0.013409 | 772 | 742 | 552 | 470 | 452 |
| 10 | 0.334274 | 0.321453 | -0.012822 | 738 | 710 | 552 | 449 | 432 |
| 11 | 0.318952 | 0.306718 | -0.012234 | 704 | 677 | 552 | 429 | 412 |
| 12 | 0.303629 | 0.291983 | -0.011646 | 670 | 645 | 552 | 408 | 393 |
| 13 | 0.288306 | 0.277248 | -0.011058 | 637 | 612 | 552 | 387 | 373 |
| 14 | 0.272984 | 0.262513 | -0.010471 | 603 | 580 | 552 | 367 | 353 |
| 15 | 0.257661 | 0.247778 | -0.009883 | 569 | 547 | 552 | 346 | 333 |
| 16 | 0.242339 | 0.233043 | -0.009295 | 535 | 515 | 552 | 326 | 313 |
| 17 | 0.227016 | 0.218309 | -0.008708 | 501 | 482 | 552 | 305 | 293 |
| 18 | 0.211694 | 0.203574 | -0.008120 | 467 | 449 | 552 | 285 | 274 |
| 19 | 0.196371 | 0.188839 | -0.007532 | 434 | 417 | 552 | 264 | 254 |
| 20 | 0.181048 | 0.174104 | -0.006944 | 400 | 384 | 552 | 243 | 234 |
| 21 | 0.165726 | 0.159369 | -0.006357 | 366 | 352 | 552 | 223 | 214 |
| 22 | 0.150403 | 0.144634 | -0.005769 | 332 | 319 | 552 | 202 | 194 |
| 23 | 0.135081 | 0.129899 | -0.005181 | 298 | 287 | 552 | 182 | 175 |
| 24 | 0.119758 | 0.115165 | -0.004594 | 264 | 254 | 552 | 161 | 155 |
| 25 | 0.104435 | 0.100430 | -0.004006 | 231 | 222 | 552 | 140 | 135 |
| 26 | 0.089113 | 0.100000 | +0.010887 | 197 | 221 | 552 | 120 | 134 |
| 27 | 0.073790 | 0.100000 | +0.026210 | 163 | 221 | 552 | 99 | 134 |
| 28 | 0.058468 | 0.100000 | +0.041532 | 129 | 221 | 552 | 79 | 134 |
| 29 | 0.043145 | 0.100000 | +0.056855 | 95 | 221 | 552 | 58 | 134 |
| 30 | 0.027823 | 0.100000 | +0.072177 | 61 | 221 | 552 | 37 | 134 |
| 31 | 0.012500 | 0.100000 | +0.087500 | 28 | 221 | 552 | 17 | 134 |

raw arm을 넣은 이유: "PyramidKV 적용"이라고 쓰려면 원식 그대로의 수치가 필요하고, 근거 없는 하한이 결과에 영향을 주는지 같은 실행 안에서 직접 확인하기 위해서다.

## 5. Gate (모두 PASS여야 HOLD; 우회·임계값 변경 금지)

- **G3 예산 불변 (CPU)**: U·PYR-beta·PYR-beta-raw에 대해 HOLD 398장의 N 목록에서 Σ k_l = 32·ceil(0.25N) 정확히 일치(문서의 clip range 사용, 독립 Hamilton과 일치).
  U·PYR-beta는 v2 `gate_g3_budget_invariance.json`(PASS, 같은 ratio SHA)과 대조하고, U는 v2 HOLD의 실제 `packed_U_meta.json` 398개의 N·`layer_keep_counts`와 대조한다.
- **G-rmin 하한 전달 (신규, CPU)**: PYR-beta-raw 문서로 fixture store(32 layer, N = 1344 / 2208 / 2880)를 만들고 store meta → packed writer·unpacked layerwise writer →
  `prefix25.validate_llava_prefix_layerwise_meta`, `packed_store.validate_packed_meta`, `ImageContext`(+`validate_visual_kv_layout`), `PackedImageContext`(+`validate_packed_layout`),
  두 selector(`CVPR25ChunkSelector` layerwise, `PackedPrefixSelector`), 164의 감사 경로(`doc_counts`)가 모두 문서의 r_min = 0.0을 쓰고 통과하는지, layer별 가시 행 수가 독립 계산 k_l과 같은지 확인한다.
  같은 store의 meta에서 r_min만 0.10으로 바꾸면(integrity 재봉인) 모든 경로가 fail closed되는지 확인한다. 추가로 store/serving 모듈과 새 스크립트에서 r_min을 기본값에 맡기는 호출을 AST로 찾는다.
  기본값에 기대는 경로가 발견되면 그 호출부에서 문서의 r_min을 넘기도록 최소 수정하고 `freeze_amendment_*.json`으로 기록한다(상수 변경·검사 완화 금지).
- **G6/G7 packed 경로 (GPU)**: v2 161 `--stage gpu`와 같은 5개 고정 사례(`correctness_manifest.json` index 0, 3, 4, 5, 6), PYR-beta와 PYR-beta-raw 각각:
  packed slice == 같은 벡터의 unpacked layerwise 파일 비트 동일, 총 바이트 동일(두 벡터·uniform 공식과 동일), 복원 cache bit·mask·첫 토큰·생성열 동일(unpacked 서빙 및 독립 reference 대비, logits 1e-4),
  hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기, 질문만 바꿨을 때 read range 불변, vision/score 호출 0. raw는 layer별 가시 행 수가 독립 계산 k_l과 정확히 같은지(마지막 layer 포함) 확인.
  gate store는 사례별로 만든 뒤 지운다(v2 amendment 규칙).
- **G-det 결정성 (필수)**: smoke에서 U_packed의 생성 토큰이 v2 HOLD의 같은 (dialog, turn) 요청과 토큰 단위로 일치(12/12)해야 main으로 간다. main 완료 후 U_packed 12,183 요청 전체의 일치율을 계산한다.
  100%가 아니면 cross-run 정확도 비교를 하지 않고 NOT_COMPARABLE로 표시한다(주 판정은 같은 실행 안 비교라 그대로 진행).
- **기존 CPU 테스트 전체 통과** (GLM 모듈은 v1/v2와 같이 별도 실행·기록).
- 실패 시 원인·수정을 `freeze_amendment_*.json`으로 남기고 재freeze한다.

## 6. 판정 (사전 고정)

**주 판정** — 같은 실행 안, 두 벡터 각각: Δ = HOLD followup 정확도(벡터 − U_packed), 이미지 단위 paired bootstrap 95% CI(10,000회, seed 1234).
- GO: Δ ≥ +1.0 pp & CI 0 제외 & SSD MB/followup ≤ U × 1.02 & TTFT 평균 ≤ U × 1.05
- PARTIAL: Δ ≥ +0.5 pp이고 CI 0 포함, 또는 GO에서 MB/TTFT 중 정확히 하나 위반
- NO-GO: 그 외
(v2 A1-2 판정과 같은 규칙. +0.5~1.0 pp이고 CI가 0을 제외하는 경우는 규칙상 NO-GO이며, 그 경우 REPORT에 "규칙상 NO-GO지만 CI는 0 제외"를 숫자 그대로 적는다.)

**하한 효과 판정** — 같은 실행 안: D = PYR-beta-raw − PYR-beta, followup 정확도 paired bootstrap 95% CI.
- CI가 0 포함 → "하한 불필요": 논문 기본안은 PYR-beta-raw(원식 그대로)
- CI 상한 < 0 → "하한 필요": PYR-beta를 쓰고 하한의 근거로 보고
- CI 하한 > 0 → "하한이 손해": PYR-beta-raw를 쓰고, A1-2·VLC 등 하한에 걸린 기존 벡터도 영향을 받았을 수 있음을 명시
결과 문장은 `decision.json`의 `floor_effect` 필드에 기계적으로 기록한다.

**부 판정 (보고만)**
- G-det가 100%일 때만: v2 HOLD 예측과 요청 단위로 짝지은 정확도 비교 — 두 PYR-beta 각각 vs PYR(선형), A1-2, VLC_offline, ReComp, FullLoad. paired bootstrap CI, "cross-run, 결정성 확인됨" 표기.
  TTFT는 cross-run 비교하지 않는다.
- 질문 유형별 Δ(v2 REPORT와 같은 regex 분류, spatial 강조), layer별 k_l·바이트 표(두 벡터 나란히), persistence·session(shared / standalone DERIVED).
- v2 U_packed와 이번 U_packed의 TTFT 평균 차이(실행 간 변동 참고치, cross-run 표기).

## 7. 저장 계획과 삭제 범위

- 디스크 여유 실측(계약 작성 시점): **266.40 GiB** (`shutil.disk_usage`, 286,044,323,840 bytes).
- v2 162 freeze의 FRESH_IMAGE_STREAMING 방식: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch를 정확한 SHA allowlist로 삭제.
  packed store 상한 = 32 layer × ceil(2880/4) 행 × 2 × 8,192 B + separator 48행 + 보조 8 MiB = 411,041,792 B (0.41 GB).
  3 stored arm → 이미지당 묶음 ≤ 1.23 GB, builder margin 4 GiB, raw·evidence 증가 상한 4 GiB(v2 7 arm main 2.1 GB 기준), 최소 reserve 30 GiB
  → HOLD 필요 여유 ≈ 39.1 GiB. GPU gate는 사례당 4 store(≤ 1.6 GB) + margin, 106의 capacity reserve 46 GiB 필요 → 초기 필요 여유 ≈ 51.5 GiB. **PASS**(정확한 수치는 `phase0/storage_plan.json`).
  부족하면 BLOCKED_STORAGE로 멈추고 사용자에게 보고한다.
- 삭제 범위: `runs/<RUN>/hold/scratch/**`(감사+commit 후 정확한 allowlist), `runs/<RUN>/llava/gpu_gates/**`(사례별), `runs/<RUN>/gates/grmin_fixture/**`(CPU fixture). 기존 runs/results/stores는 어떤 경우에도 지우거나 고치지 않는다.

## 8. 소스·산출물 보호

- before(Phase 0 freeze) / after(보고) 비교: `results/`, `docs/`, `mmimpress/`, `scripts/`, `tests/`, `README.md` 전체 파일의 SHA256(이 run의 새 경로 제외),
  `kvstore*` 전체와 **다른 모든 `runs/*`**의 (size, mtime_ns, inode) stat inventory. 변경·누락이 있으면 FAIL(기록된 freeze amendment로 설명되는 소스 변경만 예외).
- 기존 store/serving 소스가 v2 HOLD freeze(`hold/config.json`의 `source_sha256`)와 같은지 freeze에서 확인한다(v2 보고 후 수정이 기록된 `scripts/160`만 예외).
- reset/clean, 무단 upgrade/commit/push 금지. 새 산출물은 `runs/layer_budget_kv25_pyrbeta_20261110T073604Z/`, `results/layer_budget_kv25_pyrbeta_20261110T073604Z/`에만 쓴다.

## 9. 산출물

`results/<RUN>/`: REPORT.md, CONTRACT.md, decision.json(`primary`, `floor_effect`), 벡터 json 3개 + SHA, raw 벡터 검산 결과, k_l 표,
`hold/{quality,latency_io,sessions,storage,paired_analysis,question_types,layer_table}.csv/json`, cross_run_v2_comparison.json, g_det_equivalence.json,
gate 결과(G-rmin 포함), independent_audit.json, REPRODUCE.md, protection before/after, deletion_log.json, fig_vectors_pyr.png.
완료 보고에는 실제 요청 수, 실패·스킵, 소요 시간, NOT_MEASURED 항목을 명시한다.
