# 재현: 소규모 CPU 감사

기준 날짜2026-10-13 KST. 작업 위치 `/home/dblab/hj/mllm_v2`. GPU 접근 불가로 static/CPU 검증만 실행했다. 기존 main runner, 실험결과, 기존 cache를 수정하거나 재실행하지 않는다.

## 사용 환경

Python `/home/dblab/anaconda3/envs/mllm_ft/bin/python` (3.10.20), torch2.5.1+cu121, Transformers4.57.6, NumPy2.2.6. CUDA device count0. 의존성 설치 불필요. pytest는 이 환경에 없어 unittest를 사용한다.

로컬 tokenizer snapshot:

`/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23`

CPU probe가 읽는 것은 tokenizer뿐이다. pretrained7B weights를 읽거나 vision encoder를 실행하지 않는다. random Llama3 layers/H4/KV4/D16/hidden64, synthetic visual96, seed1234, FP32 decoder + production FP16 SSD store, diagnostic FP32 store. greedy3 tokens는 CPU history 진단용이다. main16-token GPU 측정과 혼동하지 않는다. tolerance는 실행 전에 고정된 rtol1e-5/atol1e-6.

## 새 디렉터리에서 실행

아래는 기존 감사파일을 덮어쓰지 않는다. 약 수초, 새 파일 수 MB 규모. shell 변수는 감사 전용이다.

```bash
cd /home/dblab/hj/mllm_v2
AUDIT_PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
AUDIT_REPLAY_ROOT=$(mktemp -d /tmp/mpic-audit-replay.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

"$AUDIT_PY" audits/mpic_20261013_cpu_01/scripts/cpu_probe.py \
  --out "$AUDIT_REPLAY_ROOT/cpu" > "$AUDIT_REPLAY_ROOT/cpu.log" 2>&1
cat "$AUDIT_REPLAY_ROOT/cpu.log"

"$AUDIT_PY" -m unittest discover -s tests -p test_mpic.py -v \
  > "$AUDIT_REPLAY_ROOT/unittest_mpic.log" 2>&1
"$AUDIT_PY" -m unittest discover -s tests -p test_llava_mt_gqa_B.py -v \
  > "$AUDIT_REPLAY_ROOT/unittest_history.log" 2>&1

"$AUDIT_PY" audits/mpic_20261013_cpu_01/scripts/collect_static.py \
  --out "$AUDIT_REPLAY_ROOT/evidence" > "$AUDIT_REPLAY_ROOT/static.log" 2>&1
"$AUDIT_PY" audits/mpic_20261013_cpu_01/scripts/check_historical.py \
  --evidence "$AUDIT_REPLAY_ROOT/evidence" > "$AUDIT_REPLAY_ROOT/historical.log" 2>&1
```

`cpu_probe.py`는 output path가 이미 존재하면 실패한다. source/main artifact는 read-only로 취급한다. fault injection은 새 `fault_store`에만 적용한다. wrapper/torch.cuda CPU shim은 해당 프로세스 내에서만 적용되고 context 종료 시 복구된다. 실제 GPU tensor를 CPU로 위장한 test는 없다.

## 기대되는 출력

probe 완료 summary:

```json
{"mechanism_cases_passed":15,"generated_history_requests":9,"multi_image_rejections":2,"legacy_fullload_single_suffix":0,"GPU":"NOT VERIFIED"}
```

FullLoad suffix1 defect 및 multi-image 미지원은 **의도적으로 발견해 기록하는 FAIL**이다. probe의 process exit0은 모든 baseline 요구가 PASS라는 뜻이 아니다. `summary.json` 및 보고서의 항목별 판정을 읽어야 한다.

기존 tests: MPIC17개, main history safety8개 통과. independent historical sample9개 확인; 이것은 저장된 source/input/timestamp/I/O arithmetic 검증이며 fresh GPU 재현이 아니다.

최종 실행 근거는 `evidence/cpu_probe_v4/`와 `evidence/cpu_probe_v4.log`다. 앞선 시도의 log/fixture도 보존했다. v1: 같은 길이 prefix fixture assertion 실패. v2: FullLoad 진단 metadata n_spatial 누락. v3: mechanism probe 완료. v4: FullLoad actual zero KV와 logits 오류, changed-prefix silent-ignore control, explicit generation attention mask를 보강한 최종 실행. production source와 수치 tolerance는 변경하지 않았다.

## 안전한 해석과 후속 GPU 재검증 범위

신규 실행이 검증하는 것은 selected-row 계산과 full linked K/V attention의 수학적 동작, production persistence/reader, causal position, boundary k, generated-history lineage다. 실제 image 의미/AnyRes pixel geometry/NF4 numerical behavior/VRAM/GPU TTFT는 NOT VERIFIED다.

GPU가 확보되면 pretrained snapshot·NF4/BF16/eager를 고정하고 약3개 image의 단일 question·3-turn method-local history·변경 pre-image text에 대해 actual layer Q/K/V·position·mask·I/O·first token을 기록한다. kN은 동일 persisted visual inputs의 dense reference와 비교하고, k32 partial reuse를 ReComp와 무조건 같다고 요구하지 않는다. tolerance를 먼저 선언하며 실패 뒤 완화하지 않는다. 다중 이미지/mixed hit-miss는 현재 서버가 지원하지 않으므로 구현 및 k 정책을 확정한 후 요청3/5를 검증해야 한다.

원 논문 재조회는 [v2 manuscript](https://arxiv.org/html/2502.01960v2)와 [version history](https://arxiv.org/abs/2502.01960v2) 기준이다. 논문/저자 사이트에서 공식 MPIC source를 확인하지 못한 한계는 재현 로그에 유지한다.
