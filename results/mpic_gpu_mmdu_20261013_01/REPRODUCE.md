# 재현 명령과 환경

이번 결과: `results/mpic_gpu_mmdu_20261013_01`. 입력 manifest를 바꾸지 않는다. 아래 경로들은 **새로 존재하지 않는 output**에만 쓴다. scripts는 기존 output이 있으면 중단하므로 기존 실험을 덮어쓰지 않는다. 최대5대화/첫3턴/cap256의 bounded pilot만 실행한다.

## 실행 환경

cwd `/home/dblab/hj/mllm_v2`, 기존 `/home/dblab/anaconda3/envs/mllm_ft/bin/python` 사용. Python3.10.20/torch2.5.1+cu121/Transformers4.57.6/bitsandbytes0.49.2/NumPy2.2.6. RTX4090/driver535.230.02. generator NF4 double quant/BF16/eager, SSD FP16. source hash는 [source_sha256.json](source_sha256.json), frozen manifest는 [subset_manifest.json](subset_manifest.json).

**GPU가 노출된 host 환경에서 실행해야 한다.** 이번 Codex sandbox에서는 NVIDIA device가 없어 tensor 생성이 실패했지만 `exec_command(sandbox_permissions='require_escalated')`는 자동 검토를 통과했고 host GPU를 사용했다. 외부 host shell은 아래 명령을 그대로 사용할 수 있다. sandbox CPU 실패를 host driver 오류로 단정하지 않는다. 패키지/driver 재설치가 필요했던 작업은 없다.

```bash
cd /home/dblab/hj/mllm_v2
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PILOT_PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
PILOT_MANIFEST=results/mpic_gpu_mmdu_20261013_01/subset_manifest.json
# 자신이 선택한 새 output parent. 이전 결과 경로와 다르게 한다.
PILOT_OUT=results/mpic_gpu_mmdu_reproduction_new
```

아래의 기능 검증과 generator/judge GPU 명령은 한 번에 하나씩 실행한다. 다른 compute 작업을 종료하지 않는다. GPU 메모리가 부족하면 실패를 보존하고 설정을 조용히 바꾸지 않는다. 원래125 generator는 다른 compute PID가 있으면 중단한다.

```bash
nvidia-smi
"$PILOT_PY" - <<'PY'
import torch, transformers, bitsandbytes
print(torch.__version__, torch.version.cuda, transformers.__version__, bitsandbytes.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
x = torch.arange(16, device='cuda:0')
y = (x * x).sum()
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info(), y.item())
assert y.item() == 1240
PY
```

pretrained local snapshots:

- Generator: `/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23`.
- Judge: `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 기존 `mmimpress.qwen25.runner`의 pinned local snapshot. exact processor/tokenizer revision은 saved runtime과 일치한다.
- Dataset: manifest의 benchmark/image absolute paths와 SHA가 필요하다. generator가 benchmark/question/image hash를 다시 확인한다. training45k는 사용하지 않는다.

## CPU regression

```bash
"$PILOT_PY" -m unittest discover -s tests -p test_mpic_multi.py -v
```

이번9tests PASS log는 `evidence/regression_tests.log`. 이전34tests 결과는 보존된 prior report에 있다. 실제 GPU 기능 검증을 CPU로 대체하지 않는다.

## 최초 단일 대화 GPU 기능 gate

```bash
"$PILOT_PY" scripts/125_eval_mpic_mmdu_smoke.py \
  --manifest "$PILOT_MANIFEST" --execute --gate-only \
  --condition ssd_cold --out "$PILOT_OUT/gpu_gate"
```

7fixed fixtures×k0/global32/per-image32/all-selected, legacy FullLoad suffix1/2, wrong prefix/cache ID rejection, mixed serial/parallel/profiler를 실제 pretrained 모델에서 실행한다. all-selected 비교는 같은 persisted visual input의 native dense logits/K,V에 `rtol=atol=1e-4`. partial32의 dense equality는 요구하지 않는다. primary 실행 디렉터리는 `gpu_gate_attempt1`이고0측정 요청이다.

## MMDU 네 방법, 각5대화·15턴

```bash
"$PILOT_PY" scripts/125_eval_mpic_mmdu_smoke.py \
  --manifest "$PILOT_MANIFEST" --execute --condition ssd_cold \
  --out "$PILOT_OUT/mmdu_ssd"

"$PILOT_PY" scripts/125_eval_mpic_mmdu_smoke.py \
  --manifest "$PILOT_MANIFEST" --execute --condition gpu_resident \
  --out "$PILOT_OUT/mmdu_gpu_resident"
```

각 실행60 method requests, methods `prefix_caching, full_reuse, mpic_global32, mpic_per_image32`. gold history나 fallback이 없다. 처음 대화에서 gate를 다시 실행한 뒤 각 대화별 별도 warm-up을 수행한다. 기존 실제 디렉터리는 `mmdu_ssd_attempt1`, `mmdu_gpu_resident`다.

`answers.jsonl`은 prompt/input IDs/image IDs/history/prediction/generated IDs/first-token timestamp/TTFT/io/모든 MPIC layer shape 및 checks/peak allocated,reserved/output count/EOS,cap/retry,fallback를 저장한다. `ttft_io.csv`와 별도 config/environment/status도 생성한다. GPU로 준비한 source/input cache reads는 `resident_setup.json`, normal source forward/persist는 `image_*_persistence.json`. 캐시 생성 비용은 TTFT 밖이다.

TTFT start는 page conditioning·CUDA sync 후 prompt 생성/토큰화 이전, finish는 첫 argmax host materialization + CUDA sync 이후다. buffered pread bytes는 KV/input sidecar에만 해당하며 pixel I/O/NAND traffic을 합친 값이 아니다. DONTNEED는 best-effort로 eviction/physical SSD cold를 증명하지 않는다. diagnostic assertions가 timer에 들어간다. inclusive prefill+SSD intervals를 더하지 않는다.

## Legacy FullLoad GPU request isolation

```bash
"$PILOT_PY" scripts/130_validate_fullload_gpu_isolation.py \
  --source-dialog "$PILOT_OUT/gpu_gate/dialog_31" \
  --out "$PILOT_OUT/fullload_gpu_isolation"
```

독립 native image KV A/B/A/B와 suffix1/2를 검증하며 `validation.json`에 source/prefix 계약과 actual native attention cache observations를 저장한다. primary run은 `fullload_gpu_isolation`. prefix 변경 API를 legacy FullLoad가 위치 독립적으로 지원한다고 해석하지 않는다.

## 네 방법 동일 fixed-history 진단

```bash
"$PILOT_PY" scripts/132_validate_mpic_fixed_history_gpu.py \
  --run "$PILOT_OUT/mmdu_ssd" --out "$PILOT_OUT/fixed_history_gpu"
```

측정이 끝난 대화31 Prefix의 실제 prediction을 공통 fixed-history로 사용하여3turn×4methods, cap3을 실행한다. module projection shape 및 actual native attention cache를 hook으로 저장한다. 이12건은 generation pilot/품질/TTFT 요약에 추가하지 않는다. primary run `fixed_history_gpu`.

## 과거 미검증36건 복원

```bash
"$PILOT_PY" scripts/128_resolve_fullload36.py \
  --prior results/mpic_gpu_mmdu_20261013_01/evidence/prior_fullload_impact.json \
  --out "$PILOT_OUT/fullload36"
```

원본 raw log/VisDial hash-checked index/당시 prompt가 필요하다. GPU는 필요 없다. prompt SHA 및 dtype+shape+binary suffix tensor SHA를 동시에 대조한다. primary authoritative 결과는 **`fullload36_final`**이다. 초기 `fullload36` 디렉터리는 진행 중 결과를 보존한 것으로 최종 판정을 덮어쓰지 않는다.

## 실제 local judge와 독립 검산

```bash
"$PILOT_PY" scripts/129_judge_mpic_mmdu.py \
  --run "$PILOT_OUT/mmdu_ssd" --out "$PILOT_OUT/judge_qwen"

"$PILOT_PY" scripts/131_report_mpic_gpu_pilot.py \
  --run "$PILOT_OUT/mmdu_ssd" --judge "$PILOT_OUT/judge_qwen" \
  --out "$PILOT_OUT/ssd_verified"

"$PILOT_PY" scripts/131_report_mpic_gpu_pilot.py \
  --run "$PILOT_OUT/mmdu_gpu_resident" \
  --out "$PILOT_OUT/gpu_resident_verified"
```

129는7차원 integer1..10 full dict를 처음 판독하며 원본 출력/실패를 보존한다. 131은 six-dimension dict + 유일하게 명시된 integer Overall Score도 읽지만 누락/N/A를 추정하지 않는다. 이 때문에 primary judge `status.json`의52/8과 최종59/1이 다르며 둘 다 보존됐다. authoritative quality는 **`ssd_final_verified/quality_paired.json`**, coverage는 `judge_failure_coverage.json`. 공통14턴에서 비교하고 실패한 대화35turn2 Full reuse는 모든 방법의 paired 비교에서 제외한다. all-valid-per-method 점수는 별도로 저장한다. 이는 parser 회복이며 judge retry/채점 기준 완화가 아니다.

로컬 quantized Qwen judge는 paper ChatGPT의 exact reproduction이 아니다. full rubric을 로컬 `papers/mpic.md`에서 가져와 `rubric.txt`에 기록한다. 두 조건의 모든 candidate/input/history 동일성을 확인했으므로 같은 judge 결과를 두 조건에 연결할 수 있다. 독립 두 번의 judge 결과로 집계하지 않는다.

## 이번 완료 상태를 다시 읽기

```bash
"$PILOT_PY" - <<'PY'
from pathlib import Path
import json
p = Path('results/mpic_gpu_mmdu_20261013_01')
for name in ('gpu_gate_attempt1/status.json', 'fullload_gpu_isolation/validation.json',
             'fixed_history_gpu/validation.json', 'ssd_final_verified/status.json',
             'gpu_resident_verified/status.json', 'FINAL_STATUS.json'):
    value = json.loads((p / name).read_text())
    print(name, value.get('status'), value.get('method_requests'))
assert json.loads((p / 'condition_equivalence.json').read_text())['matched_requests'] == 60
print(json.loads((p / 'ssd_final_verified/judge_failure_coverage.json').read_text()))
PY
```

보고서는 이전 NOT VERIFIED 기록을 변경하지 않고 최종 lineage를 연결한다. 생성120건과 judge60건, fixed-input 진단과 warm-up을 구분한다. artifacts inventory는 `ARTIFACTS.json`, source/preservation 증거는 `source_sha256.json`, `evidence/preservation.json`이다. 저장한 KV payload의 integrity는 각 store `integrity.json`의 sampled hash이며 전체 weight/KV payload full SHA를 측정했다는 뜻은 아니다.
