# GLM-4.6V-Flash v4 재현

프로젝트: `/home/dblab/hj/mllm_v2`. 실제 run은 `runs/glm46v_prefixstore3_v4_20261007T072841Z`, 파생 보고서는 같은 이름의 `results/`에 있다. 원래 v3 실행과 분모·raw·실패·source freeze를 합치지 않는다. Remote push, 기존 artifact 삭제, 전역 dependency 변경은 필요하지 않다.

## 정확한 재현 identity

현재 실행의 authoritative source/config/contract는 `active_freeze.json`이 가리키는 `freezes/pre_pilot_v2/`다. Freeze SHA256은 `e8d5868ae33c2a77e19eceaa09349421febc9320de8876bf54e6a5a8772a9b50`, validation manifest는 `9a73f3c8513482955958bec3c1462211326ca29e1c5c98c5ace43813e2edab7e`다. `source_frozen/` 아래19파일의 전체 hash와 config/environment/contract/workload hash가 freeze에 있다. 이후 source 변경이 있으면 현재 run의 resume를 거부해야 하며 frozen source를 기존 작업 트리에 덮어쓰지 않는다.

| Membership | 파일 SHA256 |
|---|---|
| Main `manifests/main_manifest.json` | `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224` |
| GQA `manifests/gqa_pilot_manifest.json` | `40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895` |
| MT `manifests/mt_pilot_manifest.json` | `d935dd7f65adee6c05a7c9ce5dbe38f22dbdc9fda9ac989703892817e8118e46` |

Main ordered dialogue/turn/question SHA256은 `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`다. 398images/4,061dialogues/3turns, 방법당12,183 및8,122followups, 총36,549 final 요청이다. GQA는40images×6독립질문×3methods=720(history 없음), MT는40dialogues×3turns×3methods=360다. Manifest/질문/gold/order를 새로 선택하지 않는다.

Pinned model·processor·tokenizer는 `zai-org/GLM-4.6V-Flash@411bb4d77144a3f03accbf4b780f5acb8b7cde4e`다. Model class는 native Glm4vForConditionalGeneration, processor는 Glm46VProcessor다. Four safetensors shard SHA256/bytes, checkpoint metadata와 installed Transformers source hashes는 active frozen config와 environment에 있다. V4 checkpoint symlink는 보존된 v3 `runs/glm46v_prefixstore3_20261007T065207Z/checkpoint`로 resolve한다. 이 공유 weight 위치를 결과정리 과정에서 제거하지 않는다.

`.venv-glm46v`는 Python3.10.20 venv이며 `include-system-site-packages=true`로 기존 읽기 전용 mllm_ft의 Torch2.5.1+cu121/torchvision0.20.1+cu121을 상속하고 Transformers5.13.1을 격리 설치했다. BitsAndBytes0.49.2, Accelerate1.14.0 등 실제 버전은 GPU fingerprint/environment를 따른다. 상속 원본 환경을 업그레이드하지 않는다. Native runner CPU processor tests의 checkpoint fixture 경로도 v3 위치를 참조하므로 이 metadata/processor 파일들을 보존해야 한다. Weight를 CPU tests에서 로드하지 않는다.

공통 설정은 RTX4090/decoder-only NF4 double/BF16 compute·native KV/SDPA/batch1/seed1234/eval-inference/greedy cap32, vision·merger·lm_head BF16/no offload다. Size={shortest_edge:12544,longest_edge:1605632}, enable_thinking=False, tools없음. Native closed answer-box token span의 최종 답변 추출을 세 arm 모두에 적용하고 raw IDs/text를 보존한다. Strict normalized EM/gold[0] 규칙은 바꾸지 않았다. Cap32는 기존 Qwen/LLaVA 및 v3의16과 다른 main이전 고정 결정이며 v3 truncation 및 parser omission 기록을 보존한다.

## 명령

아래 GPU 명령은 실제 GPU driver 접근이 되는 서버 terminal에서 실행한다. Sandbox의 CUDA mask를 GPU 부재로 해석하지 않는다. 다른 GPU 프로세스를 종료하지 않고 timing을 중단/대기한다. 최소30GiB reserve와 frozen resource plan을 유지한다.

```bash
cd /home/dblab/hj/mllm_v2
export GLM_PYTHON=/home/dblab/hj/mllm_v2/.venv-glm46v/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests
export GLM_RUN=/home/dblab/hj/mllm_v2/runs/glm46v_prefixstore3_v4_20261007T072841Z
```

CPU checks에는 새 GPU inference가 없다. 기존 timed GPU 작업과 겹쳐 전체 suite/분석을 실행하지 않는다.

```bash
"$GLM_PYTHON" -m unittest discover -s tests -p 'test_glm46v_*.py' -v
```

현재 final run의 **미완료 main만** 같은 freeze로 재개한다. 이미 COMPLETED된 phase를 다시 실행하면 runner는 거부한다. 완료 여부는 실제 receipt에서 확인한다.

```bash
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py   --run-dir "$GLM_RUN" --phase main --resume
```

완전한 새 재실행은 충돌 없는 새 run에 원래 membership/image identities를 복사하고 pinned checkpoint를 연결한 뒤119→120→GQA→MT→main 순서로 한다. Parent run의 PASS receipt/COMMITTED를 복사하여 gate를 우회하지 않는다. 아래 `GLM_NEW_RUN`은 반드시 없는 경로여야 한다. Native source가 현재 final freeze와 같을 때만 이 recipe를 사용한다.

```bash
export GLM_NEW_RUN=/home/dblab/hj/mllm_v2/runs/glm46v_prefixstore3_v4_REPRODUCE_NEW
"$GLM_PYTHON" - <<'PY_BOOTSTRAP'
from pathlib import Path
import json, os, shutil, hashlib
old = Path(os.environ['GLM_RUN'])
new = Path(os.environ['GLM_NEW_RUN'])
active = json.loads((old/'active_freeze.json').read_text())
freeze = json.loads((old/active['path']).read_text())
root = Path('/home/dblab/hj/mllm_v2')
for name, expected in freeze['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
new.mkdir(parents=True, exist_ok=False)
shutil.copytree(old/'manifests', new/'manifests')
(new/'checkpoint').symlink_to((old/'checkpoint').resolve(), target_is_directory=True)
for name in ('REQUEST.md', 'baseline_protection_cpu_audit.json'):
    shutil.copy2(old/name, new/name)
PY_BOOTSTRAP
"$GLM_PYTHON" scripts/119_prepare_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN"
"$GLM_PYTHON" scripts/120_validate_glm46v_prefixstore3.py   --run-dir "$GLM_NEW_RUN" --checkpoint "$GLM_NEW_RUN/checkpoint"
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN" --phase gqa_pilot
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN" --phase mt_pilot
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN" --phase main
```

새 run은 새 environment/config/created time과 새 raw/attempt IDs를 기록한다. Model/processor/algorithm/manifest 정책을 그대로 유지하는 재실행이며 같은 latency·prediction의 수치적 재현을 보장하지 않는다. Processor fixture가 없는 machine에서는 runner CPU checks가 skip되는 한계를 명시하고 같은 pinned metadata를 갖추어야 한다.

GPU timing이 끝난 후 채택 COMMITTED raw/setup에서 독립 분석한다. Output directory는 새 경로이며 overwrite를 거부한다. 기본 loader는 adopted image attempts만 읽고 raw/setup hash를 검증한다.

```bash
"$GLM_PYTHON" scripts/122_report_glm46v_prefixstore3.py   --run "$GLM_RUN" --phase main   --output /home/dblab/hj/mllm_v2/results/glm46v_prefixstore3_v4_REPLAY_ANALYSIS_NEW
```

Partial image는 동일 freeze 아래 완전한 새 attempt로 다시 실행하고 첫 완전한 성공만 final에 채택한다. Failed attempt/STARTED events/raw와 원래 v3 실패를 보존한다. Resume는 source/config/contract/workload/gate/pilot freeze와 committed raw/setup identity를 검사한다. 실행별 setup은 이미지/방법당 한 번, standalone-equivalent는 같은 setup을 대화마다 재부과한 DERIVED 값이다. 비-source capture 증분은 NOT_MEASURED다.

Store payload는 final run-owned image attempt의 `stores/{fullload,ours_prefixstore_kv25}`에 있다. Validation 진단은 sealed store 밖 `sample_*/diagnostics/`에 있다. 향후 run-owned payload 정리 시 반드시 full hash·독립 audit·atomic adoption·정확 allowlist receipt를 남기며, 정리된 payload의 재현은 해당 이미지의 full T1과 자기 generated history/store를 새로 생성해야 한다. 기존 artifact·원래 checkpoint는 삭제하지 않는다.

OS pread/write bytes는 NAND 물리traffic이 아니다. DONTNEED는 page-cache hint이며 cold NAND 보장이 아니다. MB=10^6 bytes, GiB=2^30 bytes, ms, 정확도차이는percentage points다. Paired image-cluster bootstrap10000/seed1234는 같은 이미지의 모든 방법·대화·turn을 함께 resample한 pooled sum/count를 사용한다. CI가0을 포함해도 동등성 증거가 아니다. Cache saturation/hit-rate/eviction 이득은 이 실행에서 측정하지 않았다.
