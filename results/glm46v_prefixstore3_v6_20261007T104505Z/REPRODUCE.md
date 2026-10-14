# GLM-4.6V-Flash v6 재현

프로젝트는 `/home/dblab/hj/mllm_v2`, final 연구 run은 `runs/glm46v_prefixstore3_v6_20261007T104505Z`, 파생 결과는 같은 이름의 `results/`다. v3/v4의 raw·실패·v4 partial 333과 v5 전체 36,549 및 inner-scope setup은 별도 archive다. 이를 v6 final에 합치지 않는다. Main 완료 여부는 실제 `main/COMPLETED.json`과 최종 분석으로 확인한다.

## 실제 freeze와 모델

현재 v6는 active pointer 없이 run root의 **`freeze.json`**을 사용한다. SHA256 `1f6f3006a8399c8b62f60d910e422e13d4d644e63fe622bfc05913ed331bb77e`. Config SHA256 `a00c8b581725755800ffd2194bcbc4cf5c2efc33b4b7c6dabfc0a774ea918d71`, environment SHA256 `99b88dd96778ed10adeb28abf37ee74c8507118ad413ccafe1367560781122cb`, contract SHA256 `e19f963a089a3d07bde3aed7d15ea71148bab474a193fc56cbe13398246f04a4`, validation membership SHA256 `1d79a90ddd1c84901c0d382e6436bfa744d652295884bbedf67f76e735cf442a`. 이 파일이 20개 source 파일와 workload/manifests를 full SHA로 결속하며 source_frozen/에 실제 snapshot을 보관한다. 이전 v4 active-freeze나 v5 freeze를 v6 재현 경로로 사용하지 않는다.

Pinned checkpoint/processor/tokenizer는 `zai-org/GLM-4.6V-Flash@411bb4d77144a3f03accbf4b780f5acb8b7cde4e`. Native model class는 Glm4vForConditionalGeneration, processor는 Glm46VProcessor다. 실제 four shard SHA256/bytes/download receipt와 checkpoint metadata hash는 config.json에 있다. V6 checkpoint는 보존된 original v3 checkpoint를 read-only로 공유한다. Native processor/history analyzer CPU fixture도 `runs/glm46v_prefixstore3_20261007T065207Z/checkpoint`를 참조하므로 해당 pinned weights/metadata를 삭제하지 않는다.

전용 `.venv-glm46v`는 Python3.10.20 및 include-system-site-packages=true로 기존 mllm_ft의 읽기 전용 Torch2.5.1+cu121/torchvision0.20.1+cu121을 상속하고 Transformers5.13.1을 격리한다. 실제 bitsandbytes0.49.2/accelerate1.14.0/huggingface_hub1.33.0/tokenizers0.22.2/Pillow12.2.0 등은 environment와 GPU fingerprint를 따른다. Installed Transformers model/cache/mask/SDPA source의 hash도 environment에 있다. 상속 환경이나 전역 dependency를 업그레이드하지 않는다.

RTX4090, decoder-only NF4 4-bit/double quantization/BF16 compute·native KV/SDPA/batch1/seed1234/eval-inference/greedy **cap32**를 세 방법에 동일하게 사용한다. Vision/merger/lm_head는 BF16 비양자화이며 CPU/disk offload는 없다. Size={shortest_edge:12544,longest_edge:1605632}, enable_thinking=False, tools 없음. 실제40layers/2KVheads/128의 BF16 KV는40,960 bytes/token이다. Cap32는 기존 LLaVA/Qwen 및 v3 cap16과 다른 main이전 고정 결정이며 v3 truncation을 보존한다.

Prediction은 native closed answer-box span의 scoring answer이고 기존 strict normalized EM/gold[0]는 그대로다. History는 실제 generated IDs의 마지막 native EOS 한 개만 제외한 literal native body다. Box/thinking controls/whitespace를 보존하고 native chat template로 처리한다. 실제 EOS집합은151329/151336/151338/151348이다. Raw IDs/전체decoded text/bodyIDs/bodytext/추출 상태/lineage를 남긴다. Gold나 다른 방법의 답변을 history에 넣지 않는다. Malformed/empty/cap/stray marker는 기록·분석하며 token suppression/forced EOS/좋은 출력까지 재시도는 하지 않는다.

## 고정 membership

| 파일 | SHA256 |
|---|---|
| Main `manifests/main_manifest.json` | `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224` |
| GQA `manifests/gqa_pilot_manifest.json` | `40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895` |
| MT `manifests/mt_pilot_manifest.json` | `d935dd7f65adee6c05a7c9ce5dbe38f22dbdc9fda9ac989703892817e8118e46` |

Ordered main dialogue/turn/question SHA256은 `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`. Main은398 images/4061 dialogues×3turns, 방법당12183 요청/8122followups, 총36549 final 요청이다. GQA40×6독립질문×3methods=720(history 없음), MT40×3turns×3methods=360. 기존 질문·gold[0]·순서·membership을 재선택하지 않는다.

## 실행 명령

GPU 명령은 실제 driver 접근이 되는 서버 terminal에서 실행한다. Sandbox에서 CUDA를 볼 수 없다는 이유로 물리GPU 부재라고 판단하지 않는다. 다른 GPU 프로세스를 종료하지 않고 timing을 중단/대기하며 최소30GiB reserve와 resource plan을 지킨다.

```bash
cd /home/dblab/hj/mllm_v2
export GLM_PYTHON=/home/dblab/hj/mllm_v2/.venv-glm46v/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests
export GLM_RUN=/home/dblab/hj/mllm_v2/runs/glm46v_prefixstore3_v6_20261007T104505Z
```

CPU suite를 실제 timed GPU 요청과 겹쳐 실행하지 않는다. 현재 전체 suite evidence는86 passed다.

```bash
"$GLM_PYTHON" -m pytest -q tests/test_glm46v_*.py
```

이미 시작했지만 **미완료인 main만** unchanged freeze로 재개한다. COMPLETED된 phase는 runner가 재실행을 거부한다.

```bash
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py \
  --run-dir "$GLM_RUN" --phase main --resume
```

전체 새 재실행은 없는 새 경로에서 원본 manifests/image identities·REQUEST를 복사하고 pinned checkpoint를 연결한 뒤119→120→GQA→MT→main 순서로 한다. 다른 run의 PASS/COMMITTED를 복사하여 admission을 우회하지 않는다. 다음 recipe는 현재 workspace source가 final v6 freeze와 같을 때만 사용한다. 새 이름이 이미 있으면 mkdir이 거부한다.

```bash
export GLM_NEW_RUN=/home/dblab/hj/mllm_v2/runs/glm46v_prefixstore3_v6_REPRODUCE_NEW
"$GLM_PYTHON" - <<'PY_BOOTSTRAP'
from pathlib import Path
import json, os, shutil, hashlib
old = Path(os.environ['GLM_RUN'])
new = Path(os.environ['GLM_NEW_RUN'])
freeze = json.loads((old/'freeze.json').read_text())
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
"$GLM_PYTHON" scripts/120_validate_glm46v_prefixstore3.py \
  --run-dir "$GLM_NEW_RUN" --checkpoint "$GLM_NEW_RUN/checkpoint"
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN" --phase gqa_pilot
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN" --phase mt_pilot
"$GLM_PYTHON" scripts/121_eval_glm46v_prefixstore3.py --run-dir "$GLM_NEW_RUN" --phase main
```

새 실행은 새 timestamp/config/environment/raw/attempt IDs를 기록한다. Model/processor/algorithm/membership 정책을 유지하는 재현이며 개별 latency·prediction의 수치적 동일성을 보장하지 않는다. Weights를 CPU tests에서 로드하지 않는다. Original pinned processor fixture가 없으면 native CPU checks의 skip 한계를 명시하고 해당 metadata를 갖추어야 한다.

GPU timing 완료 후 adopted raw와 setup events를 분석한다. Output은 새 경로이며 overwrite를 거부한다. Loader는 COMMITTED image attempts만 읽고 raw/setup full hash와 ownership/count를 검증한다.

```bash
"$GLM_PYTHON" scripts/122_report_glm46v_prefixstore3.py \
  --run "$GLM_RUN" --phase main \
  --output /home/dblab/hj/mllm_v2/results/glm46v_prefixstore3_v6_REPLAY_ANALYSIS_NEW
```

Partial image는 같은 frozen 설정에서 새 attempt로 전체 image를 다시 수행하고 첫 완전한 성공을 채택한다. Failure/STARTED/partial raw를 보존한다. Output anomaly는 retry 조건이 아니다. Source/config/environment/contract/manifest/gate/pilot freeze가 달라지면 resume를 거부한다. Main의 native actual-generated-body history와 source/store identity를 독립 tokenizer로 감사한다.

Image-major stream에서 각 이미지/방법의 source T1만 full capture/build하고 같은 이미지의 후속 dialogue에서도 T1은 full pixels다. 필요 setup은 실제 request E2E에 중복 합산하지 않고 image별 한 번만 attribution한다. Standalone-equivalent는 같은 setup을 대화마다 재부과한 DERIVED 값이며 non-source capture 증분은 NOT_MEASURED다.

Run-owned payload는 image attempt의 `stores/{fullload,ours_prefixstore_kv25}`에 있으며 validation 진단은 sealed store 밖 `sample_*/diagnostics/`에 있다. 향후 정리하면 full hash·독립 audit·atomic adoption·정확 allowlist receipt가 필요하고 해당 full T1/native history/store를 재생성해야 한다. 기존 artifact/원래 checkpoint를 삭제하지 않는다. 대용량 raw/store/weights는 local 경로와 full hash를 보관하며 GitHub 업로드나 remote push가 필요하지 않다.

OS pread/write는 NAND 물리I/O가 아니며 DONTNEED는 cold NAND 보장이 아니다. MB=10^6 bytes, GiB=2^30 bytes, ms, 정확도차이는percentage points다. Paired image-cluster bootstrap10000/seed1234는 모든 방법·대화·turn을 image별 함께 resample한 pooled sum/count를 쓴다. CI0포함은 동등성 증거가 아니다. Cache saturation/hit-rate/eviction 효과는 검증하지 않았다.

## V6 outer setup와 source 변경 범위

V5의 기록된 setup은 native writer-call/activation-core 범위로 identity 준비와 wrapper return tail이 NOT_MEASURED였다. V6는 동일한 native inference 5개 module SHA, 모델·processor·body·cap32·manifest를 유지한 새 run이다. Driver 121 caller에서 runner.persist()/runner.activate() 진입 전부터 완전한 반환까지 elapsed를 측정한다. Identity 준비, expected identity 등록, IO summary와 return dict tail을 포함하며 duration_ms가 이 outer 값이다. inner_writer_call_return_ms와 inner_activation_core_ms는 nested diagnostic으로 따로 보존하고 합산하지 않는다. Source clone/importance를 E2E와 setup에 중복 청구하지 않는다.

V6의 freeze.json/config.json/CONTRACT.md/source_frozen이 실제 재현 대상이다. Official TF source pin은 official_source_provenance.json을 읽는다. V5의 scalar setup에 미측정 시간을 추정해 더하지 않고 v6의 새 outer setup events로 shared-stream/derived session을 계산한다. 기존 v5 결과·receipt·raw·source freeze는 보존한다. V6 GQA720/MT360 및main36,549요청은 모두PASS다. 최종 완료·품질·latency·session·paired CI는 v6 COMPLETED와 raw 분석을 따른다.

## V6 terminal 완료 보충

Main은398 images/4,061 dialogues/36,549 final 요청, 방법당12,183 및8,122 followups로 완료됐다. Failed image attempt·retry는0이다. COMPLETED SHA256 `013a32a7423b1131211516b48d4eacbb2102bfc395739088cbd3be060eb1f609`. GQA720 및MT360도 별도 완료했다. 이전v3/v4/v5 요청은 primary 분모에0건이다.

독립 `independent_review/attempt_20261007T130724092714Z/receipt.json`는 모든 native scoring/body/lineage/prompt/inputIDs/3축positions/I/O 및54 paired point/CI를 원래 raw에서 검산해 PASS했고 numeric residual은0이다. Seed1234/10,000 image clusters/pooled sum-count를 사용했다. Actual caller outer physical setup과 session attribution 합은 모두55,670.386902522296ms다. Writer의새outer−inner2,089.627110864967ms 및activation41.401019319892ms가 실제 포함됐으며 nested 값을 중복 합산하지 않았다. Terminal 수치와 해석/한계는 FINAL_ANALYSIS_REVIEW.md 및FINAL_ANALYSIS_REVIEW_RECEIPT.json을 따른다.

V6 main은 이미 COMPLETED이므로 이 보존 run에 --resume를 다시 실행하지 않는다. 새로운 재현은 위 새 run recipe를 사용하고 다른 run의 PASS/COMMITTED를 복사하여 gate를 우회하지 않는다.
