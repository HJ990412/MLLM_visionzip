# 재현

위치 `/home/dblab/hj/mllm_v2`, 기존 Python `/home/dblab/anaconda3/envs/mllm_ft/bin/python`. torch2.5.1+cu121/Transformers4.57.6/NumPy2.2.6. 이번 환경은 GPU driver unavailable, CUDA device_count0. CPU tests34개 및 layer16개를 실행했고 pretrained 모델은 로드하지 못했다.

모든 명령은 **새 output path**를 요구하며 기존 source/result/cache를 덮어쓰지 않는다. 이전 frozen main config/source SHA는 변경하지 않았다. 기존 main runner를 재실행하거나 기존 frozen run을 resume하지 않는다.

```bash
cd /home/dblab/hj/mllm_v2
MPIC_PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
MPIC_REPLAY_ROOT=$(mktemp -d /tmp/mpic-multi-replay.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

"$MPIC_PY" -m unittest discover -s tests -p test_mpic_multi.py -v
"$MPIC_PY" -m unittest discover -s tests -p test_mpic.py -v
"$MPIC_PY" -m unittest discover -s tests -p test_llava_mt_gqa_B.py -v

"$MPIC_PY" scripts/124_validate_mpic_multi.py --out "$MPIC_REPLAY_ROOT/cpu"
"$MPIC_PY" scripts/127_check_fullload_before_after.py \
  --before-source results/mpic_multi_fix_20261013_01/source_before/serve.py \
  --out "$MPIC_REPLAY_ROOT/before_after.json"
"$MPIC_PY" scripts/126_audit_fullload_impact.py \
  --out "$MPIC_REPLAY_ROOT/impact.json"

# CPU preprocessing only: no pretrained weights and no generated responses.
"$MPIC_PY" scripts/125_eval_mpic_mmdu_smoke.py \
  --out "$MPIC_REPLAY_ROOT/mmdu_prepared" --max-dialogues 5 --max-new-tokens 256
```

기대 결과: 새 tests9/기존 MPIC17/main safety8 PASS, CPU selective8조건/16레이어 PASS, archived FullLoad0 KV bytes→fixed10,240 bytes/4reads. CPU shim은 regression process 안에서만 CUDA sync를 no-op으로 대체한다. CPU tensor를 GPU tensor라고 보고하지 않는다. 수치 기준은 rtol1e-5/atol1e-6으로 고정했다.

GPU에 접근할 수 있게 되면 아래 두 조건을 각각 새 디렉터리에서 실행한다. GPU가 없으면 status NOT VERIFIED와 empty answers/header-only CSV를 저장하고 **exit3**으로 종료한다. 다른 precision/backend로 fallback하지 않는다.

```bash
"$MPIC_PY" scripts/125_eval_mpic_mmdu_smoke.py \
  --out "$MPIC_REPLAY_ROOT/mmdu_ssd" --execute --condition ssd_cold \
  --max-dialogues 5 --max-new-tokens 256
"$MPIC_PY" scripts/125_eval_mpic_mmdu_smoke.py \
  --out "$MPIC_REPLAY_ROOT/mmdu_gpu_resident" --execute --condition gpu_resident \
  --max-dialogues 5 --max-new-tokens 256
```

GPU commands load pinned `llava-hf/llava-v1.6-vicuna-7b-hf` snapshot `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 double quant/BF16/eager, batch1. Native model context limit4096. GPU process exclusivity is checked. First dialogue's functional gate executes single/changed-prefix/interleaved/reordered/fixed-history inputs for k0/32/all, and real FullLoad suffix1/2 plus wrong cache ID rejection. k-all is compared with native dense over the identical persisted visual inputs at predeclared rtol/atol1e-4. Partial k32 is checked by actual installation/retention/shape/causality, without forcing dense equivalence. These GPU branches remain **unexecuted/unverified** in the present environment.

MMDU path `data/mmdu/raw/benchmark.json`, SHA `31dd8b7c15416bf22be86a4077e6b755b8adc48226f6ca6796129333385af080`; source110 dialogues/1645 turns, not45k training. Subset source IDs31/40/25/69/35; first3 turns each. Rule seed1234 SHA order, exactly2 images introduced by turn3, cap256 reserve plus64 boundary tokens under4096. All inclusion/exclusion records and image hashes are in the manifest. Default preparation runs the actual pinned processor on the selected inputs, never a random decoder.

Every source image is resized wholly to336×336 Lanczos according to the official MMDU LLaVA-NeXT generation script. This resize differs from the untouched AnyRes GQA main and from unconfirmed MPIC paper preprocessing. Prompt is Vicuna USER/ASSISTANT, generated history only; original image insertion order is retained. Official MMDU's generic [INST]/cap2048 are not silently presented as this smoke's settings. Cap256/truncation status is explicit; no EM quality substitute/judge execution is included.

TTFT outer start precedes input construction and ends after first argmax host materialization+CUDA sync. Provisioning/metadata activation are separate setup; warm-ups are not measurement rows. GPU-resident setup bytes are separate from per-request SSD bytes. DONTNEED advisory occurs before SSD-cold request timers. No NAND-cold guarantee. KV/os.pread byte counters exclude ordinary pixel-image reads, which may still occur in prefix misses.

Native prefix state contains only the previous prompt KV, not decode-output KV. It reuses only actual equal ID/image-identity positions; fully matching image spans skip pixel reading/vision. Full reuse is a two-pass independent-image reuse comparator, not the repaired fixed-prefix FullLoad. MPIC uses per-image32 (64 image rows for2 images); global32 is available as an explicit diagnostic choice. The paper's multi-image k policy remains unverified.

Final evidence in this run: `evidence/cpu_evidence/`, `evidence/fullload_before_after_final.json`, `evidence/fullload_impact.json`, `evidence/mmdu_run/`. Earlier bounded attempts are retained, not overwritten. GPU requested command exited3 after preparing5 dialogues/15 turns; completed pretrained0 dialogues/0 turns. Judge/quality/TTFT stay NOT VERIFIED.
