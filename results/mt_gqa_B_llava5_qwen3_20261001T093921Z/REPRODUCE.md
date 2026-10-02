# Reproduce / resume B full main

Run `/home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z`. GPU models sequential, source/config/contract frozen. Qwen integration G7 FAIL blocks smoke/main; never bypass its gate.

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
/home/dblab/anaconda3/envs/mllm_ft/bin/python /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/launch_llava_main.py
# After main complete, use a new output directory for every independent report:
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/113_report_mt_gqa_B.py --run /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z --output /home/dblab/hj/mllm_v2/results/mt_gqa_B_llava5_qwen3_20261001T093921Z/reaudit_01
```

The actual main runner is scripts/108_eval_llava_mt_gqa_B.py. The launcher verifies source/config/integration, runs real smoke then main, and lets the new independent 113 reporter produce all requested tables. It does not use the old 109 renderer's full-store wording. Resume accepts only evidence-bound image commits, fresh full image attempt otherwise, no choice by latency/accuracy. STOP_REQUESTED in llava/ halts at an image boundary. Old attempts/payloads remain until independently authorized cleanup; no old run deletion.

For a complete new reproduction create a NEW root through the archived bootstrap snapshot policy, then 112 --stage freeze, --stage cpu, each model --stage core, --stage integration, and only PASS models' smoke/main. Copying PASS receipts is insufficient. Exact launch script is bound in main_launch_freeze.json. Previous final B source and its model-specific replay remain referenced by baseline_source_correspondence.json. The original Qwen main FP32 failure and this new B failure are immutable.

Full main committed payloads are intentionally removed after full checksums, independent audit, close, atomic commit and exact-owned allowlist cleanup. Metadata, hash inventory, requests, physical events and mappings remain. Reproduce payloads by rerunning each image's full source T1 with the exact pinned model/processor/source/config, then all its original dialogues in original ordinal rotation, generating new own-method histories. Validation controls retain their own A/B stores separately. Prior pilot setup numbers never substitute current measured setup.

Native request/component intervals can overlap; only request E2E and whole physical persistence/activation event wall times determine critical totals. AllHead independent deployment costs are derived from the measured shared canonical bundle. Non-source capture increment remains unmeasured. Historical large-file protection uses bounded fingerprints, not full SHA; all new raw/payload records have full SHA.

## Complete fresh reproduction from the current workspace

The setup scripts used in this execution are archived in this run. The following sequence creates a new output root and preserves all prior data. The current model/data artifacts and source hashes must remain available.

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
$PY /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/protection_bootstrap.py
EXPERIMENT_RUN=$(cat /tmp/main_b_run)
$PY scripts/112_control_mt_gqa_B.py --run "$EXPERIMENT_RUN" --stage freeze
$PY scripts/112_control_mt_gqa_B.py --run "$EXPERIMENT_RUN" --stage cpu
# These are immutable full-hash validator INPUT records, not copied PASS receipts.
cp /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/llava/protected_artifacts_before.jsonl "$EXPERIMENT_RUN/llava/protected_artifacts_before.jsonl"
$PY scripts/112_control_mt_gqa_B.py --run "$EXPERIMENT_RUN" --stage core --model qwen
$PY scripts/112_control_mt_gqa_B.py --run "$EXPERIMENT_RUN" --stage integration --model qwen
# The current fixed FP32 G7 failure blocks Qwen. Preserve it; proceed independently with LLaVA.
$PY scripts/112_control_mt_gqa_B.py --run "$EXPERIMENT_RUN" --stage core --model llava
$PY scripts/112_control_mt_gqa_B.py --run "$EXPERIMENT_RUN" --stage integration --model llava
$PY /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/prepare_main_launcher.py
$PY "$EXPERIMENT_RUN/launch_llava_main.py"
$PY scripts/113_report_mt_gqa_B.py --run "$EXPERIMENT_RUN" --output "/home/dblab/hj/mllm_v2/results/$(basename "$EXPERIMENT_RUN")/final"
```

The actual AllHead validator verifies every required file against the copied INPUT hashes and executes fresh reference comparisons. A missing/changed old input store fails validation. The original ten sample order and all numerical tolerances remain unchanged. Do not use `set -e` to let Qwen's documented validation failure inadvertently prevent the independent LLaVA path. Never force Qwen main after failure. The launcher admits LLaVA only after all gates, then separate smoke, then complete main. A future resolved Qwen integration may use 110's smoke/main entry only under a new prospective freeze and the original criterion.

`monitor_status.py` is operational only and contains this execution's PID; update the PID for a new process. It is never a source of quality/latency metrics. The full artifact index produced at report time is a snapshot; a post-finalization index can additionally bind terminal logs that were still open during report generation.

## Terminal capacity and artifact receipts

For this execution, wait until `finalize_after_main.py` has exited successfully and `final_analysis_complete.json` exists. Then run the frozen terminal supplement directly (do not redirect its output into a file included in its own inventory):

```bash
cd /home/dblab/hj/mllm_v2
PYTHONPATH=.:scripts PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/report_detail_supplement_v2.py
PYTHONPATH=.:scripts PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/allhead_native_timing_supplement.py
PYTHONPATH=.:scripts PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python /home/dblab/hj/mllm_v2/runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/capacity_and_terminal_receipts_v3.py
```

This execution-specific supplement reads the independent final storage table, observes terminal free space, and estimates KV-only capacity after reserving 30 GiB. It does not measure simultaneous cache population, eviction or hit rate. The immutable original seven-table report stays at `final/REPORT.md`; root `REPORT.md` adds the capacity supplement. `artifact_index_final.json` is the authoritative terminal full-SHA snapshot, created after logs have closed, and excludes only itself. The supplement uses exclusive creation and must not be rerun into the same completed output. New reproduction runs require an equivalent run-local supplement and their own completion evidence.

The separately frozen detail supplement adds paired_analysis.json, complete session attribution/CI tables, MPIC components, H2D availability and V chunk/amplification tables from audited CSVs and committed raw. It runs after the GPU main and independent finalizer. The v2 terminal supplement includes these details before capacity and the final full-SHA index. The original unexecuted terminal script and its freeze remain preserved.

The v2 detail script also publishes actual activation auxiliary/structural residency and separately identified service integrity scans. Original pre-execution detail script/freeze remain archived. It clarifies the original report template wording: B retains the top 25% content for attention but retains LLaVA dense GPU buffers; no 75% GPU-memory reduction is asserted.

The final native timing supplement reads existing recorded host/CUDA stage intervals, without changing or rerunning inference. It separates pure scoring-stage intervals from the read/H2D/assembly-inclusive selector host interval and adds the explicit 14-stage status matrix. Terminal supplement v3 includes this report; prior unexecuted terminal versions remain archived.
