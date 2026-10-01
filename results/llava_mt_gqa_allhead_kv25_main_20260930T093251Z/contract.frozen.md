# LLaVA MT-GQA AllHead KV25 main contract

Version `llava-mtgqa-allhead-kv25-main-v1`, frozen before main on 2026-09-30.
This file specifies requirements; only executed receipts establish PASS.
Run: `runs/llava_mt_gqa_allhead_kv25_main_20260930T093251Z/`.
Results: `results/llava_mt_gqa_allhead_kv25_main_20260930T093251Z/`.

## Fixed population and execution

The complete `data/mt_gqa/dialogues.json` must have SHA256
`2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224` and ordered
workload hash `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`.
398 images, 4,061 three-turn dialogues, 12,183 requests/method, 60,915 final
requests and 40,610 T2/T3 comparison requests. ReComp T2/T3 is a comparison
cohort, not an SSD cache hit. This is reconstructed MT-GQA, not the exact
MetaCompress artifact. Membership, questions, gold and order stay frozen.
The old dataset config's `gold_teacher_forced` describes its original use;
this run explicitly uses method-local generated history and does not inherit it.

The new runner is `scripts/100_eval_llava_mt_gqa_allhead_kv25.py`; its lifecycle,
causal prompt, strict scorer, source-only persistence and global dialogue
ordinal rotation follow 73. Its actual AllHead/native Ours calls follow the
validated 98 pilot without importing that pilot's registry. Old runners stay
unchanged. ReKV, Probe3, legacy QA-Token/QA-Chunk, Ours-Chunk25, Qwen inference
and MT-VQA inference are excluded. Existing CPU fixtures may inspect those
legacy implementations without performing those experiments.

| Method ID | Actual production entry point | Store / budget |
|---|---|---|
| recompute | `mmimpress.serve.Server.recompute` | pixels every turn, full context |
| fullload | `Server.request(mode='fullload')` | canonical full K/V |
| mpic32_ssd | `mmimpress.mpic.MPICServer.request(k_recompute=32)` | `MPICContext`; all context, first 32 image rows recomputed |
| sparsevlm_ssd_kv25_allhead | `SparseVLMSSDServer.request(method_id='sparsevlm_ssd_kv25_allhead')` | `CanonicalContext(head_policy='all')`; native token Top-k |
| ours_kv25 | `Server.request_cvpr25(mode='prefix', budget=.25, budget_unit='visual_kv', sep_policy='sidecar', expected_prefix_layout='visionzip_image_only')` | image-only repacked first k rows |

Both KV25 methods retain k=(N+3)//4 real content rows at every decoder layer
and every KV head. N excludes structure/system/padding and must be positive.
All structure and logical positions remain. Ours reads first ceil(k/64) whole
chunks for K/V and masks excess rows in prefill/decode. AllHead scores the
entire original visual candidate set using all actual attention heads and
current layer hidden states, selecting exact token Top-k independently per
layer. One normal Q/K/V projection per layer is shared with scoring; decode
has no selection/read. AllHead full K is read once per layer, reused in answer
attention; selected V chunks are coalesced. Additional selected-K/probe reads
are zero. Structural sidecar duplication is actual traffic and is recorded.
Dense GPU cache is retained: no 75% GPU memory reduction claim.

## Environment, history, timing

Pinned model/processor/tokenizer: llava-hf/llava-v1.6-vicuna-7b-hf,
revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, loaded from that local snapshot.
Existing mllm_ft environment; NF4/double quantization, BF16 compute, FP16 SSD,
eager attention, existing processor-owned AnyRes resolution. No CPU/disk weight
offload. Seed=1234, batch=1, eval/inference mode, greedy, max_new_tokens=16.
Actual package/config/model/source hashes are frozen in config and manifest.

Every method and dialogue actually executes full pixels T1. T2/T3 use only that
method/dialogue's exact decoded previous predictions, including legitimate
empty answers, EOS and cap. Gold is used only for scoring. Prompt is the 73/89
causal `USER: <image>` + numbered prior QA + current question and fixed short
answer instruction. Full source logical/physical/model/method/dialogue/turn
lineage and lossless prompt/suffix IDs are retained. AllHead raters include the
entire actual suffix after the image, including history and fixed template.

Global dialogue ordinal modulo 5 gives cyclic method order, unchanged for all
turns of that dialogue. One GPU, sequential methods. Check foreign compute
processes before every timed request; never terminate other work. One fixed
existing warmup before each phase process. Diagnostics, conditioning, hashes,
logging and cleanup run between timed requests.

True TTFT starts before prompt/input preparation, includes JPEG open/decode
for pixel requests, tokenizer/processor, preprocessing/vision, SSD/pread, H2D,
cache assembly, prefill and first-token host materialization + CUDA sync. Full
request E2E includes final decoding and capture-context exit materialization.
Hit activation is separately recorded. POSIX_FADV_DONTNEED runs outside TTFT
and each actual call's success is recorded. No NAND/controller cold claim.
Metadata-ready includes immutable sys metadata and AllHead host v_hidden;
no visual K/V remains between requests. Full-K/scoring work is inside TTFT.
AllHead stage host/CUDA intervals overlap and are never summed into TTFT.

## Source capture, persistence and storage

Fresh image streaming is frozen. Source T1 is the first frozen dialogue of an
image. FullLoad captures canonical K/V+visual inputs, MPIC captures its own
canonical K/V+input sidecar, and Ours captures image-only saliency+K/V in their
normal T1. AllHead's own normal source T1 additionally captures K/V+inputs for
exact FP16 all-layer/head hash comparison against FullLoad. Only when those
bits, v_hidden, prefix IDs and image preprocessing match may they share the
immutable canonical store. These captures create no extra model/vision pass.
Other dialogues still run actual full T1 but reuse that image's stores.

Use the unchanged validated `persist_captured_raster_prefix`,
`persist_captured_mpic_prefix` and `persist_captured_visual_prefix` builders.
The existing raster builder necessarily writes a bundled three-head sidecar;
this is unused serializer overhead, not a Probe3 method. No main method opens
or reads it. AllHead is selected explicitly by head_policy=all/method_id,
never by probe=0. Preserve the builder and measure its complete bundle:
base-only AllHead persistence is NOT SEPARATELY MEASURED. Disclose unused
sidecar occupancy/write bytes. FullLoad/AllHead actual build is charged once,
to FullLoad's source dialogue. Independent AllHead deployment and standalone
per-dialogue setup are DERIVED, not additional physical measurements.

Persistence timings include existing score postprocessing/repack, write,
fsync, atomic publication. T1 capture exit materialization is already in T1
E2E and must not be added again. Record helper wall, integrity hashing and
activation separately. Physical-stream session is the three actual request
E2Es plus source-only actual setup. No background overlap is assumed.

Storage plan records current mount/device/free bytes, largest actual image
geometry and three-store bundle from existing frozen 398-image provenance,
32 GiB raw/checkpoint growth allowance, 4 GiB build/intermediate margin and
30 GiB minimum free reserve. Validation stores are separately budgeted. Recheck
before each image, preserve failed-attempt payload, never lower thresholds.
Insufficient space is BLOCKED_STORAGE. Only one main image's stores are live.
Prior artifact protection uses full SHA256 for source/small files and the
established nine-window fingerprint + inode/mtime/ctime policy for large old
files; this limitation is explicit. New payloads receive full file hashes.

Each image gets a new scratch path and creation receipt; all raw and lossless
selection/IO data are fsynced, independently audited, then an image completion
record is atomically published and its directory fsynced. Only its recorded
allowlist with matching hashes, no symlink/hardlink, may be deleted. Never
remove old/external data or raw. Resume verifies the exact source/config/
manifest/raw hash and independent completeness before skipping an image.
Uncommitted images rerun from T1 into a new attempt; prior attempts remain.
No selection by faster TTFT or higher score.

## Gates and reporting

`102_validate_llava_mt_gqa_allhead_kv25.py` first freezes configuration, full
population and four smoke dialogues (first dialogue of first four distinct
frozen images, selected before outputs; includes k%64 boundary). It reuses
unchanged AllHead G1–G12 receipt with matching source hashes, runs fresh Ours
five fixed GPU fixtures through the original 90 oracle, runs a fresh 15-request
MT integration chain through the new wrapper, AllHead independent memory
reference + sentinel + actual projection/read checks on that fresh history,
and interleaved FullLoad/MPIC/Ours/vanilla output recovery and exception cleanup.
No tolerances are widened: matched logits atol=rtol=1e-4; score 1e-5, exact
IDs/masks/KV bits. New negative audit/lifecycle/history tests and existing CPU
regressions cover G9/G11/G12. Qwen GPU remains NOT RUN. Required unresolved or
failed gate prevents smoke/main. Then a separate real 4×3×5=60 MT smoke must
pass independent audit before any of 60,915 main final requests are run.

`101_audit_llava_mt_gqa_allhead_kv25.py` independently reconstructs prompts,
lineage, strict scores, per-image and global counts, KV25 geometry, all-head
projection counts, actual OS byte ranges, zero selected-K/probe reads, timing,
Ours invariance and persistence attribution. It never imports production
selection/loader code. Tables separate turn/all/hit accuracy, T1/hit TTFT,
actual returned SSD MB, budget and stage scopes, paired comparisons, session
and provisioning policy. Image-cluster bootstrap: 10,000, seed1234, entire
image cluster across all methods/turns/dialogues, request-weighted estimates.
A zero-containing CI is not equivalence; generated history is part of the
method-level result, not a pure selector causal contrast.

Final status explicitly states dataset/integration/GPU/smoke/audit/protection,
MAIN VALID/INVALID/PARTIAL/NOT RUN with actual/expected rows, and paper readiness.
Never reuse pilot numbers or call a launched/partial process a completed main.
Failures preserve cause, all physical attempts, final counts and resume command.
No commit, push, source reset, existing artifact deletion or dependency upgrade.
