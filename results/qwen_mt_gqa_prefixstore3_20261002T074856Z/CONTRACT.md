# Qwen PrefixStore3 semantic validation contract v2

This is a Qwen-only prospective experiment. REQUEST.md is normative. Prior
LLaVA 60,915 main requests and all prior Qwen FAIL evidence are immutable.
No existing source changes are authorized in this version. New 116/117/118
and CPU tests add diagnostics, admission, and reporting; 110/111 lifecycle and
qwen25 runner/store/vision perform inference unchanged. No LLaVA GPU execution.

## Workload and production
MT-GQA-reconstructed: 398 images, 4,061 dialogues, 3 turns, 12,183 requests
per method; total 36,549. Original file and ordered workload hashes are checked.
Methods: recompute, fullload, qwen_ours_prefixstore_kv25. All T1 requests are
full-image; only first frozen source dialogue/image captures for store creation.
Each method uses its own generated history, raw decoded predictions, no gold.
Original canonical ordinal three-way rotation and strict normalized EM remain.
Qwen pinned cc594898137f460bfe9f0759e9844b3ce807cfb5, NF4 double/BF16/SDPA,
native 28 layers x 4 KV heads x 128; min/max pixels 200704/802816; seed1234,
batch1, greedy max16. No backend, precision, resolution or algorithm changes.
B full capture/materialization/stable repack, direct first k=(N+3)//4 content
serialization; structural/system/boundary separate; no padding or truncate.
Hit reads all stored k against ORIGINAL N, preserves original post-MRoPE keys
and logical positions with P2 compact assembly; no query scoring or hit vision.

## Legacy and revised validation, fixed before additional outcomes
LEGACY G7 remains the original layer0 FP32 dense-vs-compact elementwise
atol=rtol=1e-5 diagnostic. Historical FAIL stays FAIL. Its threshold is unchanged.
Prospective semantic/implementation admission is conditional on documented
root-cause evidence, not on small errors or main quality. FP64 is finite-precision
reference computation, never claimed exact arithmetic and never used in main.
Implementation gate: independently stable-rank canonical captured BF16 KV,
all 28 layers/native heads exact bits, selected/structural/logical IDs exact,
original stock MRoPE positions and independently enumerated causal visibility
exact. Actual SSD production, independent same-shape memory, and repeat require
bitwise first logits and exact first/all generated token IDs. Actual production
SDPA Q/K/V/masks from all 28 prefill layers must match same-shape reference.
Semantic gate: actual fixed production Q and suffix K/V plus independent
canonical prefix form dense logical and compact references. Convert inputs to
FP64 before QK matmul, scale, softmax and value aggregation, autocast disabled.
Atol=rtol=1e-12 on every attention output element across all heads, suffix queries,
head dimensions and all 28 layers. Dense masked unselected positions have zero
probability; original cached K is already post-MRoPE and is never rotated again.
FP32 stage diagnostics separately compare QK dots, scaling, softmax, aggregation,
and each path against FP64. Same-input shape-isolation controls and three
fixed-input repetitions identify divergence. Save tensors, stride/dtype/device,
selection, histories, masks, positions, precision flags and full SHA256.
Original old diagnostic input tensor hashes are compared when available.

Validation sets are distinct: original 10 frozen core pairs; known G7 dialogue
mtgqa_000285 with actual own generated history plus existing integration; and
8 additional dialogues fixed in additional_validation_manifest.json before any
outcomes. Prior observed Qwen images are excluded, N quartiles and k%64 boundary/
question-token-length stratification use seed1234 hash tie-breaks. Actual future
generated-history length is unavailable beforehand. They are additional Qwen
validation, not a quality holdout or guaranteed globally unseen data.
Negative fixtures reject wrong KV/IDs, missing/duplicated structural rows,
wrong MRoPE/sequence position, future suffix visibility, foreign image/store,
double budget, truncated/unsealed/corrupted payload and history/state leakage.
Same-shape faults are included and causal mutation must change attention output.
Final CPU/core/generated-history/additional/negative gates must all PASS under
frozen source; then actual 36-request main-runner smoke and complete main.
Any semantic/implementation failure blocks main. Corrections get retained attempt
history and new freeze; no sample dropping, tolerance tuning or copied PASS.

## Timing and accounting
TTFT begins before history/chat/input and image open/decode for pixel paths;
ends at first token materialization/sync. Hit includes actual pread/H2D/compact
assembly/suffix prefill. Capture score/clone exit is in outer source T1 E2E once.
Persistence measures durable publication/seal, components separately; activation
measured once/image/method; no overlapping component sums as critical path.
Both cached methods use DONTNEED outside request after activation/integrity.
OS bytes are not NAND bytes, H2D bytes are tensor bytes, missing values NOT_MEASURED.
No heavy auditing or other GPU process during timed requests. Metadata/FD residency
and native GPU peak scope disclosed. Expected 796 builds and 796 activations.
Actual shared-image session uses each physical setup once at source dialogue.
Standalone-equivalent reallocates measured image setup once/dialogue and is
DERIVED; non-source capture increment is NOT_MEASURED, not filled with zero.
Main/source captures are released before later requests; no in-memory full KV
substitution for SSD reads. Validation captures are diagnostic only.

## Audit and protection
Image-major complete execution, independent raw/I/O/store audit, full hashes,
atomic first-success commit, then owned exact allowlist cleanup. Failed attempts
remain unadopted and retained. 30 GiB reserve plus raw/validation/failure budget.
No prior artifact deletion, remote push, Git reset/clean/pull, dependency upgrades,
or termination of other GPU processes. Historical large-file fingerprints are
sampled and explicitly labeled; new evidence uses full SHA256.
Independent raw strict rescoring and image-cluster paired bootstrap 10000 seed1234
use pooled request sums/counts. Turn/all/followup and mean/p50/p95 reported;
accuracy pp, MB=1e6, GiB=2**30, ms. Legacy and revised statuses stay separate.
