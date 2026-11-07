# MT-GQA-reconstructed generated-history B main, version 1

This prospective experiment replaces only the prior full-store Ours arm with the
final validated prefix25 store. Prior run receipts and failed evidence remain
immutable. The complete user specification is archived as REQUEST.md in the run
and is normative. Execution receipts, not this contract, establish PASS.

## Population and registry

The original dialogue file SHA256 is
2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224;
ordered workload SHA256 is
0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62.
398 images, 4,061 dialogues, three turns, 12,183 requests/method, 8,122
followups/method. LLaVA methods: recompute, fullload, mpic32_ssd,
sparsevlm_ssd_kv25_allhead, ours_prefixstore_kv25 (60,915 main requests).
Qwen: recompute, fullload, qwen_ours_prefixstore_kv25 (36,549).
No A, Probe3, ReKV or legacy Ours main arm. AllHead and MPIC are not ported to Qwen.
Models run sequentially. Qwen can be BLOCKED independently of LLaVA.

108/109 retain 100/101 lifecycle, prompt, strict scorer, global ordinal rotation
and production entries. 110/111 retain 103/104 equivalents. Each dialogue uses
its original ordinal rotation throughout all three turns. Image-major execution
retains canonical membership. Actual full pixels T1 occurs for every method and
dialogue; captures are enabled only for first source dialogue T1 of each image.
Followups use exact method-local decoded history, including empty/EOS/cap answers.
Gold is never supplied to inference. Normalize only for strict exact-match scoring
against the unchanged first gold answer. Raw tokens, prompts, IDs and lineage persist.

## Frozen production semantics

LLaVA: llava-hf/llava-v1.6-vicuna-7b-hf,
c916e6cdcd760b4cecd1dd4907f84ac649f93b23, existing slow AnyRes processor,
NF4 double quant, BF16 compute, eager attention, FP16 SSD, dense GPU buffer.
Qwen: Qwen/Qwen2.5-VL-7B-Instruct,
cc594898137f460bfe9f0759e9844b3ce807cfb5, min_pixels200704/max_pixels802816,
NF4 double quant, BF16 native KV, SDPA, 28 layers/4 KV heads/head_dim128,
original full logical three-axis MRoPE with compact P2 cache. Seed1234,
greedy max16, batch1, eval/inference, OMP/MKL/Torch CPU threads2. No offload.

AllHead uses explicit head_policy=all and production method ID. Actual suffix
hidden-state Q at every layer, original full causal softmax denominator,
all attention heads/rater reduction/stable ties; exact ceil(N/4) content IDs,
all native KV heads, structure outside budget. One Q/K/V projection per layer
is shared with scoring. Full K is read once, reused by answer attention;
selected V covering chunks are coalesced, additional selected K/probe reads zero.
Decode performs no selection/read. This is a fixed-budget SSD adaptation,
not progressive-pruning/recycling SparseVLM reproduction.
MPIC production MPICServer/MPICContext recomputes the first 32 image rows,
retains full context, with actual sidecar I/O/recompute/H2D inside hit timing.

B preserves full capture, importance algorithm, full CPU materialization and
full repack; only serialization writes first k=ceil(original N/4) content rows.
It keeps all structure, original geometry and complete rank, exact k short EOF,
no padding and explicit partial mapping. No pre-pruning/gather optimization.
Selection is query independent across all hits. Illegal larger budgets fail.

## Durability, provisioning and timing

LLaVA builds three stores/image: one canonical FullLoad bundle, one MPIC, one B.
AllHead executes its own source T1 and may share canonical only after exact
all-layer/head FP16 KV, hidden input, prefix IDs and preprocess identity checks.
Unused three-head serializer sidecars remain in canonical occupancy/write costs.
There is no Probe3 serving arm. Qwen builds independent FullLoad and B stores.
Normal main build counts are 1,194 and 796; retries/validation/smoke separate.

LLaVA native durable writers retain their fsync/no-clobber/parent publication.
Their optional second full hash scan is replaced by one common completion hash
scan, immutable manifest and fsync. B and FullLoad use integrity.json; MPIC
keeps its native integrity.json and adds completion.json. MPIC's native sample
hash remains measured; it is not relabeled as a full hash. Qwen reuses native
writer hashes of every serialized payload, hashes metadata, publishes/fsyncs
completion.json and parent without unnecessary payload rehashing.

FullLoad/B ImageContext performs its completion scan on activation. MPIC adds
one full completion check to its native validation. AllHead retains native
required-file checks and verifies unused bundle probe sidecars separately;
its second context activation is a real separate event despite shared storage.
Qwen retains native full-payload activation verification. Service checks are
separate from out-of-band pre-cleanup audit rehashes. Full build/activation event
IDs, wall times, nested stages, logical/allocated bytes and /proc/self/io
wchar/syscw deltas are retained. OS writes are not NAND writes.

TTFT begins before prompt/input preparation and per-request image open/decode
for pixels, ends after first token materialization/synchronization. Hits include
pread/H2D/assembly/query scoring/recompute. Full request E2E includes all required
source capture exit. In Qwen source T1, post-generation score/clone tail is
included exactly once in outer E2E and never added again to persistence setup.
Non-source T1 capture overhead is NOT_MEASURED. Existing lightweight observers
remain common across applicable arms; diagnostic logits are excluded from main.
DONTNEED is after activation, outside request, success logged; it is an OS hint,
not a cold NAND guarantee. Foreign GPU work aborts timing; never terminate it.

Physical shared-stream session = measured request E2Es + actual once-attributed
build/activation events; canonical build is assigned FullLoad only. AllHead's
zero build attribution does not imply independent deployment is free.
Standalone-equivalent = actual shared-store requests + measured per-image
required setup reattributed to each dialogue. AllHead setup is
MEASURED_SHARED/DERIVED_FROM_SHARED including unused probe bytes. It is not
independent fresh persistence or a complete measured cold-new-image session.
Request-weighted and image-macro estimates are distinguished.

## Gates and lifecycle

New source/config/contract/workload freeze before GPU. Existing CPU suite and
new integration/lifecycle/short-file tests. Fresh five fixed LLaVA B/A controls
at original 1e-4 matched tolerance; ten AllHead-only original validator samples,
new interleaved main integration and same-image multiple-dialogue fixture.
Fresh Qwen ten fixed B/A native bits/exact logits/IDs/MRoPE/decode controls,
then real generated-history integration including n313060/mtgqa_000285/T2,
FP32 dense/compact atol=rtol=1e-5. Previous v1 and prior main G7 failure are
separate evidence. No FP64 waiver, omission or relaxed threshold. Unresolved
mandatory gate blocks that model's smoke/main.
Actual runner smoke uses the prior fixed manifests (LLaVA first four images;
Qwen min/max N, longest question sequence, earliest boundary image):
60 LLaVA and 36 Qwen requests, distinct from full main. CPU/GPU reference calls,
fault injections, retries and failed attempts never enter final main counts.

At least 30 GiB free reserve; fresh mount/geometry/storage plan records raw,
validation and incomplete-attempt allowances. One main image live at a time.
Each complete image: flush raw/events, full file hashes/mapping, independent
raw audit, atomic evidence-bound commit, close contexts, exact allowlisted
new scratch cleanup. Symlinks, hardlinks and outside paths fail. Existing
artifacts are never deleted. Partial image reruns wholly into a new attempt;
all failures persist. Same source/config/manifest required for resume; never
choose faster/more accurate trials. Payload regeneration is required after
cleanup. This bounded streaming policy is not a hit-rate/saturation experiment.

Old source/small artifacts receive full SHA. Old files above 8 MiB receive nine
64 KiB framed SHA windows plus stable inode/size/mtime/ctime, explicitly not full
byte protection. All new raw/manifests/payload evidence use full SHA. No existing
source modifications, dependency upgrades, destructive git commands or push.

## Independent analysis

Rescore raw strict MT exact match, T1/T2/T3/all/followup; compare B to every
same-model baseline using matched requests, agreement and four correctness
cells. Method-history divergence is part of method-level results. Independently
reconstruct every OS range, exact selection/mask budgets, B invariance, AllHead
layer selection T2/T3 Jaccard/V chunk coverage, history and source ownership.
10,000 image-cluster paired bootstrap resamples, seed1234, preserve all requests
and methods within image and use pooled sums/counts for request-weighted means.
Report mean/p50/p95 separately; CI containing zero is not equivalence. All
storage, setup and session tables label MEASURED/SHARED/DERIVED/NOT_MEASURED.
MB=1e6 bytes, GiB=2^30, ms, accuracy differences in percentage points.
