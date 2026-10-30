# MT-VQA-v2-reconstructed-25k prospective contract

Normative request: runs/mt_vqa_v2_reconstructed_25k_20261021T091600Z/REQUEST.md.
This is a new reconstruction of independent VQAv2 validation questions, grouped
by one COCO val2014 image into three sequential requests. It is not the authors'
released subset, question order, or exact MetaCompress protocol. The 250-image
development workload and all previous MT-GQA results remain immutable.

## Frozen population and metric

25,000 distinct images, one dialogue/image, three distinct question IDs/dialogue,
75,000 questions/method. Original questions, image/question IDs, ten answer
dictionaries including answer_id/confidence, answer_type/question_type and full
annotation survive in the manifest. CPython random.Random(1234) samples 25,000
images from canonical numeric image-ID order, then three sorted question IDs
without replacement per image using the same RNG in sampled image order. This
order is the dialogue/turn order. No first-question exclusion, four-question
requirement, text deduplication, output-dependent selection or rewriting.
Python/random source hashes, all source archives, per-image hashes/decode status,
ordered full workload hash and original/selected statistics are retained.
Overlap is disclosed; no globally unseen holdout claim.

Primary: official GT-Vision-Lab/VQA revision
a013f0043c1e2cdc995922dfe257f7149aa9af06, leave-one-out over all ten annotations.
Only two Python2 print statements are ported. Original blob/hash and exact diff
are in mmimpress/mtvqa25k/vendor. Preserve the pinned evaluator's conditional
normalization: punctuation/digit/article/contraction processing is skipped when
all whitespace-cleaned GT answers agree. Case variants can therefore score zero
on unanimous GT under this revision. This is explicitly documented behavior of
the pinned official reference, not unconditional normalized EM. Fixtures and
actual raw compare with an independently syntax-translated upstream evaluate
loop and independent count-based leave-one-out calculation. Primary tables say
VQA score (%). Legacy min(matches/3,1), raw/normalized EM are secondary diagnostics.

Preserve prompt IDs, generated-only IDs, literal native decoded output, native
answer, scorer answer and scorer normalization. LLaVA/Qwen use their validated
skip-special-token decoding with cleanup=false and strip. GLM uses its native
closed answer-box span; malformed output stays literal. No word/sentence picking.
History uses method-local native generated answer (GLM literal body after one
terminal native EOS removal), never normalized scoring text or gold. Empty,
EOS-only and capped outputs remain in the denominator and are not retried for
quality. T1 predictions and first tokens must agree across same-model arms.

## Production and validation

LLaVA NeXT Vicuna7B c916e6cdcd760b4cecd1dd4907f84ac649f93b23:
ReComp, FullLoad, MPIC-32 SSD, SparseVLM-SSD-KV25-AllHead, PrefixStore-KV25 [B].
NF4 double quant/BF16 compute/eager/FP16 SSD, existing slow AnyRes.
Qwen2.5-VL7B cc594898137f460bfe9f0759e9844b3ce807cfb5: ReComp,
FullLoad, qwen_ours_prefixstore_kv25. NF4 double/BF16 native KV/SDPA,
min/max pixels200704/802816; 28 layers/4 native KV heads/128 head dimension.
GLM4.6V Flash 411bb4d77144a3f03accbf4b780f5acb8b7cde4e: ReComp,
FullLoad, validated v6 B. Decoder NF4 double/BF16 vision+merger+lm_head/native
KV/SDPA; 40 layers/2 KV heads/128; native size12544/1605632, thinking=false.
Seed1234/batch1/greedy/CPU Torch OMP MKL threads2. LLaVA/Qwen cap16; GLM cap32
preserves the successful v6 contract (v3 cap16 truncated a native close boundary).
No weight offload, resolution/precision/backend change or full-KV fallback.

Native production entries are 108/110 plus qwen25 modules, and GLM v6 121 plus
glm46v modules. Current Oct13 FullLoad request identity validation/single-token
prefill fix and its 108 call-site are retained, not reverted to Oct1 source.
New adapter changes experiment metadata, official scoring, archival and Qwen's
E2E endpoint to the same outer runner return for all requests. Qwen activation
also measures the complete _activate caller boundary including expected-identity
preparation; original inner time remains a nested diagnostic. Algorithms remain
unchanged. All current source hashes and differences from historical freezes
are recorded. Existing source files are not edited by this experiment.

B: full normal T1/capture/image-only importance/full materialize/full repack;
directly serialize first ceil(original N/4) content rows plus all structure.
Exact short EOF, no B padding, complete mapping/provenance. T2/T3 frozen first-k
pread, no query scoring, no hit vision, no decode reads/selection. Native
post-RoPE K/logical positions/masks are preserved. MPIC recomputes32 rows with
full context. AllHead uses all heads and actual suffix/layer Q; full scoring K
read once reused in attention, covering V chunks. Its attention25% is not IO25%.
FullLoad/AllHead share only after native all-layer/head prefix identity checks.
Independent AllHead deployment receives shared persistence cost, including
unused serializer sidecars, in a separately labeled derived table.

Prospective validation uses per-model min/max N, k%64 edges, longest tokenized
question and question-history proxy, portrait/landscape extremes. Same union
plus fixed seed-hash fill defines 40-image pilot before any generation.
CPU/source/scorer/dataset tests precede fresh GPU native-bit/position/mask,
same-shape logits+IDs, semantic references, actual production smoke and negative
fixtures. LLaVA logits1e-4/1e-4; Qwen/GLM native same-shape exact; FP64 semantic
reference1e-12/1e-12. Prior FP32 shape-rounding FAIL stays FAIL at1e-5/1e-5 and
is a separate diagnostic under the successful latest semantic contracts. FP64
is never inference precision. Unresolved semantics block that model only.

## Execution, timing, storage and analysis

GPU models execute sequentially. Within model, phase canonical ordinal modulo
arm-count rotation is frozen across three turns. All T1s run pixels. Fresh own
source T1 creates stores once/image. Phases: correctness, actual smoke, old250
pilot (3750/2250/2250), new40 pilot (600/360/360), main375000/225000/225000.
No pilot/reference/warmup/failed request enters825000 final target requests.

TTFT starts before history/input/image decode; first native token materialized
and synchronized ends it. E2E runs through the model-specific outer return,
including required source capture/importance tail once. Qwen source and other
requests now all use core_returned_at_s. Persistence/activation are outer caller
walls; nested components and source capture are never added again. Session is
three E2Es plus once/image/method persistence and activation, no extra-dialogue
amortization. Shared physical events and derived standalone-equivalent are
separate. Report physical attribution conservation. MB1e6, GiB2^30, ms, pp;
unobserved fields NOT_MEASURED, OS bytes not NAND, DONTNEED not cold-NAND proof.

One image live: build, all turns/arms, independent raw/store/actual-range audit,
full hashes, lossless journal gzip with decompression SHA check, atomic first
success commit, close contexts, exact owned allowlist payload cleanup. Gzip
replaces only verified redundant uncompressed journals after commit; raw bytes
are preserved. Partial image uses a fresh attempt; failures retained. No faster
or more accurate attempt selection. Resume requires unchanged full source,
config, dataset, environment and model freezes. Foreign GPU work blocks timing
without termination. Minimum30GiB reserve plus working set is enforced; main
also needs measured pilot raw-growth/IO/wall-time admission. Existing files and
validation artifacts are never removed. This is bounded image-major storage,
not saturation/hit-rate/eviction/parallel-throughput measurement.

Independent analysis streams committed raw, rechecks official scores, IDs,
history, selection, original-N budgets, real IO and setup conservation. Report
T1/T2/T3/All/Followup score and mean/p50/p95 latency, IO, storage/setup/session.
Compare B with every same-model baseline; image-cluster paired bootstrap10000,
seed1234, pooled sums/counts in bounded batches, preserving all arms/turns per
image. History divergence belongs to method-level effects. No cross-model
pooled effect, equivalence from CI crossing0, hidden accuracy/setup loss, or
claim of GPU memory75% reduction from KV25%. Only terminal audited counts can
establish COMPLETE. Background start never establishes completion.
