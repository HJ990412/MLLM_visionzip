# G7 review: prospective Qwen PrefixStore3 v2

## Finding and evidence
Classify the reproduced failure as case ②: numerical validation methodology,
with no observed production semantic/implementation mismatch. This conclusion
uses exact matched references and controlled fixed-input tests, not error size.

The latest B source generated method-local T1 history for n313060,
mtgqa_000285/T2. It yielded N=529,k=133,S=21,prefix154,suffix60. Layer0 Q,
suffix K/V and selected prefix K/V match the original diagnostic tensors by
full byte hashes. Legacy FP32 dense/compact: 16/215040 violations,
max absolute difference 2.162158489227295e-5 at [batch0,head6,query51,dim5].
This is the same original FAIL, preserved in old runs and fresh diagnosis_v2.

Actual production SSD B and independently ranked/gathered canonical BF16 KV
match all layers/native heads, original logical IDs, stock MRoPE, causal mask,
first logits and first/all generated IDs. An actual production SDPA observer
captures post-MRoPE Q/K/V and masks for all 28 layers. Same-shape memory and
repeat agree. Original K is never rotated again. Three fixed-input CPU oracle
repeats are deterministic. Backend remains native BF16 SDPA in inference.

Fixed-input stage analysis isolates FP32 shape effects. Actual FP32 reference
matmul/softmax/aggregation run on CPU with autocast disabled, not CUDA/TF32.
Only dense versus compact key-axis shape changes. Visible QK dot products
already differ. At the failing head6/query51, maximum visible dot difference
is 0.000732421875 and probability difference is 1.8656253814697266e-5.
Using dense logits in compact softmax/aggregation reduces the worst-output
residual from 2.162158489227295e-5 to 3.725290298461914e-8; global residual is
8.344650268554688e-7. Separate identical-logit padding and identical-probability
padding controls isolate smaller softmax/reduction-shape contributions.
Thus the main observed error originates at shape-dependent FP32 QK rounding,
with additional reduction effects. We do not claim a specific undocumented
BLAS microkernel. Numerical environment/strides/dtypes are in fixed_inputs.pt.

Each FP32 path also differs from its FP64 reference: maxima approximately
3.4425895e-5 (compact) and 3.4440796e-5 (dense). Neither FP32 result is treated
as exact ground truth. FP64 inputs are converted before QK, scale, softmax and
V aggregation. Dense and compact FP64 outputs meet fixed 1e-12/1e-12 over every
head/query/output element in all 28 layers; maximum across the failing T2
request is 2.4868995751603507e-14. FP64 is finite precision, not exact arithmetic.

A separate already-known dialogue mtgqa_000001 supports the same explanation:
T3 legacy FP32 fails at 1.3597309589385986e-5 while all-layer FP64 passes (max
2.842170943040401e-14), matched production is exact and negatives are rejected.
These diagnosis samples are NOT called new holdouts.

## Change and admission
No existing production runner/store/vision or old validator source is changed.
New validator/experiment adapter separates exact implementation correctness
from high-precision attention semantics. Legacy FP32 criterion stays unchanged
and is recorded alongside revised gates. No tolerance is tuned to the maxima,
no failure sample is dropped, and no assertion is silently disabled.

The first new diagnostic attempt failed a metadata scalar assertion: Python
1/sqrt(128) versus installed Qwen 128**-0.5 differ by one float ULP. The code now
checks the actual installed scalar and uses it in FP64 semantic calculations;
legacy FP32 keeps its original division-by-sqrt computation. diagnosis_v1,
its frozen source, failure and tensor evidence remain. No additional sample
was evaluated when this correction was made. See development_attempts.jsonl.

The final contract/source are frozen before original-ten core, full integration
and the prospective eight additional dialogues. Additional IDs/rule were fixed
before outcomes in additional_validation_manifest.json; prior executed Qwen
image IDs were excluded. No criteria may be adjusted from their outcomes.
Main admission requires all CPU/core/exact/semantic/integration/additional and
negative gates, then actual 36-request smoke. G7_review_verdict.json is emitted
only after those required validation results; this review alone cannot admit main.

Negative controls include same-shape wrong KV/selected IDs, missing/duplicate
structure, wrong positions, future visibility (demonstrably changed attention),
foreign image, original-vs-stored double-budget, truncated/corrupted/native-unsealed
store, and method/dialogue/state leakage. Qwen native meta.json is its durable
publication marker; the experiment completion.json is a separate seal receipt.

## Immutable evidence
Original blocked runs: qwen_mt_gqa_kv25_main_20261001T031459Z and
mt_gqa_B_llava5_qwen3_20261001T093921Z/qwen. Fresh diagnostic trials:
diagnosis_v1 (metadata assertion FAIL) and diagnosis_v2 (implementation/semantic
PASS; legacy G7 FAIL preserved). G7_stage_attribution.json replays saved tensors.
Production main precision/backend is never changed to fit the references.
