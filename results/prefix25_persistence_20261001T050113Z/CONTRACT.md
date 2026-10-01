# Prefix25 persistence contract

Frozen before GPU execution. A = original importance-repacked FullStore KV25;
B = same full CPU materialization, same full stable permutation and repack,
then serialize content rows [0, ceil(N/4)) only. C=64, short final chunk,
no real extra rows, no padding in B. Structural/system KV stays separate.
Original N, positions, geometry and full permutation stay distinct from stored k.
B rejects chunk budget and any ratio >.25. No pruning, merging or new scoring.
LLaVA full-size physical buffer and BF16 restoration from FP16 stay unchanged;
Qwen native BF16 logical compact assembly and original MRoPE stay unchanged.

CPU: N=1,63,64,65,127,128,129,255,256,257,349,2200; zero rejected;
independent stable-rank/gather; all layer/head KV bits, structural, short EOF,
invalid identity/mapping/checksum/truncation, double-budget and oversized requests.
Independent real os.pread returned byte/call traces must equal production counters.
GPU: five original LLaVA pairs and ten original Qwen v2 pairs, reused validation
samples (not unseen holdout). Same full-image capture for independent A/B stores.
Independent score rank and canonical-KV gather; content k, structural, original
positions and actual masks from prefill through last decode; first/generated IDs
and prediction exact. LLaVA every generated-step logits atol=1e-4, rtol=1e-4;
Qwen matched first logits bitwise exact plus all generated IDs exact, stock MRoPE
and masks, FP32 dense/compact GQA oracle atol=rtol=1e-5. No relaxation after results.
Release source/capture/reference KV, clear serving contexts, activate B alone and
require fresh independent B payload reads, zero vision/query-scoring, deterministic
repeat, history invariance and interleaved image/method isolation. Any required
FAIL or unexecuted gate blocks that model's performance experiments independently.

Each performance arm executes its own full-image T1 and own physical persistence.
No shared timing/capture/store. ReComp none; FullLoad canonical; A full; B prefix.
Full integrity hashes and file/directory/parent fsync inside both A/B total
persistence. Existing helper components may overlap; total uses a wall boundary.
/proc/self/io wchar/syscw deltas measure OS successful write returned bytes/calls,
not NAND traffic. Integrity envelope, system, metadata and padding counted.
LLaVA saliency hooks run in forward; Qwen saliency computation occurs in VisionScoreCapture.__exit__ after normal
request generation; outer request E2E includes this and required capture cloning
exactly once, before persistence. normal_request_e2e_ms records the inner boundary.
Full CPU copy/repack costs are retained. Serialization timing separate from write.
Service session = T1 E2E + persistence + one activation + T2 E2E + T3 E2E.
GQA sum is six independent requests + one-time persistence/activation, not MT.
Page-cache DONTNEED outside request timers for all stored arms; no NAND cold claim.
Activation integrity reads are separate and included once in session. Hits do not
decode image files. Independent tracing overhead is present symmetrically on hits.
Warmup excluded; methods rotate by image and turn. Never measure with another GPU
compute process. Seed=1234, greedy max16, unchanged model revisions/backend/processor.
Smoke 4x3x4; GQA frozen40x6x4; MT frozen40x3x4, own generated history per arm.
No 4061-dialogue full run. Image-cluster paired bootstrap 10000 seed1234 95% CI.
MB=1e6 bytes, disk safety>=30 GiB. Keep raw/failures/inventories; delete only this
run's stores after all image measurements and hash/mapping receipts committed.
Cleaned payloads require regeneration. No old artifacts changed and no push.
