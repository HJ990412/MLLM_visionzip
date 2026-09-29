# Qwen2.5-VL correctness v2: prospective second-run addendum

This addendum is fixed **before** the second GPU validation. It supplements
`docs/qwen25_correctness_contract_v2.md` without changing its G1–G15 gates,
independent references, frozen model/workloads, exact matched-path criteria,
FP32 oracle tolerance (`atol=1e-5, rtol=1e-5`), or legacy v1 numerical
criterion (`atol=0.125, rtol=0.02` elementwise). The original prospective
`20260928T073413Z` validation remains `UNRESOLVED` and pilot-ineligible;
its result is not reclassified or used as a second-run GPU result.

## Reason for the new run

The first v2 GPU validation completed all ten fixed pairs and G1–G15 passed.
Its D1 full-prefill-versus-split diagnostic encountered one newly observed
first divergence: layer-0 pre-MRoPE Q for image `n272098`, question
`201535625`. Under the original prospective code, that branch was
`UNRESOLVED`, so no smoke or pilot ran. The first result and source have
SHA256 `1f3fcbfd0efb224e9f668a2458b27288bec49eea0b42f8b0ede0d0fc366a190b`
and `6d8ff9d819d939f58b8eeb25818f9d12a957fc8659736a74b5e910f75b920c7d`
respectively.

An isolated **post-first-run, pre-second-run** GPU control inspected this
exact pair with the same frozen model/backend. Its source is
`diagnostic_qproj_shape_201535625/shape_control.py` in the first run,
SHA256 `916f67d2910a6a13b7442ad3ac29a695ff7fef4dd86419d40b2594508e34b526`;
its result is `shape_control.json`, SHA256
`dc29204732b6cc16b03ab2e49bf630e6c950c49c281d80e4e6e980ebbac189ec`.
The full request had 505 prefix and 20 suffix rows. The suffix RMSNorm
input was bitwise identical in full and split paths. Direct 20-row NF4/BF16
Q/K/V projections differed from the corresponding full 525-row suffix
outputs; padding the *same* suffix to a 525-row projection restored all
three projection outputs bitwise. Repeated calls at each shape were bitwise
stable, and original first-token logits hashes reproduced. This directly
supports a matrix-row-shape numerical branch for that pair. It does not
establish that arbitrary later Q divergences are harmless.

## Prospective rule for second-run D1 attribution

The second validator keeps the original numerical diagnostics unchanged.
It may classify a first D1 divergence at layer-0 `q_pre_mrope` as an
**explained numerical branch only for** `n272098`/`201535625`, and only
after independently checking the above result file's SHA256, status,
request identity, identical suffix RMSNorm, repeated determinism, direct
20-row reproduction, and exact padded 525-row Q/K/V controls. The original
accepted layer-0 K/V/attention branch sites remain as in the first frozen
validator. Any other new first divergence, failed evidence check, new
position/assembly issue, non-determinism, or failed G1–G15 gate is
`UNRESOLVED` or `FAIL` according to the original contract and blocks smoke
and pilots. No observed logit error changes a tolerance.

The same ten validation pairs, 20-image smoke set, original frozen 40-image
GQA and 40-dialogue MT workloads, production P2 SSD path, model/config,
and order `validation → smoke → GQA → MT → analysis` remain fixed. The
second-run `frozen_inputs.json` must hash this addendum, both new scripts,
all original code/config/manifests, shape evidence, first-run JSON, and the
new manifests before second GPU execution. Previous source/data/store/result
artifacts are protected and never overwritten.
