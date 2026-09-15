# ImageOnly-Repack sequential-Prefix budget sweep

## A. Workload

This is the frozen GQA workload: **40 images / 240 questions**, questions
`[4:10]` for every image.  Index SHA256 is `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a` and ordered
workload SHA256 is `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`.  All arms ran in one process with the
same model, prompt, decoding, physical SSD store, and schema-v2 true-TTFT
contract.

## B. Budgets

Main budgets are 20/25/30/35/40/45/50%.  Every budget uses the same immutable
ImageOnly VisionZip permutation; only `k = round(n_chunks * budget)` changes.
No calibration question, online score, diversity, or budget-specific layout is
used.

The Prefix reader was already optimized before this sweep: adjacent first-k
chunks are merged into one contiguous span per layer/K-or-V file.  Therefore
every Prefix request performs 64 normal preads plus one separator-sidecar
pread, not one syscall per logical chunk.

## C. Main table

| Budget | Accuracy | Delta vs FullLoad | TTFT | TTFT reduction | SSD MB | SSD reduction | Preads |
|---:|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 61.25% | -- | 727.37 ms | -- | 1165.07 | -- | 64.0 |
| 20% | 54.58% | -6.67 pp | 233.78 ms | 67.86% | 252.39 | 78.34% | 65.0 |
| 25% | 57.92% | -3.33 pp | 265.17 ms | 63.54% | 306.08 | 73.73% | 65.0 |
| 30% | 58.75% | -2.50 pp | 300.06 ms | 58.75% | 371.51 | 68.11% | 65.0 |
| 35% | 59.17% | -2.08 pp | 336.78 ms | 53.70% | 435.26 | 62.64% | 65.0 |
| 40% | 59.17% | -2.08 pp | 375.32 ms | 48.40% | 498.18 | 57.24% | 65.0 |
| 45% | 60.42% | -0.83 pp | 404.78 ms | 44.35% | 551.03 | 52.70% | 65.0 |
| 50% | 60.00% | -1.25 pp | 435.31 ms | 40.15% | 603.87 | 48.17% | 65.0 |

Accuracy is binary normalized GQA match.  `summary.csv` also reports the
image-cluster bootstrap 95% CI for each accuracy and each paired FullLoad
delta.  With only 40 image clusters, sub-point differences should not be
over-interpreted.

## D. Pareto frontier

- Accuracy vs SSD: FullLoad, 20%, 25%, 30%, 35%, 45%
- Accuracy vs true TTFT: FullLoad, 20%, 25%, 30%, 35%, 45%

Dominated points and their dominators are explicit in the two Pareto CSVs.
Pareto membership uses observed point estimates; uncertainty remains in the
paired confidence intervals.

## E. Accuracy recovery

- Aggressive (loss <=4 pp): 25%
- Balanced (loss <=2 pp): 45%
- Quality-oriented (loss <=1 pp): 45%
- Saturation by the predeclared <=0.5 pp best-future-gain rule: 45%

## F. SSD/TTFT cost

| Step | Accuracy gain | SSD cost | TTFT cost |
|---:|---:|---:|---:|
| 20% -> 25% | +3.33 pp | +53.69 MB | +31.39 ms |
| 25% -> 30% | +0.83 pp | +65.43 MB | +34.89 ms |
| 30% -> 35% | +0.42 pp | +63.75 MB | +36.72 ms |
| 35% -> 40% | +0.00 pp | +62.91 MB | +38.54 ms |
| 40% -> 45% | +1.25 pp | +52.85 MB | +29.46 ms |
| 45% -> 50% | -0.42 pp | +52.85 MB | +30.52 ms |

Doubling 25% to 50% changes accuracy by +2.08 pp,
adds 297.80 MB/request, and adds
170.14 ms mean true TTFT.

## G. Importance coverage

| Budget | VisionZip mass | SparseVLM mass (analysis only) | Accuracy |
|---:|---:|---:|---:|
| 20% | 68.01% | 35.82% | 54.58% |
| 25% | 73.15% | 41.60% | 57.92% |
| 30% | 78.41% | 47.98% | 58.75% |
| 35% | 82.65% | 54.22% | 59.17% |
| 40% | 86.19% | 59.79% | 59.17% |
| 45% | 88.68% | 63.82% | 60.42% |
| 50% | 90.95% | 68.10% | 60.00% |

VisionZip coverage/accuracy correlation: Pearson
`0.9096593029641282` and Spearman
`0.9549937104572924`.  SparseVLM
coverage is explicitly analysis-only and never affected serving.

## H. Error analysis

`per_question_sensitivity.csv`, `per_image_sensitivity.csv`, and
`error_analysis.csv` separate 25%-wrong questions recovered at 30/35%, those
never recovered through 50%, prediction stabilization, and the existing GQA
question categories (yes/no, color, count, spatial, material/attribute,
object, other).

Restricting the attribution to questions that FullLoad answers correctly,
25% loses 19 questions; 7 recover by 30/35%, while
6 never recover at any tested budget through 50%.

## I. Recommended operating point

**Recommended operating point = 45%**.
Rule: minimum budget with point-estimate FullLoad loss <= 2 pp.  At this point the FullLoad
accuracy gap is -0.83 pp, SSD
traffic is 47.30% of FullLoad,
and true TTFT is reduced by
44.35%.

This recommendation is based on the observed point estimate.  Its paired
image-cluster bootstrap delta CI is
`[-3.75,
+2.08] pp`; therefore
this sample does **not** establish that the population accuracy loss is at
most 1 or 2 pp.

## J. Research conclusion

**Claim assessment: SUPPORTED, with non-monotonic accuracy and small-sample caveats.**  Every tested Prefix budget reduces observed
SSD traffic and true TTFT versus same-run FullLoad, and higher budgets recover
some accuracy.  The curve is not monotonic: 45% is more accurate than 50% on
this 240-question sample.  No budget was selected in advance.

Direct answers: Q1=25%,
Q2=45%,
Q3=45%,
Q4=45%,
Q6=45%. Q5 is quantified in
Section F; Q7 is quantified by the coverage correlations in Section G.

Known historical warning retained: raster FullLoad vs ImageOnly-repacked
FullLoad strict prediction identity was
`238/240`, so cross-layout exactness
remains FAIL even though mapped FP16 KV structural integrity passed.  This does
not invalidate the present same-layout budget comparison, but it must not be
reported as strict raster equivalence.

## Reproduction

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/04_eval.py --index data/index.json --store kvstore_image_only_visionzip --limit 40 --questions 6 --skip 4 --ratio 0.25 --budgets 0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selectors visionzip_repack_prefix --prefix-layout visionzip_image_only --sep-policy sidecar --metric gqa --dataset gqa --expect-images 40 --expect-questions 240 --no-recompute --run-dir runs/image_only_repack_budget_sweep/main_20_50
```
