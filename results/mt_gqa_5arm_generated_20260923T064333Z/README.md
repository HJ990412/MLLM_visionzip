# MT-GQA generated-history five-arm results

Validated same-run comparison over 398 images, 4,061 three-turn dialogues,
and 60,915 requests. See [ANALYSIS.md](ANALYSIS.md) for the main table,
paired quality, latency, I/O, persistence, and limitations. The machine-readable
headline table is [summary.csv](summary.csv); validation gates are in
[validation.json](validation.json) and [report_validation.json](report_validation.json).

The 3.6 GB full raw JSONL and generated KV stores remain in the local
`runs/mt_gqa_5arm_generated_20260923T064333Z/` directory. The local result
directory also retains a copy of the raw JSONL and eight lossless gzip parts.
The original raw SHA-256 is
`6e98b3127f5a3d0bfcefe8f63690988c34c945f686a1339168cf305852147673`.
The published commit contains the derived tables and validation receipts.
Re-running the independent report requires the local raw and image-level
artifacts, which are outside this Git commit.
