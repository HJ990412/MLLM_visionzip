#!/bin/bash
# Builds the H2O-ordered store used by the h2o arms.  Source: a COPY of the
# canonical importance store (kvstore/), restored to raster, then repacked
# with the H2O metric.  kvstore/ itself is only read.
set -euo pipefail
cd /home/dblab/hj/mllm_v2
NEW=runs/impress_fidelity_20261116T052410Z/kvstore_h2o
# v_hidden.pt (SparseVLM rater input) and static.pt (VisionZip metadata) are
# deliberately NOT copied: the h2o path must work without them.
rsync -a --exclude v_hidden.pt --exclude static.pt kvstore/ $NEW/
python scripts/02_reorder.py --order raster --allow-composed-raster-target --store $NEW --limit 40 --profile-out results/impress_fidelity_20261116T052410Z/logs/reorder_raster_profile.json
python scripts/02_reorder.py --order importance --calib-questions 4 --importance-metric h2o --store $NEW --limit 40 --record-layout-provenance --rebuild-separator-sidecar --profile-out results/impress_fidelity_20261116T052410Z/logs/reorder_h2o_profile.json
