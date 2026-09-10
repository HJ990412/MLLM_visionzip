# Reorder-prefix baseline experiment

This directory separates the effect of calibration-based importance reordering
from the online Static+Diverse selector on the frozen GQA 40-image / 240-question
workload.

## Primary result: calib=4

- Reorder + Prefix25: **60.42%**, 262.16 ms true TTFT, 306.079 MB/request
- Reorder + Static+Diverse25: **60.83%**, 337.59 ms, 300.401 MB/request
- Accuracy delta (Static+Diverse - Prefix): **+0.42 pp**
- Image-cluster 95% CI: **[-2.08, +3.33] pp**
- McNemar: SD-only 7, Prefix-only 6, exact p=1.0
- Mean/median chunk-set Jaccard: **0.5431 / 0.6000**
- Calibration importance-mass coverage at 25%: Prefix **73.50%**,
  Static **71.11%**, Static+Diverse **65.62%**
- Preregistered decision: **RETHINK**

See [`calib4/`](calib4/) for the complete validated report, raw request copy,
selection trace, overlap rows, paired statistics, and importance coverage.

## Supplement: calib=1 composed layout

- Prefix25: **58.33%**
- Static+Diverse25: **57.92%**
- Delta: **-0.42 pp**, image-cluster CI **[-3.33, +2.50] pp**
- Mean Jaccard: **0.5330**

See [`calib1_pair25/`](calib1_pair25/). This store was made by applying a
calib=1 composed permutation to an independent copy of the calib=4 store;
fresh-raster tie ordering is therefore not guaranteed bitwise.

The protected historical GQA result trees and files retained their preregistered
SHA-256 digests, and the original 50.8 GB calib=4 store retained its full content
digest `e570a6847743a203fc1e2892d736ebe8aa946647cbb0e280f212388da2c09d68`.
