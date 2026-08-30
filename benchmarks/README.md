# External baselines

The repository does not vendor baseline source code or neural weights. The
setup command fetches the exact upstream revisions into ignored `.cache/`
directories:

| Paper name | Upstream | Frozen revision/profile |
|---|---|---|
| Audfp-M / Audfp-Q | `dpwe/audfprint` | `cb03ba99feafd41b8874307f0f4e808a6ce34362` |
| Panako | `JorenSix/Panako` | `e4b0e1dbb55e340bc66c90bac0ceb82b2cf84211` (version 2.1) |
| OLAF | `JorenSix/Olaf` | tag `v2.0.10`, commit `532f1991ba170b39d2156b43935428814e833614` |
| NMFP-Triplet | `raraz15/neural-music-fp` | `e95e2b4009751274b060a6b74c26ae1323daae59`; official Zenodo checkpoint |

Audfp-M uses density/fanout 28/4. Audfp-Q keeps that same reference index and
changes only query construction (density 1440, fanout 86, per-frame peak cap
11, search depth 2000). OLAF's main-table row is explicitly the
`paper` profile with storage-matched parameters and closed-set
`min_match_count=1`; it is not an upstream-default row. Panako uses its official fingerprinting and
reject-disabled closed-set thresholds. NMFP uses the official pretrained
Triplet checkpoint and exhaustive candidate scoring.

Each adapter records the upstream commit, configuration, result rows, and
summary. Upstream licenses continue to apply to downloaded sources.
