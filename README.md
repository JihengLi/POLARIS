# POLARIS

POLARIS is a training-free audio fingerprinting system based on locally
normalized saliency landmarks, Delaunay triplet fingerprints, and asymmetric
query-side expansion. This repository contains only the implementation,
dataset adapters, baseline adapters, and experiments needed to reproduce the
paper:

> **POLARIS: Training-Free Audio Fingerprinting with Delaunay Landmark Grouping and Query-Side Expansion**

## Frozen paper protocols

| Protocol | References | Trials | Reported use |
|---|---:|---:|---|
| PEX Hard Medium, exact scale | 953 | 791 oracle-cropped annotations | controlled synthetic complement |
| SD-RR v1.0 | 496 | 1,488 ten-second queries | real smartphone re-recordings |

PEX rows are retained only when `tempo` is empty or `100` and `pitch` is empty
or `0`. PEX Hard Small was used only during development to freeze parameters;
it is not run or reported by the test script. SD-RR v1.0 is archived at
[Zenodo](https://doi.org/10.5281/zenodo.22169646).

## Systems reproduced

The main comparison contains:

- Audfp-M: Audfprint density/fanout 28/4;
- Audfp-Q: the same reference index with a payload-matched dense query;
- Panako 2.1;
- official OLAF v2.0.10 with the separately labelled storage-matched paper
  profile and closed-set `min_match_count=1` on both protocols;
- pretrained NMFP-Triplet;
- POLARIS-A, the development-frozen adaptive query policy;
- POLARIS-F, the complete fixed query.

The SD-RR ablation table contains exactly five controlled variants:
spectrogram-magnitude maxima, two-hop only, dense-landmark only, no query-side
expansion, and OLAF-style sequential target-region triplets. All other modules
are unchanged within each control.

Exact revisions and profile disclosures are in
[`benchmarks/README.md`](benchmarks/README.md) and the machine-readable
[`configs/experiment_matrix.json`](configs/experiment_matrix.json).

## Requirements

The core implementation requires Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/),
and FFmpeg. External baselines additionally require:

- JDK 17 and LMDB for Panako 2.1;
- LMDB and Zig 0.16 for OLAF v2.0.10;
- enough memory for the pretrained TensorFlow NMFP model (CPU is supported).

The complete experiment is long-running. Every reference index and expensive
embedding cache is reused, and POLARIS writes atomic checkpoints every ten
query tasks. Closing a laptop suspends processes on most systems; use `caffeinate`
on macOS or an equivalent inhibitor if uninterrupted execution is required.

## 1. Install the repository

```bash
git clone https://github.com/JihengLi/POLARIS.git
cd POLARIS
uv sync --python 3.12
uv run pytest
```

## 2. Prepare data

Download, verify, and extract SD-RR without retaining duplicate ZIP archives:

```bash
uv run polaris-data sdrr
```

This produces `data/sdrr/`. The command validates all 496 references and 1,488
manifest rows before accepting the extraction.

Download PEX Hard Medium from the official PEX distribution and place the
untouched directory at `data/pex_hard_medium/`:

```text
data/pex_hard_medium/
├── annotations.csv
├── fma_tracks.csv
├── references/
└── queries/
```

The loader rejects any directory that does not have exactly 953 references,
219 montage query files, and 791 pitch/tempo-preserving annotations.

## 3. Fetch pinned baselines

```bash
bash scripts/setup.sh
```

Sources, built binaries, virtual environments, and NMFP weights are placed in
ignored per-baseline cache directories. Nothing is vendored into the POLARIS
package. To prepare only selected systems, repeat `--only`, for example:

```bash
bash scripts/setup.sh --only audfprint --only olaf
```

## 4. Reproduce the paper

Run the complete matrix with six POLARIS/query workers:

```bash
bash scripts/reproduce_all.sh --workers 6
```

The command is resumable. To run one dataset or one system:

```bash
bash scripts/reproduce_all.sh --dataset sdrr --only polaris --workers 6
bash scripts/reproduce_all.sh --dataset pex --only nmfp
```

Panako intentionally remains internally single-worker because its official
macOS native backend and LMDB writer are not safe under the same parallel
strategy as the other adapters.

## 5. Results

Per-system evidence remains traceable in `outputs/<dataset>/<system>/`, with a
resolved configuration, per-trial CSV, and summary JSON. The paper-facing
outputs are intentionally small:

```text
outputs/
├── results.csv       # accuracy, logical payload, runtime, and confidence intervals
└── comparisons.csv   # clustered paired comparisons
```

Regenerate them without rerunning audio:

```bash
uv run polaris-report --outputs outputs
```

Logical payload—not serialized database size—is reported uniformly. Discrete
hash methods use a 16-byte reference posting and a 12-byte query record;
Panako uses a 20-byte reference record; NMFP uses 520/516-byte reference/query
embedding records. Native file sizes are not compared across database engines.

SD-RR confidence intervals and paired comparisons cluster by `reference_id`
(three queries per track). PEX comparisons cluster by montage `query_id`.
Failures and no-match results count as incorrect. SD-RR track-and-offset Top-1
additionally requires absolute reference-offset error no greater than 0.1 s.
The evaluation code intentionally emits no R@K, MRR, mAP, F1, latency
percentiles, or distortion-wise diagnostic metrics.

## Data and software licenses

The POLARIS code in this repository is MIT licensed. External baseline source
and model licenses apply to their downloaded caches.

SD-RR does **not** have a blanket audio license. Its tracks use mixed CC BY,
CC BY-SA, CC BY-NC, CC BY-NC-SA, and Free Art License terms. The exact license
for every audio file is recorded by the downloaded `attribution.csv`; preserve
that file and read the accompanying `LICENSES.md`. SD-RR-authored metadata and
documentation are CC BY-SA 4.0.

## Citation

Software citation metadata is in [`CITATION.cff`](CITATION.cff). Cite SD-RR by
its version DOI, `10.5281/zenodo.22169646`, rather than a mutable URL.
