# POLARIS

This repository reproduces the experiments in **POLARIS: Training-Free Audio
Fingerprinting with Saliency-Based Landmarks and Delaunay Grouping**. It
contains the final method, the two controlled variants used in the ablation
table, and adapters for every baseline in the paper.

**Paper:** [arXiv:2609.14820](https://arxiv.org/abs/2609.14820) | [Citation](#citation)

POLARIS selects local maxima of a locally normalized saliency field, stores
Delaunay-face triplets for each reference, and optionally adds triples from
two-hop Delaunay neighborhoods at query time.

## Paper protocols

| Dataset                      | References |                   Test queries | Use                              |
| ---------------------------- | ---------: | -----------------------------: | -------------------------------- |
| PEX Hard Medium, exact scale |        953 | 791 oracle-cropped annotations | controlled synthetic distortions |
| SD-RR v1.0                   |        496 |    1,488 ten-second recordings | real smartphone re-recordings    |

The PEX loader retains only annotations whose tempo is empty or `100` and
whose pitch is empty or `0`. PEX Hard Small was used for development and is
not part of the reported test matrix.

The reproduced POLARIS systems are:

- `polaris_o`: four STFT shifts, adjacent-bin probing, and original Delaunay
  faces only;
- `polaris_f`: `polaris_o` plus at most 24 additional two-hop query triples
  per landmark;
- `polaris_a`: starts from `polaris_o` and adds the two-hop query hashes unless
  the top result has at least eight matching hashes and a score margin of at
  least 0.40.

The SD-RR controls are exactly `magnitude_maxima` and `target_region`.
POLARIS-O versus POLARIS-F is the two-hop ablation.

## Installation

Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg are required.

```bash
git clone https://github.com/JihengLi/POLARIS.git
cd POLARIS
uv sync --python 3.12
uv run pytest
```

Panako additionally requires JDK 17 and LMDB. OLAF requires LMDB and Zig 0.16.
Install all pinned external systems and the pretrained NMFP checkpoint with:

```bash
bash scripts/setup.sh
```

Use repeated `--only` flags to install selected baselines, for example:

```bash
bash scripts/setup.sh --only audfprint --only olaf
```

Exact upstream revisions and evaluated profiles are listed in
[`benchmarks/README.md`](benchmarks/README.md).

## Data

Download and verify SD-RR from Zenodo:

```bash
uv run polaris-data sdrr
```

This creates `data/sdrr/`. SD-RR is archived at
[10.5281/zenodo.22169646](https://doi.org/10.5281/zenodo.22169646).

Download PEX Hard Medium from the official PEX distribution and place it at
`data/pex_hard_medium/` with this layout:

```text
data/pex_hard_medium/
├── annotations.csv
├── fma_tracks.csv
├── references/
└── queries/
```

Both loaders validate the complete paper protocol before an experiment starts.

## Reproduce the paper

Run all reported systems sequentially; each POLARIS evaluation uses six worker
processes:

```bash
bash scripts/reproduce_all.sh --workers 6
```

Run a subset with:

```bash
bash scripts/reproduce_all.sh --dataset sdrr --only polaris_f --workers 6
bash scripts/reproduce_all.sh --dataset pex --only nmfp
```

The runner resumes complete outputs and reuses compatible reference indexes.
The two controls run only on SD-RR. POLARIS-O, POLARIS-A, and POLARIS-F share
one reference index. Reference insertion, fingerprint serialization, query
iteration, and score ties use explicit deterministic orderings, so changing
the worker count does not turn tied candidates into chance outcomes.

Every experiment writes `query_results.csv` and `summary.json`. POLARIS and its
controls additionally write the fully resolved method configuration:

```text
outputs/<dataset>/<system>/
├── resolved_configuration.json
├── query_results.csv
└── summary.json
```

Baseline configurations and provenance are stored directly in their
`summary.json` files.

After the complete matrix finishes, regenerate the paper-facing result and
comparison files without rerunning audio:

```bash
uv run polaris-report --outputs outputs
```

This creates `outputs/results.csv` and `outputs/comparisons.csv`. The first
contains only track Top-1, SD-RR track-and-offset Top-1 at 0.1 s, logical
payload, mean query time, and confidence intervals. The second contains the
predeclared clustered paired comparisons used by the paper.

Failures and no-candidate outputs count as incorrect. SD-RR uncertainty and
comparisons cluster its three queries by reference track; PEX clusters by
montage query file. Logical payload excludes database serialization overhead.

## Citation

If you use POLARIS, please cite the paper:

```bibtex
@article{li2026polaris,
  title = {{POLARIS}: Training-Free Audio Fingerprinting with Saliency-Based Landmarks and {Delaunay} Grouping},
  author = {Li, Jiheng},
  journal = {arXiv preprint arXiv:2609.14820},
  year = {2026},
  eprint = {2609.14820},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  doi = {10.48550/arXiv.2609.14820},
  url = {https://arxiv.org/abs/2609.14820}
}
```

If you use SD-RR, also cite the
[dataset release](https://doi.org/10.5281/zenodo.22169646).

## Licenses

Repository code is MIT licensed. Downloaded baseline code and model weights
retain their upstream licenses.

SD-RR audio does not have one blanket license. Each file is governed by the
license in the release's `attribution.csv`; the collection contains CC BY, CC
BY-SA, CC BY-NC, CC BY-NC-SA, and Free Art License works. Preserve
`attribution.csv` and `LICENSES.md`. SD-RR-authored metadata and documentation
are CC BY-SA 4.0.
