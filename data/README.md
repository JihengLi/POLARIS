# Data layout

Audio is deliberately not stored in the Git repository.

```text
data/
├── sdrr/                    # extracted SD-RR v1.0 Zenodo record
└── pex_hard_medium/         # original PEX Hard Medium download
```

Run `uv run polaris-data sdrr` to download and verify SD-RR v1.0 from
<https://doi.org/10.5281/zenodo.22169646>. The command removes downloaded ZIP
archives after successful extraction, so it does not keep duplicate audio.

PEX must be obtained from its official distribution. Place the untouched Hard
Medium directory at `data/pex_hard_medium`; it must directly contain
`annotations.csv`, `fma_tracks.csv`, `references/`, and `queries/`. POLARIS evaluates all 791
annotations whose tempo is empty/100 and whose pitch is empty/0. PEX Hard
Small was used only as a development set and is not part of the reported test
runner.

SD-RR audio has mixed per-track licenses. Retain the downloaded release's
`attribution.csv` and `LICENSES.md`; neither file is replaced by a blanket
repository license.
