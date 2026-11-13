# static2fate

[![CI](https://github.com/MMatinGerami/static2fate/actions/workflows/ci.yml/badge.svg)](https://github.com/MMatinGerami/static2fate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

HeLa cells treated with the same dose of staurosporine do not all die at once. From one image
of a treated cell, how early can its fate be read? And given that first image and a fate, can
a generative model draw the cell's future?

Data: one plate of the Way lab's ChromaLIVE time-lapse
([S-BIAD2515](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2515), CC0), ten doses,
13 frames over 6 h, then an AnnexinV stain. The project tracks 6,676 cells from the first frame
to the stain, labels their fate, checks that label, and trains a flow-matching video model
written from scratch.

![fields by dose](results/figures/fig01_fields_by_dose.png)

*One field per dose at 0 h and 6 h, and the AnnexinV image taken afterwards. Blue Hoechst,
green ChromaLIVE 488, red ChromaLIVE 561. Circles: tracked nuclei by fate group.*

## The project in five steps

| | Step | Result | Figure |
|---|---|---|---|
| 1 | **Data.** Segment, track and crop single cells in 119 fields | 23,902 nuclei at 0 h, 6,676 followed to the endpoint with a fate label | [1](results/figures/fig01_fields_by_dose.png), [gallery](docs/gallery.md) |
| 2 | **Label, and its audit.** AnnexinV above the control 90th percentile, or a condensed nucleus; checked against two markers read from the movie alone | The AnnexinV clause has a 10% background by construction: 44% of the "dies, AnnexinV only" cells at the model doses are expected from the threshold alone, and the stain misses a third of cells with a condensed nucleus. Both outcomes are common at one dose only, 39.06 nM | [2](results/figures/fig02_labels.png) |
| 3 | **A new live marker.** The cytoplasm turns from red to green | Of the cells whose nucleus condenses, 95 to 98% turn green, most of them first, by a median of half an hour | [4](results/figures/fig04_colour_shift.png) |
| 4 | **How early is fate readable from one frame?** | Not before the cell starts to change. The rise in AUROC under the fate label (0.50 at 0 h to 0.84 at 6 h) also appears in untreated wells (0.41 to 0.79). Frozen DINOv2 and V-JEPA 2 do not beat eleven nucleus measurements | [5](results/figures/fig05_readability.png) |
| 5 | **Drawing the future.** A latent flow-matching video model given the 0 h frame and the fate group | Right look, colour and motion for each group, and texture close to its autoencoder's limit. But its futures are closer to the cell's real future than the 0 h frame only for condensing cells, and they are not recognisable as that cell | [6](results/figures/fig06_scorecard.png), [7](results/figures/fig07_real_vs_generated.png), [9](results/figures/fig09_counterfactual.png) |

The full account with numbers and confidence intervals is in **[docs/report.md](docs/report.md)**.

![counterfactual](results/figures/fig09_counterfactual.png)

*The same 0 h cell drawn at 6 h under each of the three fate classes, from the same noise.*

![scorecard](results/figures/fig06_scorecard.png)

## Limits

- One plate, three wells per dose, dose not randomised across the plate; held-out wells are one row.
- The fate label is the weakest part (step 2). The three groups are stages reached by 6 h more than three fates.
- The model is told the fate. Nothing here predicts it, and counterfactual futures cannot be checked cell by cell.
- 28% of nuclei end with a label, and almost no dividing cell.
- One training run per model.

## Where things are

| | |
|---|---|
| [docs/report.md](docs/report.md) | the direct path: data, label audit, two questions, limits, next steps |
| [docs/gallery.md](docs/gallery.md) | many more images of real and generated cells |
| [docs/code_tour.md](docs/code_tour.md) | the code file by file |
| [docs/log.md](docs/log.md) | all 25 experiments in the order they were run, dead ends included |
| `src/static2fate/` | library: fields, cells, audit, flow, video, autoencoder, evaluate, features, display |
| `scripts/` | the pipeline, in order ([scripts/README.md](scripts/README.md)); `scripts/archive/` holds superseded experiments |
| `kaggle/` | the GPU jobs for the autoencoder and the latent models |
| `results/figures/` | the report's figures; `supporting/` holds the log's |
| `results/tables/` | every number as CSV |

## Quickstart

One well is about 1.5 GB and runs on a laptop; all 30 wells are 45 GB.

```bash
uv sync
scripts/00_download_subset.sh C-08       # one well of S-BIAD2515, 19.53 nM; no argument = all 30 wells
scripts/run_all.sh                       # segmentation, tracking, fate labels, crops, baseline
uv run scripts/46_label_audit.py         # label audit and readability (needs all wells)
uv run scripts/47_gallery.py real        # image galleries
uv run pytest
```

The models were trained on Kaggle T4 GPUs: `uv run scripts/export_kaggle.py <user>` writes the
crops as a dataset, and `kaggle/ae` and `kaggle/fate3` train the autoencoder and the latent
video model. Scoring and figures then run on a laptop (see [docs/report.md](docs/report.md#reproduce)).

## Citation

```bibtex
@software{gerami2026static2fate,
  author  = {Gerami, Matin},
  title   = {static2fate: single-cell fate from live-cell imaging},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/MMatinGerami/static2fate}
}
```

Data: S-BIAD2515, Lippincott et al., Way lab, CC0
([BioImage Archive](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2515)).

## License

MIT
