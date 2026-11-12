# scripts

Numbers are the order in which the scripts were written. The ones here are the direct path of
[docs/report.md](../docs/report.md); the gaps are in `archive/` (dead ends and superseded
versions, each still importable and described in [docs/log.md](../docs/log.md)).

Run any of them with `uv run scripts/<name>.py`. Shared code is in `src/static2fate/`.

## 1. Data: from images to labelled single cells

| script | what it does | writes |
|---|---|---|
| `00_download_subset.sh` | downloads one well or all 30 of S-BIAD2515 | `data/raw/images/` |
| `01_segment_track.py` | segments nuclei on Hoechst, measures them, links them over time, registers the endpoint image | `data/processed/tracks/` |
| `02_fate_table.py` | one row per track: dose, frames covered, first and last nucleus measurements | `fate_table.parquet` |
| `03_cell_annexin.py` | cell masks on the last live frame, AnnexinV per cell in the registered endpoint image | `cell_annexin.parquet` |
| `04_fate_labels.py` | the fate label: AnnexinV above the control 90th percentile, or a condensed nucleus | `fate_labels.parquet` |
| `05_crops.py` | 128 px crops of every labelled cell at every frame | `crops.npy` |
| `16_divisions.py` | flags divisions, to see how many "dying" cells are dividing | `results/qc/` |
| `run_all.sh` | scripts 01 to 06 in order | |

## 2. Labels and readability

| script | what it does |
|---|---|
| `46_label_audit.py` | the label against two live markers; event timing; single-frame readability with controls (figures 2 and 5) |
| `44_green_red.py` | the cytoplasm's 488 / 561 ratio in real tracks and in generated crops |
| `06_baseline.py` | single-frame fate baseline, handcrafted against frozen DINOv2 (also caches the embeddings) |
| `07_vjepa.py` | the same with frozen V-JEPA 2 on the first k frames |

## 3. Model

| script | what it does |
|---|---|
| `09_train_flow.py` | the split (held-out wells), the loaders and the fate classes every other script imports; its `main` trains the first pixel-space model |
| `31_train_autoencoder.py` | the KL autoencoder, 128 px crop to 4 x 32 x 32 code |
| `32_reconstruct.py` | how much of a cell survives the autoencoder |
| `33_encode_latents.py` | encodes all crops once |
| `34_train_latent.py` | the flow model on the codes: frame or video, two or three fate classes |
| `35_generate_latent.py` | draws futures for the held-out cells; guidance scale and fate override as arguments |
| `export_kaggle.py` | writes the crops as a Kaggle dataset; the kernels are in `kaggle/` |

## 4. Scoring the generated futures

| script | question | figure 6 panel |
|---|---|---|
| `20_feature_score.py` | caches handcrafted and DINOv2 features of real and generated frames | |
| `21_real_vs_generated.py` | can a probe tell generated from real? | A |
| `42_energy_terms.py` | distance to the cell's own future, against its 0 h frame and a random cell | B |
| `22_identity.py` | is the cell's own future the nearest real one? | C |
| `45_guidance_mixture.py` | fraction of generated frames that are condensed, and green | D, E |
| `30_temporal.py` | does a clip move like a real cell? | F |
| `48_scorecard.py` | collects the six panels | |

## 5. Figures and movies

| script | what it does |
|---|---|
| `47_gallery.py` | every image figure of the report and gallery (figures 1, 3, 4, 7, 8, 9) |
| `36_movies.py` | the GIFs in `results/movies/` |
