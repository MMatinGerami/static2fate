# static2fate

[![CI](https://github.com/MMatinGerami/static2fate/actions/workflows/ci.yml/badge.svg)](https://github.com/MMatinGerami/static2fate/actions/workflows/ci.yml)

From one image of a drug-treated cell, can we tell which of two futures it will take, and when
that became decided?

Static2Dynamic ([Boyer et al., 2026](https://doi.org/10.64898/2026.05.18.725860)) generates a
video of a process from static snapshots, assuming one trajectory. This project works where the
same starting cells take two: HeLa cells under a staurosporine dose series, filmed every 30 min
for 6 h and then stained for AnnexinV, so every cell has a video and a known fate
([S-BIAD2515](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2515), Lippincott et al.,
Way lab, CC0).

![two cells](results/figures/fig3_two_cells.png)

Status (October 2026): data pipeline, baselines, a fate-conditioned generative model, a
version that also sees an earlier frame of the cell, and GPU runs showing that more training
or resolution does not change the picture.

## What is here

- Nuclear segmentation, tracking over 13 frames and registration of the terminal AnnexinV image
  for all 30 wells (119 fields): 6,676 cells followed from start to end with a fate label.
- Fate label: AnnexinV above the control 90th percentile inside a cell mask, or a condensed
  nucleus (area halved, Hoechst doubled). Replicate wells agree within a few points.
- Single-cell crops (13 frames x 3 channels x 128 px) and single-frame fate baselines.

## Findings so far

| dose (nM) | 0 to 2.4 | 4.9 | 9.8 | 19.5 | 39.1 | 78 | 156 |
|---|---|---|---|---|---|---|---|
| fraction apoptotic at 6 h | 0.11 to 0.17 | 0.19 | 0.19 | 0.31 | 0.74 | 0.91 | 0.92 |

- Both fates occur in one band, 9.8 to 39 nM. Static2Dynamic's ChromaLive model was trained at
  78 and 156 nM, where nearly all cells die.
- From a single frame, fate is not readable before about 2 h and largely readable by 4 h
  (within 19.5 nM, leave-one-field-out AUROC 0.51 at 0 min, 0.60 at 2 h, 0.78 at 4 h, 0.84 at 6 h).
- Eleven handcrafted nuclear measurements match or beat frozen DINOv2 at every informative dose.
- A frozen video encoder does not see fate earlier either: V-JEPA 2 on frames 1 to 8 reaches
  within-dose AUROC 0.64 at 3.5 h against 0.68 for one frame of handcrafted features
  (difference -0.045, 95% CI -0.067 to -0.021; `results/tables/vjepa_vs_frame.csv`). It does
  beat DINOv2 averaged over the same frames at 2.5 h (+0.036, CI 0.007 to 0.067).

![single-frame baseline](results/figures/fig2_single_frame_baseline.png)

## First generative model

A small conditional flow-matching model (9.4 M parameters, written from scratch in
`src/static2fate/flow.py`) generates a 64 px cell image given the time after drug and the fate.
Trained 45 min on an Apple M5 Pro (MPS) on 1,919 cells; one replicate well per dose held out.

Starting from the same noise and asking for both fates gives two branches of the same cell. They
look the same up to 2 h and separate from 3 h, when the "dies" branch condenses its nucleus.

![counterfactual pairs](results/figures/fig4_counterfactual_pairs.png)

On held-out wells, generated dying cells change at the right time: nucleus brightness rises and
nucleus area falls between 2 and 3 h, as in real dying cells. They overdo it, though. At 6 h the
generated dying nucleus is 150 px against 219 px real, and Hoechst is 0.47 against 0.22. Survivors
come out about 15% too large. Nucleus is the Otsu object nearest the crop centre
(`src/static2fate/readouts.py`); an earlier fixed-threshold version exaggerated the size gap
because generated images are slightly dimmer.

![generated vs real](results/figures/fig5_generated_vs_real.png)

## Starting from a real cell

Run the flow backwards from a held-out cell's 0 h image (fate left out) to get its noise, then
forwards under each fate to 3 to 6 h. The round trip gives the cell back almost exactly (pixel
MSE 0.002). Is the branch with the cell's true fate closer to what the cell actually looked like
later than simply copying its 0 h frame? On 816 cells from four held-out wells (distances on
nucleus area, nucleus Hoechst, cytoplasm green and red; 95% CIs from resampling the 12 fields):

| cells | 3 h | 4 h | 5 h | 6 h |
|---|---|---|---|---|
| die, nucleus condensed (95) | 0.46 | 0.57 | 0.67 | 0.72 (0.54 to 0.80) |
| survive (571) | 0.47 | 0.44 | 0.49 | 0.45 (0.39 to 0.49) |
| die, AnnexinV only (150) | 0.34 | 0.33 | 0.35 | 0.37 (0.28 to 0.44) |

So the dies branch predicts the course of cells that visibly die better than "nothing changes",
from 4 h on. The survives branch is close to a frozen copy of the first frame and adds nothing.
Cells that are AnnexinV-positive but still look normal at 6 h are pushed too far. In the images
the generated dying cytoplasm turns green, while real dying cells end as a condensed red and blue
body, so the colour of the change is wrong even when its timing is right.

![model vs copy](results/figures/fig6_inversion_vs_copy.png)

![branches from real cells](results/figures/fig7_branches_from_real_cells.png)

## Giving the model an earlier frame of the cell

The model above has no notion of how far along a cell is. A second model (same network, context
image as three extra input channels, `scripts/13_train_ctx.py`, 60 min on MPS) is also given one
observed frame of the cell, at 0 to 3 h, and generates a later frame under each fate. Scored the
same way on the same 816 held-out cells (`scripts/14_eval_ctx.py`):

| cells, model beats copying the 0 h frame | 3 h | 4 h | 5 h | 6 h |
|---|---|---|---|---|
| die, condensed: inversion | 0.46 | 0.57 | 0.67 | 0.72 |
| die, condensed: context | 0.48 | 0.71 | 0.72 | 0.82 |
| die, AnnexinV only: inversion | 0.34 | 0.33 | 0.35 | 0.37 |
| die, AnnexinV only: context | 0.36 | 0.33 | 0.38 | 0.42 |
| survive: inversion | 0.47 | 0.44 | 0.49 | 0.45 |
| survive: context | 0.34 | 0.40 | 0.40 | 0.42 |

Paired on the same cells (`results/tables/ctx_vs_inversion_paired.csv`), context helps cells
that die with a condensed nucleus (+0.14 at 4 h, CI 0.06 to 0.24; +0.11 at 6 h, CI 0.03 to
0.21). It does not help the AnnexinV-only cells, which is what I added it for: they are still
pushed too far. For survivors it is worse at 3 h (-0.13, CI -0.17 to -0.08), because inversion
keeps the cell's own noise while the context model draws new noise for every sample, and one
random sample is penalised for that spread (scored with several samples below, this goes away). The overshoot of
dying nuclei is unchanged (163 px generated against 219 px real at 6 h). Given the 2 h frame
instead and compared with copying that frame, the pattern is the same
(`results/tables/model_vs_copy.csv`).

![context vs inversion](results/figures/fig8_context_vs_inversion.png)

Looking at examples (`results/figures/fig9_context_examples.png`) also turned up a data
question: some cells round up with bright chromatin and flatten again an hour later. That looks
like division, not apoptosis, and a rounded mitotic cell can pass for a dying one in a single
frame.

## Scoring several samples per cell

One sample from a stochastic model is a harsh test against a deterministic copy. So each
held-out cell got 8 samples per frame from the context model (0 h frame, true fate), scored with
the energy score (mean distance to the real readouts minus half the mean distance between
samples; for a single forecast it is the plain distance, so copy and inversion are on the same
scale; `scripts/17_energy_score.py`). Because the energy score credits spread, I added a
baseline that never looks at the image: the cell's 0 h readouts plus the change of a random
training cell with the same fate, 8 draws ("copy + typical change").

| energy score difference, 6 h (below 0 = first is better) | survive | die, condensed | die, AnnexinV only |
|---|---|---|---|
| context model vs copy | -0.48 (-0.57 to -0.41) | -2.25 (-2.55 to -1.52) | -0.72 (-1.08 to -0.33) |
| copy + typical change vs copy | -0.41 (-0.46 to -0.36) | -2.03 (-2.16 to -1.73) | -0.70 (-0.99 to -0.38) |
| context model vs copy + typical change | -0.07 (-0.13 to -0.01) | -0.22 (-0.41 to 0.25) | -0.02 (-0.13 to 0.12) |

Scored this way the context model beats copying the cell for every group and hour, survivors
included, and beats the inversion model everywhere. But most of that is what any forecast gets
by knowing the usual course of each fate. Beyond that baseline the images add a clear gain only
for cells that die with a condensed nucleus (-0.24, -0.23, -0.34 at 3, 4, 5 h, CIs below 0; at
6 h the CI crosses 0), a small one for survivors, and nothing for AnnexinV-only cells. Both the
model and the baseline are told the true fate, so this measures the dynamics given the fate,
not fate prediction. Distances are on four readouts only.

![energy score](results/figures/fig10_energy_score.png)

## Eight times the training, twice the resolution

Two more context models were trained on Kaggle (two T4 GPUs, 8 h each, `kaggle/`): the same
64 px model for 98,000 steps instead of 7,000, and a 128 px model for 51,000 steps. Scored on
the same 816 cells with the same 8-sample energy score, paired by cell
(`scripts/18_compare_models.py`, `results/tables/model_comparison.csv`):

| energy score minus the 1 h Mac model, 6 h | survive | die, condensed | die, AnnexinV only |
|---|---|---|---|
| 64 px, 8 h | -0.03 (-0.05 to 0.00) | +0.06 (-0.04 to 0.14) | +0.01 (-0.09 to 0.10) |
| 128 px, 8 h | -0.01 (-0.03 to 0.02) | -0.01 (-0.13 to 0.06) | +0.02 (-0.04 to 0.07) |

Nothing changes by more than 0.1 at any hour, and the few intervals that exclude 0 go both
ways (the longer run is a little better for survivors and a little worse for condensed dying
cells at 3 to 4 h). The training loss is flat from about 2,000 steps in all three runs. So on
this measure the model is not limited by compute or pixels. The next section shows the measure
was the problem.

![model comparison](results/figures/fig11_model_comparison.png)

## A finer comparison than four readouts

The sampled futures (8 per cell, kept as images by `scripts/19_generate_samples.py`) were
rescored in three feature spaces (`scripts/20_feature_score.py`): the four readouts, nine
crop-level nucleus measurements (area, eccentricity, solidity, nucleus and ring mean per
channel, `src/static2fate/features.py`) and frozen DINOv2-base embeddings, each standardised
on the real held-out frames; at 64 px, with everything averaged down, and at 128 px, with the
64 px samples upsampled and the real crops native. In DINOv2 space "copy + typical change" is a
linear extrapolation that leaves the manifold of real images, so the baseline there is a random
real training cell of the same fate at the same frame (8 draws). Energy score minus that
baseline at 6 h; below 0 means the model's futures are closer to this cell's real future than
another real cell of the same fate is:

| | 64 px, 1 h | 64 px, 8 h | 128 px, 8 h |
|---|---|---|---|
| readouts: survive / die condensed / die AnnexinV only | -0.12 / -0.13 / +0.18 | -0.15 / -0.05 / +0.11 | -0.13 / -0.07 / +0.14 |
| DINOv2 at 64 px | +3.0 / +5.4 / +5.0 | +0.7 / +1.0 / +1.2 | +1.1 / +2.0 / +1.8 |
| DINOv2 at 128 px | +4.3 / +6.3 / +6.6 | +2.0 / +2.2 / +3.2 | +1.6 / +2.7 / +2.6 |

In readout and handcrafted space the three models are still the same (paired differences
within 0.15, mixed signs), and each is slightly closer than a random same-fate cell for
survivors and condensed dying cells, not for AnnexinV-only cells. In DINOv2 space the 8 h
models are 1.9 to 4.4 units closer to the real future than the 1 h model, paired by cell, with
CIs excluding 0 in every group (`results/tables/feature_score_vs_1h.csv`). So the extra
training did change the images: they look more like real cells, which four readouts could not
see. At native 128 px the 128 px model is also closer than the upsampled 64 px model for
survivors and AnnexinV-only cells (1.6 against 2.0, 2.6 against 3.2), not for condensed ones.

Still, every model is farther from the cell's real future than a random real cell of the same
fate is, and a linear probe tells generated from real with AUROC 0.995 to 1.000 in DINOv2
space for all three (0.58 to 0.75 on the handcrafted features, 0.51 to 0.55 on the readouts;
`scripts/21_real_vs_generated.py`). DINOv2 distance here is mostly realism and time after
drug: the cell's own 0 h image is 17 units farther from its future than a random cell at the
right frame is. So it measures how real the futures look, not whether they are this cell's.

![feature spaces](results/figures/fig12b_feature_score_vs_random_cell.png)

## Is it still the same cell?

The energy scores mix three things: realism, time after drug and whose cell it is. To isolate
the last, every real held-out cell of the same fate at the target frame is a candidate, the
candidates are ranked by distance to the query, and the percentile rank of the query cell's own
real future is recorded (0 = nearest, 0.5 = chance; `scripts/22_identity.py`). Queries: each
model's 8 samples (ranks averaged), the cell's own 0 h image, and its real 3 h image as a
reference for how recognisable a cell is from a real frame 1 to 3 h earlier.

| percentile rank of own future, 6 h | survive | die, condensed | die, AnnexinV only |
|---|---|---|---|
| DINOv2 at 64 px: models (1 h / 8 h / 128 px) | 0.47 / 0.44 / 0.46 | 0.48 / 0.43 / 0.46 | 0.49 / 0.48 / 0.48 |
| DINOv2: copy 0 h | 0.35 | 0.50 | 0.33 |
| DINOv2: real 3 h | 0.25 | 0.29 | 0.25 |
| readouts: models (1 h / 8 h / 128 px) | 0.37 / 0.32 / 0.35 | 0.40 / 0.38 / 0.40 | 0.49 / 0.45 / 0.48 |
| readouts: copy 0 h | 0.26 | 0.59 | 0.35 |
| readouts: real 3 h | 0.18 | 0.38 | 0.21 |

The generated futures keep little of the cell. In DINOv2 space their own future ranks near
chance (0.43 to 0.51), while the cell's 0 h image ranks it at 0.33 to 0.35 for survivors and
AnnexinV-only cells, and a real 3 h image at 0.25 to 0.29. Paired by cell, the models are 0.09
to 0.16 worse than the 0 h copy in those groups, CIs excluding 0
(`results/tables/identity_rank_paired.csv`). Only condensed dying cells change so much that
the 0 h image stops being useful (0.50; 0.59 in readouts, worse than chance), and there the
models match or beat it. Top-1 rates tell the same story: models 0.2 to 1%, copy 1 to 5%, a
real 3 h frame 5 to 34%. Longer training helps a little (64 px 8 h against 1 h: -0.03 for
survivors, CI -0.04 to -0.02), 128 px does not.

So the context model makes plausible cells of the right fate and time, not this cell's future:
the identity in the context image is mostly lost in sampling. That is the same thing the
inversion comparison showed from the other side (the survives branch added nothing over a
copy), now measured directly.

![identity](results/figures/fig13_identity.png)

## Are some "dying" cells dividing?

`scripts/16_divisions.py` flags a division when a nucleus turns small and bright for a frame or
two, comes back larger and dimmer within 1.5 h, and a new track is born next to it (the second
daughter). Apoptotic nuclei condense too but do not come back. Image strips for checking the
calls by eye are in `results/qc/divisions_*.png`: of the 16 flagged cells shown, about 10
clearly divide, 4 are unclear and 2 are dying cells.

| labelled cells (all 13 frames tracked) | n | divided | condensed in the last hour, AnnexinV negative |
|---|---|---|---|
| survive | 2,724 | 21 (0.8%) | 12 |
| die, condensed | 1,416 | 5 (0.4%) | 31 |
| die, AnnexinV only | 530 | 4 (0.8%) | 0 |

So divisions do not explain the AnnexinV-only problem: 4 of 530 cells. The second way mitosis can
enter the label is a cell caught rounding up at the end, which the "condensed" clause calls
apoptotic; mitotic cells are AnnexinV negative, so the 31 cells in the last column are the
suspects. By eye about a third of them look mitotic and the rest like ordinary apoptosis. At
most 40 of 1,946 dying cells (2.1%) are affected, probably fewer.

The bigger finding is what is missing. The nucleus count per field grows 6 to 17% over 6 h at
0 to 9.8 nM (`results/tables/nucleus_count_growth.csv`), so roughly a tenth of cells divide,
but only about 1% of labelled cells do. Division breaks the track (the nucleus rounds, moves and
splits), so dividing cells rarely end up with a complete track and a label. The labelled set,
and every model trained on it, is mostly cells that did not divide during the movie.

## Next

- Invert the context frame under the context model, so it keeps the cell's identity and uses
  its state.
- Close the realism gap: a linear probe still separates generated from real perfectly. More
  steps at 128 px, and a perceptual or adversarial term in the loss.
- Keep the cell: start the sampler from the inverted context frame instead of fresh noise, or
  tie the output to the context more strongly (feature-level conditioning or guidance towards
  the context), and re-run the identity ranking.
- Link daughters to mothers in the tracker so dividing cells get a fate too; do cells that
  divide under drug die more or less often?
- Get the colour of apoptosis right; check whether the ChromaLIVE channel scaling per field is
  part of the problem.
- Video: generate the whole 3 to 6 h course at once, with temporal consistency, instead of
  one frame at a time.

## Run

```
uv sync
scripts/00_download_subset.sh          # all 30 wells, 45 GB, from EBI
scripts/run_all.sh                      # segmentation to baseline
uv run scripts/09_train_flow.py --minutes 45     # flow models, then 10 to 15 for evaluation
uv run scripts/13_train_ctx.py --minutes 60
uv run pytest
```

On a GPU: `uv run scripts/export_kaggle.py <user>` writes the 2,735 cells the models use
(1.7 GB) as a dataset folder; `kaggle/train_kaggle.py` trains from it on Kaggle.

Notes, decisions and corrections are dated in [notes/notebook.md](notes/notebook.md).

Known limits: one plate; the label fixes a 10% positive rate in controls; tracking keeps a
quarter to half of cells, fewer in dense regions, and almost none that divide; two deposited files are truncated and the
deposited terminal images need field matching (handled in `src/static2fate/fields.py`).
