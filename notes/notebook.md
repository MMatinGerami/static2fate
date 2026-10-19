# Lab notebook

## 2026-10-19

Why this project. Genovesio's Static2Dynamic preprint (bioRxiv, May 2026) reconstructs a video from
static snapshots by learning a one-dimensional pseudotime and conditioning a diffusion model on it.
Its discussion lists what it cannot do: branching fates, stochasticity, uncertainty about a cell's
future. Those are the three things I want to work on.

Question, first version. Given one frame of a HeLa cell after staurosporine, generate the two
futures (apoptosis, survival), estimate the probability of each, and locate the pseudotime at which
the two generated branches separate: a morphological point of no return. Check all of it against
real videos with per-cell AnnexinV ground truth.

Data. S-BIAD2515 released 21 Jan 2026 (Way lab). 30 wells, 4 fields, 14 timepoints every 30 min
(the study text says 13 live + 1 terminal), 4 live channels, terminal Hoechst + AnnexinV. The
BioStudies listing holds only the TIFFs plus a file list; the description also promises tracks,
CellProfiler features and scDINO embeddings, which I have not located yet. To do: find them (Way
lab GitHub?) before downloading 45 GB of images.

Compute. Static2Dynamic is CUDA-only in its README. On this Mac I can do pseudotime estimation
(CPU) and small latent-space experiments; the full diffusion training needs a GPU I do not have.

Baseline to beat before anything generative: frozen DINOv2 features of one frame plus logistic
regression, fate prediction at each timepoint. If that already does it, the generative part needs
a different justification (counterfactual videos, commitment time), and I should say so.

Open doubts. Branching pseudotime from snapshots alone is not identifiable; some tracked cells
with known fate will be needed to learn the fate direction. How few is itself a question.

Later the same day. Found the source of the processed data: preprint Lippincott et al. 2025
(bioRxiv 10.1101/2025.10.23.684203). Code and profiles on Zenodo 10.5281/zenodo.17419422
(image processing, tracking with Ultrack, profiling) and 10.5281/zenodo.17419412 (analysis,
models). Project repo github.com/WayScience/live_cell_timelapse_apoptosis_analysis.
Doses: ten, 2-fold serial from 156 nM down to 0, triplicate wells, 0.1% DMSO vehicle; 13 live
timepoints (0 to 360 min) and one terminal fixed image (AnnexinV-633 + Hoechst) registered to the
last live frame. Ground truth is a continuous AnnexinV intensity per tracked cell at 6 h only;
there is no binary label and no fate information before 6 h. So "when did the cell commit" has no
direct ground truth; it can only be tested indirectly (prediction-from-frame-t curves, and dose
monotonicity).

## 2026-10-20

Downloaded the two Zenodo records (code snapshots of the two Way lab repos, 21 MB and 333 MB,
unpacked under data/raw/zenodo/). They contain code and figures, not data. The only tables are
well-level: 5.bulk_timelapse_model/data_splits/{train,test}.parquet (20 + 10 wells, 2,858
columns of median CellProfiler + scDINO features, Metadata_apoptosis_ground_truth = control /
negative / positive at the well level). The per-cell profiles, Ultrack tracks and per-cell
AnnexinV values are not deposited anywhere I can find: not on BioStudies (images only, checked the
study JSON: one study component, "Raw Images"), not on Zenodo, not in the GitHub repos, and the
preprint's data statement points only to the two code records. The repo's 0.download_data README
is a placeholder. So the processed data would have to be regenerated from the images.

Useful facts from the code: plate map in data/platemap_6hr_4ch.csv (dose per well); illumination
correction and max-projection per well/FOV; nuclei segmented with Cellpose, tracked with Ultrack
(5.cell_tracking/scripts/0.nuclei_tracking.py, 14 frames of 1900x1900); terminal AnnexinV image
registered to frame 13 by an x-y offset (4.cellprofiler_analysis/results/*_offsets.parquet, these
ARE in the repo); endpoint cells matched to tracks by nucleus centroid within 10 px
(7.sc_timelapse_model/scripts/0.data_cleaning.py). No binary apoptosis label is ever defined;
the per-cell target is cytoplasmic integrated AnnexinV intensity.

A shortcut exists. Static2Dynamic's Hugging Face repo (thethomasboyer/Static2Dynamic) has a
"chromalive" folder: a trained 98 MB diffusion model, train/test sample lists, and a pseudotime
parquet named heldout_test_trajs_2_doses_10_11 ... 380px patches. So Boyer already cut 380 px
single-cell patches from this dataset, 3 channels as PNG, and held out tracked trajectories at
the two highest doses. If the parquet lists patch paths with well/FOV/time/track ids, that is
their tracking for free, and training on the same patches makes any comparison with Static2Dynamic
exact. Need to look inside those parquet files (about 33 MB) before deciding how to cut patches.

Plan for the first real step, pending the image download: one control well and one high-dose
well, all 4 FOVs, 13 live frames + terminal, 4 channels: 2 x 4 x (13 x 4 + 2) = 432 TIFF, about
3 GB. Segment nuclei (Cellpose, CPU is fine at this size), track, register the AnnexinV frame
with the offsets from the repo, measure AnnexinV per track. That gives real videos with fate for
a few hundred cells and is enough to see whether the question is answerable before 45 GB.

Read the Static2Dynamic Datasets section (preprint text). Their ChromaLive use is not single-cell
and not tracked: 48,000 random 380x380 crops over whole fields of view, rescaled to 128x128, from
the 24 field movies of the two highest doses only (156 and 78 nM), where every cell dies. The
563 true-versus-generated trajectory pairs come from segmenting cells inside those crops
afterwards. Lower doses were used only to test pseudotime ordering. So the Hugging Face
parquet will not contain tracking, and the pretrained model has only ever seen a one-fate regime.

That sharpens the question. The branching lives at the intermediate doses (about 5 to 39 nM),
where some cells die by 6 h and some do not, and AnnexinV says which. Static2Dynamic validated
itself where there is one future; static2fate works where there are two. Same data, same
preprocessing style (so the comparison is fair), different regime.

Consequence for the subset: not control plus top dose, but control, one top dose, and two
intermediate doses, e.g. C-02 (0), C-07 (9.77), C-08 (19.53), C-10 (78.13): 4 wells x 4 FOV x 54
files, 864 TIFF, about 6 GB.
