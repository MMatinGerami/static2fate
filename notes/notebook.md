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
