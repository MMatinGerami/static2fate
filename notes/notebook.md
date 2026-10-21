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

## 2026-10-20, evening

Downloaded the four-well subset (864 TIFF, 16 fields, all intact) and Boyer's three ChromaLive
parquet files. The parquet confirms random crops: file names carry well, FOV, time, crop
coordinates, rotation and flip; the train set uses C-11, D-10, D-11, E-10, E-11 and C-10 (the
156 and 78 nM wells). No tracking in there, as expected.

Segmentation. Cellpose-SAM (cpsam_v2) is too slow here: the CPU run did not finish one
1900x1900 frame in 10 min; on MPS in fp32 at half resolution it takes 7.8 s per frame and finds
174 nuclei where the classical route finds 209. For the pilot I use Otsu on smoothed Hoechst plus
distance-transform watershed: 1 s per frame, masks look right on inspection
(results/qc/C-02_F0001_overview.png), one over-split on a touching pair in the crop I looked at.
Cellpose stays in the code as an option for a later check.

Tracking. trackpy with a constant-velocity predictor, search range 70 px, memory 2. Nearest-
neighbour displacement between frames is p50 13 px, p90 44 px, p99 89 px, so the obvious 30 px
range fragments nearly everything (1 full track out of ~210 cells). Yield of complete tracks
(frame 1 to terminal) is 23 to 54% of the nuclei present at frame 1, lower in the control wells,
which are denser. Good enough for a pilot; Ultrack (what the Way lab used) is the upgrade.

Registration, and a problem in the deposit. The terminal AnnexinV image is offset from live
frame 13 by up to 180 px; plain cross-correlation (normalization=None) recovers it and agrees
with the Way lab's offsets on fields 1 and 2. On fields 3 and 4 nothing matched, including the
Way lab's own offsets (r = 0.03). Cross-matching every live field against every terminal field
in all four wells gives the same answer each time: the terminal file named F0004 is the partner
of live F0003 (r = 0.48 to 0.60), live F0004 matches none of the four terminal files, and the
terminal file named F0003 matches none of the live fields. So one terminal field per well is
either mislabelled or from elsewhere, and the Way lab's F0003 offsets are in fact the F0003-to-
F0004 shifts. The code now matches terminal fields by correlation and gives no fate to live
F0004. Twelve of sixteen fields have ground truth. Worth telling the Way lab (Greg Way is the
PI; dataset first author Michael Lippincott).

First result. 952 complete tracks with a terminal AnnexinV value. Figure
results/figures/fig0_terminal_annexin_vs_area_by_dose.png. At 78 nM every cell is shrunken (median
terminal/initial nucleus area 0.33) and AnnexinV-high: one fate, which is the regime Static2Dynamic
trained on. At 0 nM nuclei keep their size and AnnexinV is low apart from a small tail. At 19.53 nM
the AnnexinV distribution is broad with a second mode and the nuclei are mostly not yet shrunken:
PS exposure without condensation, cells in early apoptosis next to cells that look untouched.
That is the two-fate regime the project needs, and it exists in the data. 9.77 nM is mostly
control-like with a tail. Fate labels: a two-component mixture on log AnnexinV (numbers below).

Caveat on the AnnexinV measure: it is the mean in an 8 px ring around the nucleus, which in dense
areas includes neighbours' membranes, and density differs by dose because dead cells detach.
Cell-level masks (ChromaLive channels) would be the proper cytoplasm; later.

Fate label numbers (complete tracks). Two-component mixture on log10 AnnexinV ring mean, pooled
over doses: component means 2.38 and 1.91 (sd 0.19, 0.27). Fraction in the high component:
0 nM 0.26, 9.77 nM 0.28, 19.53 nM 0.41, 78.13 nM 0.95. With a plain threshold at the control
90th percentile (240): 0.10, 0.11, 0.20, 0.60. Neither is right on its own: the mixture calls a
quarter of control cells "high", the threshold misses 40% of the 78 nM cells whose nuclei have
all collapsed (median area ratio 0.21 to 0.33). Nuclear shrinkage is the cleaner signal at the
lethal dose, AnnexinV the earlier one at 19.53 nM. Provisional label for now: AnnexinV above the
control 90th percentile OR terminal nucleus below half its initial area. To be replaced by
AnnexinV measured inside a cell mask from the ChromaLive channels. Saved
data/processed/fate_table_complete_labelled.parquet.

## 2026-10-20, night

Cell masks. The terminal image has no ChromaLIVE channels, so cell masks are drawn on live
frame 13 (seeded watershed from the nuclei on the log of the summed ChromaLIVE channels, cells
limited to 60 px from their nucleus) and applied to the registered AnnexinV image; the Way lab
did the same. QC in results/qc/C-02_F0001_cell_masks.png: masks follow the cells, a few
neighbours are merged in dense spots, cells drift 5 to 10 px between frame 13 and fixation.

Fate label, final for now: AnnexinV 90th percentile inside the cytoplasm mask above the control
wells' 90th percentile of the same quantity (311). Fraction apoptotic: 0 nM 0.10 (by
construction), 9.77 nM 0.21, 19.53 nM 0.37, 78.13 nM 0.89. Checks: at 78 nM Hoechst brightens
3.2x (condensation) and nuclei shrink to a third; in control the 17% of "shrunken" nuclei have
normal Hoechst, they are daughters of divisions, not deaths, so nuclear area alone would have
been a wrong label. 851 labelled cells.

Single-frame baseline (scripts/06_baseline.py, results/figures/fig2_single_frame_baseline.png).
128 px patches (Hoechst, CL488a, CL561) per cell per live frame; frozen DINOv2-base CLS
embedding or 11 handcrafted nucleus measurements; logistic regression, leave-one-field-out over
the 12 fields.
- Pooled over doses the curve sits on the dose-only floor (AUROC 0.84) until about 3 h; most
  of the pooled "prediction" is just knowing the dose. The honest number is within one dose.
- Within 19.53 nM, the two-fate well: AUROC 0.52 at 0 min (nothing before the drug, as it
  should be), 0.60 at 2 h, 0.69 at 3 h, 0.75 to 0.80 from 4 h on (handcrafted), 0.75 to 0.80
  (DINOv2). So from a single frame the fate becomes readable between 2 and 4 h, two to four
  hours before the AnnexinV readout. The Way lab's "100 minutes" was a population-level
  statement across doses; this is per cell within a dose.
- Handcrafted nucleus measurements match or beat frozen DINOv2 at every frame. Genovesio's
  "what are we really learning" point, reproduced on this problem without trying.

What this sets up. The generative model has to add something a classifier cannot: the two
counterfactual videos and the time at which they separate. The baseline curve above is the
calibration target for the commitment-time estimate: if the model says a cell is committed at
2 h, the classifier should start seeing it then, not later.

Caveats carried forward. 851 cells from 12 fields of one plate; the control false-positive rate
is fixed at 10% by the label definition; tracking keeps a quarter to a half of cells, with a
bias against dense regions and mobile cells; crops are 128 px and some cells are larger.

## 2026-10-21

All 30 wells. 6,480 files, 45 GB. Two files are truncated on the EBI server itself
(E-04_F0004_T0009 CL488a, 4 KB; C-06_F0001 terminal Hoechst, 1.4 MB), so E-04_F0004 is skipped
and C-06_F0001 has no fate. The terminal-field pattern holds in every well: F0001 and F0002 match
themselves, F0003 matches the file named F0004, F0004 matches nothing, 30 out of 30. That is a
systematic error in the deposit, not noise. 119 fields segmented and tracked, 89 with fate.

Label, revised. The AnnexinV readout stops rising at the top dose (0.65 positive at 156 nM
against 0.70 at 78 nM) while nuclear condensation keeps rising (0.54, 0.75, 0.81 at 39, 78,
156 nM): late apoptotic cells round up and the frame-13 cytoplasm mask no longer fits them. So
apoptotic = AnnexinV-positive OR condensed (terminal nucleus below half its initial area and
Hoechst more than doubled). Condensation is 1% in control, so this adds nothing there.
Fractions: 0 to 2.44 nM 0.11 to 0.17; 4.88 and 9.77 nM 0.19; 19.53 nM 0.31; 39.06 nM 0.74;
78 and 156 nM 0.91 and 0.92. Replicate wells agree within a few points at every dose
(e.g. 19.53 nM: 0.29, 0.31, 0.25). 6,676 labelled cells.

The two-fate band is 9.77 to 39.06 nM. Below it, apoptosis is at the control rate; above it,
nearly everything dies.

Single-frame baseline on all cells (results/figures/fig2_single_frame_baseline.png,
results/tables/baseline_auroc.csv). Within 19.53 nM: AUROC 0.51 at 0 min, 0.60 at 2 h, 0.69 at
3 h, 0.78 at 4 h, 0.84 at 6 h (handcrafted); DINOv2 0.51, 0.59, 0.66, 0.72, 0.80. Within
39.06 nM the curve is lower (0.79 at 6 h) because at 74% dying the "survivors" are partly label
errors. At 78 and 156 nM, chance, as it should be when there is one fate. Handcrafted beats
frozen DINOv2 from 2 h on at every informative dose. Pooled numbers are inflated by dose (dose
alone gives 0.85) and should not be quoted.

Reading of the curve: fate is not visible in a single frame before 1.5 to 2 h, and is largely
readable by 4 h. The branching generative model therefore has a concrete target: its commitment
time should fall in the 2 to 4 h window, and a model that separates the two futures at 30 min
is making it up.

## 2026-10-21, correction

Before sending the note I re-read the Way lab's preprocessing code. 1.pre_process_data/scripts/
2.sort_files_properly.py says in a comment that the terminal stain was acquired with six fields
per well, that timelapse FOV3 = terminal FOV4 and FOV4 = terminal FOV5, that terminal FOV3 and
FOV6 are unused, and that they renamed the files accordingly after checking visually
(verify_fov_mismatch.ipynb). So their ground truth is correct, and my earlier sentence that
"the Way lab's F0003 and F0004 ground truth might be wrong" was wrong; struck. What my
cross-correlation found is the state of the BioStudies copy, which carries the raw numbering
with only F0001 to F0004 deposited: deposit F0003 is the unused field, deposit F0004 is live
F0003's partner, and the F0005 files (live F0004's partner) are absent. Their documented mapping
and my matching agree in all 30 wells, which is also the best check I could ask for on my
registration code. Note rewritten as a deposit packaging issue.

