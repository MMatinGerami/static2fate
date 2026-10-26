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


## 2026-10-21

Does a video encoder see fate earlier than one frame? scripts/07_vjepa.py: frozen V-JEPA 2
(ViT-L, facebook/vjepa2-vitl-fpc64-256), clip = frames 1..k at native 128 px, tokens mean-pooled,
same leave-one-field-out logistic regression, 2,735 cells at 4.88 to 39.06 nM, 35 fields. AUROC
within dose, averaged over the four doses; CIs by resampling fields within dose (1,000, seed 0).
Run on MPS, 10 min (smoke test: MPS and CPU agree to 7e-5 on pooled features).

| last frame | V-JEPA clip | DINOv2 mean of 1..k | DINOv2 frame k | handcrafted frame k |
|---|---|---|---|---|
| 30 min  | 0.53 | 0.52 | 0.50 | 0.54 |
| 90 min  | 0.52 | 0.53 | 0.55 | 0.56 |
| 150 min | 0.60 | 0.57 | 0.58 | 0.63 |
| 210 min | 0.64 | 0.63 | 0.67 | 0.68 |

V-JEPA minus handcrafted at 210 min: -0.045 (-0.067 to -0.021). V-JEPA minus DINOv2 averaged
over the same frames at 150 min: +0.036 (0.007 to 0.067). So the video model holds some temporal
information that frame-averaging loses, but not more than the latest frame alone. Off the shelf it
does not detect fate earlier.

Why it might be unfair to V-JEPA, to test before concluding: mean pooling discards the temporal
order the model encodes; frames 1..k includes early frames that carry no fate signal and dilute
the pooled vector (a sliding window of the last 4 frames would be fairer); 128 px is half its
training resolution and fluorescence is far from its training data. These are the representation
and domain-shift questions, not reasons to drop it.

## 2026-10-21, first generative model

Conditional flow matching (rectified flow), 64 px crops, 3 channels, conditioned on frame and
fate with classifier-free guidance (fate dropped 10%). UNet with FiLM, 9.4 M parameters, EMA
0.999, batch 64, AdamW 2e-4, 45 min on MPS (5,578 steps), seed 0. Held out E-06 to E-09 (one
replicate well per two-fate dose). MPS smoke test first: 0.74 s/step vs 5.5 s on CPU, finite.

Same-noise pairs: identical at 0 to 2 h, separate from 3 h (condensed bright nucleus, green
cytoplasm in the dies branch). Nobody gave it the timing.

Held-out check on two nuclear readouts. Hoechst brightness: real dies 0.02, 0.02, 0.06, 0.19,
0.26, 0.31, 0.35 at 0 to 6 h; generated dies -0.06, -0.06, 0.01, 0.22, 0.34, 0.36, 0.43;
survives flat in both. Timing right, end overshoots. Nucleus area: wrong. Generated dies 81 px at
0 h against 184 px real, survives also too small. Could be the fixed threshold on a different
intensity scale, or the model really shrinking dying nuclei too early; not settled.

Not yet: starting from a real cell (inversion), video consistency, 128 px.

## 2026-10-24

Nucleus size mismatch, diagnosed. The fixed Hoechst threshold (-0.2) was the main problem:
generated images are about 0.1 dimmer in Hoechst than real ones (p99 0.07 vs 0.16 for
survivors at 0 h), so fixed-threshold masks dropped nucleus pixels in generated cells. With the
Otsu object nearest the crop centre (now src/static2fate/readouts.py), median areas real/gen:
survives 559/584, 636/706, 672/780 px at 0, 3, 6 h; dies 533/466, 405/295, 219/148. So the
model reproduces the shrinkage of dying nuclei over time, but exaggerates it (dying 13 to 32%
too small, survivors 5 to 15% too large) and is slightly dim overall. Not an artefact any more,
just a smaller real bias.

Starting from real cells. Added flow.invert (backward Euler, 50 steps, fate left out) and
scripts/11_invert_branches.py: invert the held-out cell's 0 h frame, regenerate it (round trip),
then generate 3 to 6 h under each fate from the same noise. Dry run on 64 cells: round trip pixel
MSE 0.002, readouts within 0.03 to 0.14 SD. Surviving cells: the survives branch is closer to
their real future 83 to 92% of the time. Dying cells: the opposite, 17 to 33%. Guess: many
"dies" labels are AnnexinV-only cells that still look normal at 6 h, while the dies branch
condenses hard. Full run on 816 cells split by condensed vs AnnexinV-only to check.

Full inversion run, 816 held-out cells (12 fields). True-fate branch closer to the real future
than the other branch, balanced over fates: 0.62, 0.67, 0.71, 0.73 at 3 to 6 h. By group:
survives 0.77 to 0.91; dies condensed 0.58 to 0.73; dies AnnexinV-only 0.41 to 0.45. My guess
held: AnnexinV-only dying cells still look like survivors and the dies branch condenses them.

But that comparison flatters the model: the survives branch is almost a frozen copy of 0 h, so
it wins for survivors by default. Against the copy-0h baseline (true-fate branch closer to the
real future than the 0 h frame is): survives 0.44 to 0.49, no gain; dies condensed 0.46, 0.57,
0.67, 0.72 (6 h CI 0.54 to 0.80, field bootstrap); dies AnnexinV-only 0.33 to 0.37, worse than
copying. Median distance at 6 h for condensed dying cells 2.3 (model) vs 3.6 (copy). Identity:
generated future closer to the cell's own real future than to another same-fate same-dose cell's
0.73 at 3 h falling to 0.63 at 6 h.

Looking at the examples: real dying cells end as a red and blue condensed body; generated dying
cells turn the cytoplasm green. Timing right, colour wrong. Population readouts redone with the
Otsu readout: generated dying nucleus 150 px vs 219 real at 6 h, Hoechst 0.47 vs 0.22.

Context model. The inversion results suggest the model needs to know how far along a cell is.
UNet(context=True): one observed frame of the cell as three extra input channels, plus its
frame index; trained on pairs (context frame 0 to 6, target frame after it), 60 min on MPS,
7,162 steps, same split. Evaluated with scripts/14_eval_ctx.py, which also re-scores the
inversion run; the inversion numbers come out identical to the table above (the copy-baseline
comparison was an inline command before, now it is a script).

Beats copy-0h, context 0 h vs inversion, paired by cell: condensed dying +0.14 at 4 h (CI 0.06
to 0.24), +0.11 at 6 h (0.03 to 0.21), reaching 0.82; AnnexinV-only no change (0.33 to 0.42);
survivors worse, -0.13 at 3 h (-0.17 to -0.08). So my hypothesis was wrong for the cells it was
meant for: the AnnexinV-only cells are still condensed by the dies branch. Survivors lose
because a fresh noise sample changes the cell, while inversion keeps it. One sample per cell
penalises any stochastic model against a copy; should use several samples and an energy score,
or invert under the context model. Dying nucleus at 6 h still 163 px vs 219 real, Hoechst 0.47
vs 0.22.

In the example grid one "AnnexinV-only" cell is round with bright chromatin at 2 h and flat
again at 3 h. Probably a division. Mitotic cells round up and condense their chromatin, so in
one frame they can look like early apoptosis. Need to flag divisions (track splits, or a
rounding then flattening within 1 to 2 frames) and see how many labelled cells are affected.

GPU: Kaggle job ready (kaggle/train_kaggle.py, 128 px on one T4 and 64 px for 8 h on the
other), data export 1.7 GB. Waiting for the account to be set up.

## 2026-10-25

Divisions. Wrote static2fate/divisions.py: condensation (area < 0.6x and Hoechst > 1.4x the
previous 3 frames), recovery within 3 frames (area 1.5x, Hoechst 0.8x the condensed frame), and
a sister track born within 70 px with a similar nucleus. Only tracks with all 13 live frames
(4,670 of 6,676 labelled). Strips in results/qc: most flagged divisions really show two
daughters; a few are crowded crops or dying cells.

Counts: divided 21 of 2,724 survivors, 5 of 1,416 condensed dying, 4 of 530 AnnexinV-only.
So the "AnnexinV-only cell that rounds up and flattens" in the context examples was a rare
case, and divisions are not why the dies branch fails for that group.

Other route: mitosis at the end of the movie reads as "condensed". 31 condensed dying cells are
condensed only in the last hour and AnnexinV negative. By eye maybe a third look like mitosis
(flat cell rounding up suddenly), the rest shrink progressively and bleb. Onset frame does not
separate them (some clear apoptosis also crosses the threshold only at frame 13). Upper bound
40 of 1,946 dying cells.

But 0.8% dividing survivors in 6 h is far too low for HeLa (doubling about a day). Random
control survivors: 0 of 16 divide by eye. Nucleus count per field, frame 13 over frame 1: 1.06
to 1.17 at 0 to 9.8 nM, falling below 1 from 39 nM. So about a tenth of cells divide and almost
none of them keep a complete track. Births of new tracks away from the border are about 0.9
per cell, so most track starts are fragmentation, not division. The labelled set is selected
for non-dividing cells. Has to go in the limits; fixing it means mother-daughter linking.

Also checked a possible dark frame 2 seen in a few strips: median crop brightness per frame is
flat (Hoechst 17 to 19, 561 23 to 27); 488 rises from 17 to 24 over the movie in all cells. Dark
crops increase late (shrinking dying cells). No acquisition artefact.

Energy score. One sample per cell penalises a stochastic model against a deterministic copy.
scripts/17_energy_score.py: 8 samples per held-out cell and frame (context model, 0 h frame,
true fate), energy score in the same standardised readouts. First version averaged the spread
term over all pairs including i = j, which biases the score against the model; switched to
distinct pairs (unbiased for any number of samples). DOI of Gneiting and Raftery checked on
Crossref.

Worry before the full run: the energy score credits spread, so maybe any noisy forecast beats
copy. Added "copy + typical change" (0 h readouts plus the change of a random training cell
with the same fate) and "random same-fate cell". The 32-cell smoke test suggested the baseline
might beat the model.

Full run, 816 cells, 80 min on MPS. Model vs copy, energy score difference at 6 h: survivors
-0.48 (CI -0.57 to -0.41), condensed dying -2.25 (-2.55 to -1.52), AnnexinV-only -0.72 (-1.08
to -0.33); below 0 at every hour. Copy + typical change gets most of that: -0.41, -2.03, -0.70.
Model vs that baseline: condensed -0.24, -0.23, -0.34, -0.22 at 3 to 6 h (CI below 0 at 3 to
5 h), survivors -0.04 to -0.07 (CI below 0 at 4 and 6 h), AnnexinV-only about 0. Even the mean
of 8 samples as a point forecast beats copy for survivors (-0.05 to -0.12). So the "worse than
copy for survivors" result was the one-sample penalty, and what the images add beyond the
average course of each fate is mostly in visibly dying cells. Smoke test numbers were noise.

Kaggle runs scored. Training (8 h each on one T4, float16): 64 px for 98,298 steps (seed 1),
128 px batch 32 for 50,999 steps. Evaluation needed the batch halved at 128 px (out of memory
with 32 cells x 8 samples) and the 128 px cells split over the two GPUs (about 4 h per half;
readouts standardised on all 816 cells in both halves, then joined).

Energy score paired by cell against the 1 h Mac model: 64 px 8 h, survivors -0.02 to -0.06 (CI
below 0), condensed dying +0.11, +0.13 at 3, 4 h (CI above 0), AnnexinV-only about 0; 128 px,
survivors -0.01 to -0.02, condensed +0.08, +0.10 at 3, 5 h, AnnexinV-only -0.08, -0.09 at 3,
5 h. All within 0.15 and mixed in sign. Loss curves: all three drop to about 0.1 by 2,000
steps and sit there (last-20 mean 0.097, 0.084, 0.082). So 14x the steps or 4x the pixels
buy nothing visible in four readouts. Either the readouts are too coarse to see the difference
(plausible: they are computed after averaging 128 px down to 64) or the model is data or
conditioning limited. A finer metric would settle which.

Kaggle quota used this week: about 17 h of the 30 (training 8.5, evaluations 2.3 and 4.5,
plus the aborted runs). The train job's first version used batch 64 at 128 px and ran out of
memory.

Finer comparison, first model. Saved 8 futures per held-out cell as images (19), then scored
them (20) in three spaces, each standardised per dimension on the real held-out frames 6 to
12: the four readouts, nine crop-level nucleus measurements (features.crop_features: area,
eccentricity, solidity, nucleus and 8 px ring mean per channel), and DINOv2-base CLS (768 d).
Two baselines: "copy + typical change" as before, and a random same-fate training cell at the
same frame (M draws), which stays on the manifold of real images. Both at 64 px (everything
averaged down) and 128 px (64 px samples upsampled, real crops native).

1 h model, 6 h, model minus random real cell: readouts -0.13 condensed, -0.12 survivors,
+0.18 AnnexinV-only; handcrafted -0.15, -0.07, +0.04 (CIs mostly crossing 0); DINOv2 +5.4,
+3.0, +5.0 (CIs 2.7 to 6.4, larger at 128 px). So in DINOv2 space a random other real cell
of the same fate is much closer to the real future than the model's samples. Also the copy of
the cell's own 0 h frame is 17 units farther than a random real cell at the right frame, so
DINOv2 distance here is mostly time after drug plus realism, not cell identity. Linear probe
real vs generated (21, leave-one-field-out): DINOv2 AUROC 1.000 at 64 and 128 px,
handcrafted 0.59 (0.72 at 128 px, where the upsampling is a giveaway), readouts 0.51.
The generated images are telling in a way the readouts cannot see. Drift baseline in DINOv2
space is off-manifold (model beats it by 1 to 2.4) and is not the right comparison there.

All three models in feature space (Kaggle generation 2 h for 64 px, 4.6 h for 128 px; 128 px
runs 24 s per cell with 8 samples). Readouts and handcrafted: Kaggle models minus the 1 h
model within -0.12 to +0.17, mixed signs; nothing new. DINOv2 (paired, 6 h): 64 px 8 h -2.3
survivors, -4.4 condensed, -3.6 AnnexinV-only at 64 px and -2.2, -4.1, -3.5 at 128 px; 128 px
model -1.9, -3.3, -3.1 and -2.7, -3.7, -4.0. All CIs well away from 0, same sign at every
hour. Against a random same-fate real cell (6 h): 1 h model +3.0 to +6.6; 64 px 8 h +0.7 to
+1.2 at 64 px, +2.0 to +3.2 at 128 px; 128 px model +1.1 to +2.0 at 64 px, +1.6 to +2.7 at
128 px. At native 128 px the 128 px model is closer than the upsampled 64 px 8 h model for
survivors (1.58 vs 1.99) and AnnexinV-only (2.59 vs 3.24), not condensed (2.74 vs 2.23,
overlapping CIs). Downsampled to 64 the 64 px 8 h model is the closest, probably its 98k
steps against 51k at batch 32.

Probe real vs generated, DINOv2: 1.000 / 0.997 / 0.995 (1 h, 64 px 8 h, 128 px 8 h) at 64 px;
1.000 / 1.000 / 0.996 at 128 px. Handcrafted 0.59 / 0.58 / 0.63 at 64 px and 0.72 / 0.75 /
0.61 at 128 px (the upsampled models give themselves away there). Readouts 0.51 to 0.55.

So the readout comparison of 25 Oct morning was blind: longer training made the images more
real by a large margin in DINOv2 terms, and resolution helps at native resolution. But a
frozen encoder still separates generated from real without error. Two things to keep apart
from now on: realism (how far from the real manifold) and identity (whose future this is).
DINOv2 distance as used here is the first; it cannot answer the second because time after
drug and realism dominate it. The drift baseline is not usable in DINOv2 space.

Identity. evaluate.rank_of_own + scripts/22_identity.py: candidates are all real held-out cells
of the same fate at the target frame (571 survivors, 245 dying); percentile rank of the query
cell's own real future, 0 = nearest. Bug on the first run: `tab.query` is the DataFrame
method, so the plot was empty; use tab["query"].

6 h, DINOv2 at 64 px: models 0.43 to 0.51 (near chance), copy 0 h 0.35 survivors, 0.33
AnnexinV-only, 0.50 condensed; real 3 h frame 0.25 to 0.29. Models minus copy, paired:
+0.09 to +0.16 for survivors and AnnexinV-only (CIs exclude 0), -0.07 to -0.01 condensed.
Readouts the same pattern, and copy is worse than chance for condensed cells (0.59) because
they change so much. 64 px 8 h minus 1 h: -0.03 survivors (CI -0.04 to -0.02), -0.05 condensed
(CI crosses 0); 128 px minus 64 px 8 h +0.02 to +0.03. Top-1: models 0.2 to 1%, copy 1 to 5%,
real 3 h 5 to 34%.

So the sampler loses the cell: the futures are plausible same-fate cells, not this cell. The
inversion model kept identity by construction (round trip MSE 0.002) and its survives branch
was a near copy; the context model got the dynamics of condensed cells better but at the cost
of identity. These are the two ends of one trade-off, and the next model has to hold both:
noise from inverting the context frame, or stronger conditioning on it.

## 2026-10-26

Trailer and tag. A commit from another session on 25 Oct (CI workflow, citation, v0.1.0
release) carried a co-author trailer; rewrote the six messages since, force-pushed, and moved
the v0.1.0 tag (annotated, release still attached) to the rewritten commit. The old commit is
unreachable from any ref now.

Keep the cell, first try. flow.sample takes x_start and s0: integration starts at s0 from
(1 - s0) noise + s0 x_start (SDEdit). 23_keep_cell.py: 64 px 8 h model, 0 h image as both
context and start, s0 0.3 / 0.5 / 0.7, 4 samples per cell (27, 19, 12 min on MPS; the 0.7 run
was interrupted by sleep overnight and finished at 06:01). Scored with 20, 21, 22 under
STATIC2FATE_TAG=_keep and the new 24_tradeoff.py.

6 h, identity rank (DINOv2, 64 px) / dynamics (readout energy score minus drift baseline):
from noise 0.44 / -0.10 survivors, 0.43 / -0.17 condensed, 0.48 / +0.03 AnnexinV-only;
s0 0.3 0.38 / +0.14, 0.48 / +1.77, 0.39 / +0.48; s0 0.5 0.38 / +0.27, 0.48 / +1.80,
0.37 / +0.61; s0 0.7 0.38 / +0.34, 0.49 / +1.90, 0.36 / +0.70; copy 0.35 / +0.41,
0.50 / +2.02, 0.33 / +0.74. So partial noising moves straight from the model's corner to
the copy's corner; the first 30% of the path already removes nearly all the dynamics. For
condensed dying cells the starting layout (large flat nucleus) cannot be undone from s0 = 0.3.
Probe: handcrafted 0.69 to 0.72 (from 0.58), readouts 0.63 (from 0.54): the noised starts
leave a trace. DINOv2 1.0 as always.

Conclusion: identity cannot come from the initial state in a straight-path flow; it must come
from the conditioning. Next model: context features injected at every UNet level (or
cross-attention), plus an identity term in the loss, then the same trade-off figure.

Paired flow, first two noise levels. flow_matching_loss takes a source; 25_train_paired.py:
source = context image + sigma x noise, target = later frame of the same cell, fate and frame
conditioning as before, 45 min on MPS each (5,674 steps at sigma 0.5, loss 0.05; 5,710 at
sigma 1, loss 0.11). Sampling starts from the real 0 h image + sigma x noise (19 reads sigma
from run.json). 4 samples per cell, scored with TAG=_paired next to the 8 h noise-start model
and the SDEdit s0 0.5 sampler.

6 h: dynamics (readout energy minus drift) sigma 0.5 -0.06 survivors, -0.13 condensed, +0.11
AnnexinV-only; sigma 1 -0.05, -0.23, +0.09; noise-start -0.10, -0.17, +0.03. So training on the
paired path fixes what SDEdit lost: the model condenses the nucleus from a start that still
holds the cell. But identity did not come with it: rank of own future in DINOv2 0.47 to 0.49
(noise-start 0.43 to 0.48, copy 0.33 to 0.35), and the same in readouts and handcrafted at
every hour, 3 h included. The example panel shows why: the layout of the 0 h cell is kept, but
the model repaints texture and cytoplasm, and that is what the features respond to. Realism
is worse than the 8 h noise-start model (DINOv2 model minus random cell 4 to 7 against 0.7 to
1.4), at the level of the 1 h noise-start model, which is a fair comparison at 45 min of
training. sigma 0.5 and 1 behave the same, so the noise is already large relative to the
image (pixel std about 0.4): the start carries little. Training sigma 0.2 now.

Paired flow, sigma 0.2, and per-cell skill. sigma 0.2: 5,604 steps, loss 0.02; identity rank
0.47 to 0.50, same as the other sigmas, dynamics -0.05 / -0.04 / +0.04 (survivors, condensed,
AnnexinV-only), realism worst of all (DINOv2 model minus random cell 6.4 to 9.0; handcrafted
probe 0.73). So three sigmas, one answer: the paired path keeps the layout and the dynamics
but not what the feature spaces call the cell.

Then a looser test, 26_percell_skill.py: Spearman across cells, within group and hour, of the
real readout at 6 h against the prediction (mean over samples). Noise-start 8 h model:
nucleus area 0.50 / 0.35 / 0.48 (survivors / condensed / AnnexinV-only), copy 0.50 / 0.42 /
0.23, real 3 h 0.66 / 0.52 / 0.62; cytoplasm red 0.55 / 0.38 / 0.40 vs copy 0.54 / 0.19 /
0.32. So the model does track the cell's own measurements, as well as the copy and better
where the copy is stale (dying cells). The ranking was answering a different question:
"is this the nearest real image", which texture decides. Paired models: nucleus area 0.39 to
0.41 for survivors, below the noise-start model; the start adds nothing.

Revised reading of the whole identity thread: the noise-start context model carries the
cell's measurements but not its texture; starting from the cell (SDEdit, paired flow) does
not fix texture and costs skill. Realism is the thing to work on next. The README identity
section's conclusion was too strong and is corrected.
