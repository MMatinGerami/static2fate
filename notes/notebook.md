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
alone gives 0.79 on the 6,676 cells of all 30 wells, 0.84 in the four-well pilot) and should
not be quoted.

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

## 2026-10-27

Realism: two continuations of ctx64_long on Kaggle (continue_kaggle.py, one T4 each, 8 h,
seed 2): ctx64_more (same loss, 101k more steps) and ctx64_perc (plus 0.5 x VGG16 feature
distance between the x1 estimate and the real image, 82k steps; the term was 0.15 against an
MSE of 0.08). Generated 8 samples per held-out cell for both on the Mac (80 min each) and
re-ran 20, 21, 22, 26 and 24 with tag _realism.

Nothing moved. MSE over the last 20 logged steps: 0.084 (ctx64_long), 0.086, 0.085; the
loss had already flattened before the continuation started (first 20 logged steps of the
continuation 0.083). From the same noise draw the three models give nearly the same image:
mean |diff| 0.036 (+8 h) and 0.039 (perceptual) against 0.16 between two samples of one
model. Probe AUROC DINOv2 0.997 for all three; handcrafted 0.58 / 0.58 / 0.61; readouts 0.54
/ 0.54 / 0.56. DINOv2 energy score minus random real cell at 6 h (survive / condensed /
AnnexinV-only): base 0.68 / 0.97 / 1.40; +8 h 0.68 / 0.80 / 1.20; perceptual 1.39 / 0.67 /
1.64, so the VGG term made survivors look less real to DINOv2 (CIs 1.11 to 1.63 against 0.46
to 0.89). Dynamics, identity (0.41 to 0.48) and per-cell nucleus-area skill (0.50 / 0.51 for
survivors) unchanged within CIs; condensed skill 0.24 for the perceptual model against 0.35
(CIs overlap).

Reading: at this capacity (ch 64, 1,919 training cells) the model has converged, and the
pixel loss plus a VGG term does not buy texture that DINOv2 accepts. Next for realism is
capacity and data (wider UNet, all labelled two-fate cells, or a latent autoencoder), not more
hours. Kaggle quota this week is spent; the new week starts Saturday.

## 2026-10-30

Before choosing between a wider model and a latent model, checked whether the context model is
short of data at all: 27_fit_check.py, flow matching MSE with fixed draws (16 per cell, fate
given, no dropout, no augmentation) on train and held-out cells. Held-out is slightly lower
than train for every model (1 h 0.086 vs 0.090; 8 h 0.081 vs 0.084; +8 h 0.081 vs 0.083; SE
about 0.001), in every flow-time bin. So no memorisation, and no gain past 8 h: capacity-limited,
not data-limited. The bin s in (0.9, 1] carries most of the loss (0.30), the fine-detail end,
which fits the texture gap seen in DINOv2.

Held-out lower than train is a little odd; the held-out wells are one replicate per dose, so
it is probably the mix of cells (the two sets are not exchangeable), not something to read into.

Next run (Kaggle, both T4s, 8 h): ch 128 (35 M parameters, about 3x slower per step) on the same
1,919 cells, and ch 128 on all ten doses (5,860 training cells, 42% dying; held-out wells
unchanged). Snapshots every 2 h to plot held-out loss against steps, since 8 h at ch 128 is
only about a third of the ch 64 steps. Loader now converts in chunks so all doses fit in RAM;
it returns the same tensors as before (checked). Kaggle dataset re-exported with all doses
(4.27 GB).

## 2026-11-01

Kaggle quota reset; wide-model job (ctx64_w128, ctx64_w128_all) pushed at 10:03, running.

While it runs: the video model. Measured first what independent draws cost: on the held-out
cells the frame model (ctx64_long) moves 0.16 to 0.18 per pixel per hour against 0.12 to 0.15
for real cells, its nucleus area jumps 150 to 190 px per hour against 100 to 120, and over the
three hour steps the area changes direction 1.2 to 1.3 times (2 possible) against 0.65 for real
condensing cells. So the frame model's "futures" of one cell are not one future.

Video model: context UNet on every frame plus temporal self-attention over the 7 frames (3 to 6 h)
after each stage, zero-initialised output projections so it starts as the frame model (test
checks the equality to 1e-5). 10.3 M parameters. Trains on MPS at 1.7 s per 16 clips; 6 min
smoke run falls to loss 0.20. Generation is slow on the Mac (8 cells x 8 clips in about 3 min,
so 816 cells would be 4 h), so the Kaggle job samples after training. 29 writes clips plus the
4-frame samples layout, so the whole scoring chain (20, 22, 26) and the new 30 run on it. Job
staged in kaggle/video/, to push once the wide job frees the GPUs (about 18:30).

Evening: both Kaggle jobs done (wide 18:12, video 18:40); gen kernel for the wide models'
samples (8 per held-out cell, one model per T4) ran 18:20 to 21:45. Video clips were sampled
inside the video job.

Video model, temporal (30_temporal.py, tag _video): the jumps are gone. Nucleus area change
per hour 98 / 96 / 105 px (survive / AnnexinV only / condensed) against real 102 / 119 / 114
and frame model 153 / 187 / 176; reversals for condensing cells 0.74 against real 0.65 (CI
0.56 to 0.83) and frame model 1.22. At 30 min steps too. Two misses: AnnexinV-only dying
cells get the condensing cells' monotone shrink (0.79 reversals, real 1.07), and survivors
move a bit less than real (0.117 vs 0.138 per pixel). The all-doses video model is too still
(dying cells' nucleus changes 54 px per hour, half the real), presumably because at 78 and
156 nM cells are dead before 3 h and the clips of dying cells are static.

Per frame (chain 20 to 26 at 6 h): realism a step behind the frame model (DINOv2 distance to
a random real cell 1.45 / 2.91 / 2.58 against 0.68 / 0.97 / 1.40), probe AUROC unchanged,
identity at chance as before, per-cell skill for nucleus area in dying cells lower (0.18,
0.22 against 0.35, 0.48; the all-doses model keeps 0.48, 0.34). The video run saw half the
images of the frame run (28k steps of 16 clips in 7 h). Temporal consistency bought with some
single-frame quality at this budget; a longer video run or a video run initialised from the
frame model (--init) would show whether that is just budget.

Figure fix in 30: example clips now one block per cell (rows sources, columns hours), so the
images are not postage stamps.

Wide models (tag _wide, samples from the gen kernel): nothing moves. Every score within the
64-channel model's interval (DINOv2 vs random real cell 0.67 / 0.92 / 1.29 against 0.68 /
0.97 / 1.40; probe 0.996; identity 0.44 / 0.43 / 0.49; nucleus-area skill 0.51 / 0.37 / 0.45).
All doses: same, and AnnexinV-only dying cells worse (2.01). The telling number: from the same
noise the 128-channel model's sample differs from the 64-channel model's by 0.030 per pixel
(0.041 all doses), against 0.162 between two samples of one model. Different width, seed and
training set, same image. The velocity field is set by the data; capacity is not the lever.
Next lever is the representation (latent autoencoder), not the UNet. Fit check on the wide
models and their 2 h snapshots started on the Mac (numbers under 2 Nov).

22:50: second video round pushed (kaggle/video2/, 9 h per T4 then sampling): video64_init
starts from ctx64_long with the temporal layers at zero (28 now accepts a frame model as
--init; test checks it equals the frame model at step 0), video64_cont continues video64 to
16 h in all. If the per-frame gap closes in either, it was budget; if not, the coupling costs
something. Week's Kaggle quota after this: about 2 h.

## 2026-11-02

13:00: the second video round hit Kaggle's 12 h session limit. Both models trained the full
9 h (video64_init 39429 steps, video64_cont 38700 steps on top of video64's 28265), but
clip generation for 816 cells at 8 clips takes about 3.4 h on a T4, so it was cut at
704 (init) and 692 (cont) cells with no track files written. Kept the written cells as
clips_0_704.npy / samples_0_704.npy (and 0_692) with tracks from train.load("test"), in
the file order 29_generate_clips.py would have used; the remaining cells are generated on
the Mac as a second part. Lesson for kaggle/*: training time + 3.4 h sampling must stay
under 12 h, so 8 h of training is the ceiling with sampling in the same session.
Temporal comparison on the written cells (10 of 12 fields): both new models keep the
smoothness of the 7 h video model (nucleus area change per 60 min in condensing cells
109 / 94 px for 16 h / from frame model vs 105 video 7 h, 114 real, 176 frame model).

18:00: the Mac was swapping (44 GB of swap on 24 GB of RAM): the fit check held 28 GB of
unified memory and the clip generation 16 GB, both from the Apple GPU allocator caching
blocks across models and batches, and neither had progressed for hours. Fixed both
scripts: allocator capped at half the recommended working set
(torch.mps.set_per_process_memory_fraction), emptied after each model / batch, batch 64 in
the fit check and one cell per batch in the generation on the Mac. Capped, a wide model's
fit check takes 6 min (allocator 4 GB) instead of over an hour, and clip generation about
15 s per cell. Rule from now on: one GPU job at a time on the Mac.

19:40: fit check on the wide models, capped allocator, 39 min for 11 models. The 128-channel
model ends on the same held-out loss as the 64-channel one (0.081 overall, 0.304 at flow time
0.9 to 1, both to the third decimal) and moves by 0.002 over its last 4 h; the all-doses
model lands on the same numbers (scored on the two-dose training cells). Held-out loss at
or below training loss for every model and flow time. So the floor is the data's, not the
model's: same floor from 10 M and 35 M parameters, 1,919 and 5,860 cells. Written into
docs/results.md under "Twice the width changes nothing".

20:10: second video round scored (tag _video2). It was budget: the video model started from
the 8 h frame model (9 h) matches the frame model on every per-frame score (DINOv2 minus
random cell 0.81 / 1.01 / 1.55 vs 0.68 / 0.97 / 1.40, nucleus-area skill in dying cells
0.33 / 0.39 vs 0.35 / 0.48, CIs overlapping) and keeps the smooth clips (reversals 0.76 in
condensing cells, real 0.65, frame model 1.22). The 7 h model continued to 16 h closes about
half the gap and its survivors get too still (83 px per hour, real CI 93 to 111). Recipe
from now on: frame model first, then the video model from it. docs/results.md has the
subsection "Is the per-frame gap only training budget?", README row updated.

22:30: latent autoencoder started (src/static2fate/autoencoder.py, scripts/31_train_autoencoder.py,
scripts/32_reconstruct.py, tests/test_autoencoder.py, kaggle/ae/). KL autoencoder, factor 4,
4 latent channels (128 px -> 4 x 32 x 32), 7.1 M parameters, L1 + VGG16 perceptual + 1e-6 KL,
optional PatchGAN hinge term; trains on every frame of every dose (76k images), held-out
wells untouched. 32_reconstruct writes the held-out reconstruction in the samples layout so
the probe can score it: that is the ceiling of latent generation. Smoke run on the Mac,
3 min at batch 8 (681 steps): held-out PSNR 28.9 dB, L1 0.048, perceptual 0.22, Spearman
between real and reconstructed readouts 0.97 (nucleus area), 0.97 / 0.97 / 0.98 (Hoechst,
green, red), nucleus area ratio 1.00. Kaggle job kaggle/ae/ (5 h per T4: plain and with
GAN from step 5,000) waits for the quota reset on Saturday 7 Nov. Then: the flow and video
models on the latent (UNet with 4 + 4 input channels at 32 px, decode before scoring).

## 2026-11-07

00:10: Kaggle still refuses the autoencoder job ("Maximum weekly GPU quota of 30.00 hours
reached" at 23:12 UTC Friday); a loop retries the push every 20 min until the reset. Meanwhile
the latent flow models are written so both stages can run this Kaggle week:
flow.UNet and video.VideoUNet take in_ch (4 for the code; the context has the same channels,
so 8 input channels), sample and sample_video take clamp=False for latents;
static2fate.latent loads an autoencoder and encodes and decodes in batches with latent_std;
33_encode_latents.py writes train.npz and test.npz (cells, 13, 4, 32, 32) float16 plus an
equivariance check (PSNR of decoding a rotated or flipped code against the rotated or
flipped image, since augmentation happens on the code); 34_train_latent.py trains the frame
or the video model on the codes (same pairs, clip, EMA and dropping as 13 and 28, copies the
autoencoder next to the model); 35_generate_latent.py samples in the code and decodes to
128 px in the samples and clips layout. 30_temporal.py now averages 128 px clips down to
64 px before the readouts (they are defined at 64 px). Kaggle job kaggle/latent/: encode,
then frame model on GPU 0 and video model from scratch on GPU 1, 6 h each, then 8 futures per
held-out cell; takes the autoencoder from the static2fate-ae kernel (AE=ae128 or ae128_gan).

00:50: movies (scripts/36_movies.py, results/movies/, now tracked): per fate group three clean
held-out cells (track inside the field, no jump over 25 px; 57 / 15 / 8 candidates) with the
real course at 128 and 64 px next to three clips of the video model started from the frame
model, the generated columns showing the given 0 h frame until 3 h; a README strip with one
cell per group and two clips (1.6 MB); and the central quarter of E-08_F0001 (19.53 nM) over
13 frames with every labelled nucleus outlined by its fate (2.4 MB). GIFs from Pillow, 128
colours, 400 ms per frame. To redo with another generator: uv run scripts/36_movies.py <dir>.

01:05: per-field scaling check (scripts/37_field_scaling.py, 10 s for 89 fields on a pixel
subsample). Hoechst white point 280 to 693 across fields, Spearman 0.73 with the field's death
fraction and 0.72 with dose; within dose only 0.23. At the two-fate doses 357 (4.88 nM) to
493 (39.06 nM): the same nucleus 28% dimmer at the high dose from the scaling alone, and
field brightness correlated with fate through dose (0.68 over 35 fields). ChromaLIVE 488 white
point varies 3.4x between fields but not with death; 561 falls slightly with dose. Written up
in docs/results.md and the README limitations. The fix (plate-wide constants) means rebuilding
the crops, re-exporting the Kaggle dataset and retraining; proposed for the next round, not
done now, since the autoencoder job on the current crops is queued.

10:30: the autoencoder kernel finished (5 h per T4, 06:43; the waiter matched a lower-case
status and never downloaded, fetched by hand). ae128: 29.7k steps, held-out PSNR 33.3 dB, L1
0.028, perceptual 0.099, readouts Spearman 0.99 all four, area ratio 1.00; ae128_gan (hinge
term 0.1 from step 5,000): 31.1 dB, 0.035, 0.139, worse on every row, so ae128 goes
downstream. Outputs in results/kaggle/ae128 and ae128_gan (model, two snapshots, train_log,
recon/). Latent job pushed at 10:30 (static2fate-latent v1, AE=ae128, 6 h per model); a
case-insensitive waiter downloads it to the scratchpad.

11:10: reconstructions scored with the chain (tag _recon): the DINOv2 probe tells the 33 dB
reconstruction from the real cell at 0.995 (0.999 at 128 px), the same as the generated
futures (0.997), while the energy score puts the reconstruction 28 units closer to the cell's
own future than a random real cell and the generated future one unit further. So the probe
was not measuring distance from real. scripts/38_probe_sensitivity.py scores the real frames
under known changes (tag _probe): the frame itself 0.50; Gaussian blur 0.5 px 0.87 / 0.91
(64 / 128 px); blur 1 px 1.000 / 0.99; reconstruction + noise at the residual's level 0.995
(the noise energy is not what it reads). Handcrafted probe at 128 px: blur 1 px 0.74,
generated frames 0.75, reconstruction 0.67, blur 0.5 px 0.62. Conclusion written into
docs/results.md and the README table: the DINOv2 AUROC is saturated above 0.99 on sub-pixel
texture; models are compared from now on with the handcrafted probe at 128 px, the energy
scores and the readouts. Decision (Matin): the per-field scaling stays a documented
limitation; no crop rebuild.

## 2026-11-08

08:50: latent job downloaded (static2fate-latent v1, finished 17:57 on the 7th; waiter
fetched it at 18:03). lat_frame 235,642 steps in 6 h (batch 64, loss 1.35 to 1.31),
lat_video from scratch 89,037 steps (batch 16, loss flat at 1.31 to 1.34); 8 futures per
held-out cell decoded to 128 px, clips (816, 8, 7, 3, 128, 128). Encoding: latent_std 1.03,
equivariance PSNR plain 33.3 dB, rot90 28.9, flip 28.9 (augmentation on codes costs 4.4 dB).
Outputs in results/kaggle/lat_frame, lat_video, latents/info.json. Chain run on the Mac
(tag _latent, 25 min, after the Mac was back on mains): 20, 21, 22, 26, 24 with the pixel
frame model, the pixel video from the frame model and the ae128 reconstruction alongside,
30_temporal for the clips. Result: texture gap closed (handcrafted probe at 128 px 0.75 to
0.64 for both latent models, reconstruction 0.67; DINOv2 0.999, saturated), content not
(DINOv2 energy minus random cell 1.40 / 1.75 / 1.68 against 0.68 / 0.97 / 1.40, CIs apart for
survivors; identity rank 0.44 unchanged; nucleus-area skill in condensing cells 0.17 for the
latent frame model, 0.38 for the latent video). Latent video clips: pixel change 0.12,
nucleus-area change 101 / 107 / 102 px per hour (real 102 / 119 / 114), reversals
1.11 / 0.89 / 0.78 (real 1.11 / 1.07 / 0.65), best condensed dynamics of any model (-0.43).
In fig 14b the latent samples add or move neighbouring cells more often. Written up as
"Generating in the latent"; README rows updated. Next run: latent video from the latent
frame model, and a check of what the energy score counts.

09:40: second latent job pushed (kaggle/latent2/: latent video from the latent frame model
and the latent frame model continued, 6 h each, seed 4; Kaggle slug static2fate-latent-2,
running since 09:36, waiter in place). Kaggle week used so far: 5.4 + 7.2 + about 7.5 h.

10:30: masked energy score (scripts/39_energy_masked.py, 15 min on the Mac; first run had
the 64 px pixel samples pooled to 32 px and a 128 px radius on them, fixed and rerun with a
test). Whole crop: pixel frame 0.73 / 0.70 / 1.26 (survive / condensed / AnnexinV), latent
frame 1.42 / 1.28 / 1.61, latent video 1.58 / 1.49 / 1.83. Cell (r 40 px): pixel
1.32 / 2.09 / 2.81, latent frame 0.69 / 1.33 / 1.61, latent video 0.72 / 0.90 / 1.40. Nucleus
(r 24 px): pixel 2.10 / 0.57 / 2.76, latent frame -0.07 / 1.03 / 0.72, latent video
-0.23 / -0.24 / 0.71. The order reverses: the latent models get the cell and invent the
neighbourhood, the pixel model keeps the neighbourhood and blurs the cell. Written up as
"Where the DINOv2 distance lives"; README rows updated. From now on the masked score is
reported next to the plain one.

17:20: second latent job downloaded (static2fate-latent-2, ran 09:36 to about 16:10, 6.6 h;
week on Kaggle about 26.7 of 30 h). lat_frame_cont 228,822 more steps (464k in all, batch 64,
loss 1.313 to 1.307, flat), lat_video_init 84,261 clip steps from the latent frame model's
weights (loss 1.317 to 1.299, below the from-scratch 1.322). Chain on the Mac (tag _latent2,
24 min, 7 models) and 39_energy_masked rerun with all six models (18 min; existing rows
unchanged, point spacing in fig 18 now adapts to the model count). Neither run buys content:
DINOv2 energy minus random 1.51 / 1.75 / 1.59 (12 h frame) and 1.42 / 1.57 / 1.48 (video
from frame) against 1.40 / 1.75 / 1.68 and 1.60 / 1.76 / 1.77, all CIs overlapping; identity
rank 0.42 to 0.44 for all four; handcrafted probe at 128 px worse (0.66 and 0.68 against
0.64); masked to the nucleus the continued frame model is further in every group (condensed
2.21 against 1.03). The warm-started video model inherits the frame model's weak condensed
nucleus-area skill (0.20 against 0.38 from scratch) and loses the condensed dynamics score
(-0.01 against -0.43); its AnnexinV reversals are the closest to real of any video model
(0.98, real 1.07). Written up as "Do more steps or a warm start buy content?"; README row
and bullet updated; Next bullet now points at conditioning (keep the neighbourhood, draw the
cell). Mail: Aysan's README-only PR 5 on crc-singlecell-reference-mapping matches what was
agreed; its CI run was approved (not merged); JMB review of JMB-D-26-01418 due 22 Nov.

## 2026-11-09

03:45: anchored sampling started (scripts/40_anchor_neighbourhood.py). The samplers in
static2fate.flow and static2fate.video take a `project(x, s)` hook applied after every Euler
step; the anchored sampler resets the latent outside a disc around the crop centre to
(1 - s) x noise + s x context code, so at s = 1 the neighbourhood is the 0 h neighbourhood and
only the cell is drawn (radius 40 px as in the masked score, 10 code px, 31% of the code free;
also 56 px for the video model; edge softened over one code pixel). On 8 cells the output
holds the neighbours with no visible seam. Running on the Mac (0.6 min per 8 cells, about
1 h per model and radius): latent video r40 and r56, latent frame r40, M = 8, the same noise
as 35. 39_energy_masked.py now takes STATIC2FATE_TAG. Scoring to follow with tag _anchor.

09:10: anchored round scored (generation 03:44 to 06:22: latent video r40 and r56 about 65 min
each, latent frame r40 27 min; chain with tag _anchor 06:22 to 07:20, rerun with the composite
controls 07:23 to 08:55). The rim is held (anchored sample against the autoencoder's
reconstruction of the 0 h crop: 0.24 grey levels outside the disc). Whole-crop DINOv2 energy
minus random: anchored r40 7.9 / 8.6 / 6.9 against 1.6 / 1.5 / 1.8 free; masked to the nucleus
0.12 / 0.68 / 0.67 against -0.23 / -0.24 / 0.71 (CIs overlap); identity rank 0.39 / 0.44 / 0.39
(free 0.45 / 0.43 / 0.47, copy 0.35 / 0.50 / 0.33); handcrafted probe at 128 px 0.68 (0.64);
clips too still (pixel change 0.03 per hour, real 0.14). Controls (41_composite_control.py):
the real 6 h cell pasted into the 0 h rim is still no closer than a random cell on the whole
crop except for survivors (-1.4 / 3.3 / 1.3), so the neighbourhood must be predicted; the free
samples pasted into the rim are worse than the anchored sampler (7.7 / 12.6 / 11.9, hard seam).
Then split the energy score into its two terms (static2fate.evaluate.energy_terms,
42_energy_terms.py, and 39 now writes them): the distance to the real 6 h frame is 36 to 38
for every model since the 8 h one, 2 below a random cell and equal to the 0 h copy's for
survivors and AnnexinV-only cells; every between-model difference in "model minus random" so
far was spread (pixel 32.9, latent 31.0, anchored 18.1; random 38.3). The masked advantage of
the latent models at the nucleus is a distance (31.7 against 35.3 random and 34.7 pixel for
survivors). Against the 0 h copy the models win only for condensing cells (38.0 against 44.5
at the nucleus); AnnexinV-only nuclei are further than the copy for every model. Written up as
"Keeping the neighbourhood by hand" and "What the energy score was counting"; README rows
updated; Next points at the two terms as the measure and at AnnexinV-only cells (three fate
classes) as the model step. Mac on battery from 08:56 (at university), no further compute.
