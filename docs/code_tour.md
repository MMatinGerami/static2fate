# Code tour

The code in the order of the [report](report.md): about 1,200 lines in `src/static2fate/` and
the scripts listed in [scripts/README.md](../scripts/README.md). For each file: what it does,
the choice behind it, and what it does not do.

## 1. From images to cells

**`src/static2fate/fields.py`** (one field of view).

- `Field.load` reads the 13 live frames of four channels and finds the endpoint image. The
  deposited endpoint files carry the microscope's raw field numbers, which do not match the
  live fields, and the stage has shifted by up to 180 px. `match_terminal` cross-correlates the
  last live Hoechst frame against every endpoint Hoechst image of the well at 4x downsampling,
  takes the best, refines the shift at full resolution, and gives up below a correlation of
  0.3. This reproduces the Way lab's own field mapping in all 30 wells.
- `segment_nuclei_classical`: Gaussian blur, Otsu threshold per frame, distance transform,
  watershed to split touching nuclei. About 1 s per frame. Cellpose is supported
  (`segment_nuclei_cellpose`) but takes minutes per frame on a laptop.
- `measure_frame`: per nucleus, area, eccentricity, solidity, and each channel's mean inside
  the nucleus and in an 8 px ring around it (the cytoplasm proxy).
- `link_tracks`: trackpy with a nearest-velocity predictor, search range 70 px, memory 2
  frames. The range comes from measured displacements (median 13 px, 99th percentile 89 px per
  30 min).

**`src/static2fate/cells.py`**. `cell_masks` is a seeded watershed on the summed ChromaLIVE
channels with the nuclei as seeds, limited to 60 px from the nucleus. `annexin_per_cell` takes
the 90th percentile of AnnexinV in the cytoplasm part of each mask. The masks are drawn on the
last live frame and applied to the registered endpoint image.

**`scripts/04_fate_labels.py`** (50 lines, the whole label). Threshold = 90th percentile of the
untreated cells' AnnexinV value; `condensed` = endpoint area below half and Hoechst above twice
the 0 h values; `apoptotic` = either.

**`scripts/05_crops.py`**. A 128 px window around the nucleus centroid at each frame, three
channels, scaled to uint8 with one pair of percentiles per field and channel over all frames.

What this part does not do: it does not link daughters to mothers, so dividing cells drop out;
it does not correct illumination; the per-field scaling makes intensities field-dependent.

## 2. Checking the label

**`src/static2fate/audit.py`** (three small functions).

- `sustained_onset`: the first frame from which a per-frame criterion holds until the end of
  the movie. A nucleus that rounds up for one frame and recovers (a division) has no onset.
- `cumulative_incidence`: fraction of cells past onset at each frame.
- `background_share`: the share of a label that its background rate alone would produce.

**`scripts/46_label_audit.py`**. Builds the two live markers per cell (condensation and the
488 / 561 ratio, both relative to the cell's own 0 h frame), then the tables behind figures 2
and 5. Confidence intervals come from `boot_fields`, which resamples whole fields, because
cells in one field share focus, density and illumination. The readability part trains one
logistic regression per frame and always scores fields the classifier has not seen
(`GroupKFold` or `LeaveOneGroupOut` on the field).

## 3. The model

**`src/static2fate/flow.py`** (190 lines).

- `UNet`: three resolutions, two residual blocks each, no attention. The condition vector is
  the sum of four embeddings (flow time, frame after drug, fate class, frame of the context
  image) and enters every block as a scale and shift after the second GroupNorm (FiLM). The
  context image is concatenated to the input as extra channels.
- `flow_matching_loss`: draw noise `x0` and a time `s`, form `xs = (1 - s) x0 + s x1`, and
  regress the model's output on `x1 - x0`. With probability 0.1 the fate is replaced by the
  null class, so one network learns both the conditional and the unconditional velocity.
- `sample`: 50 Euler steps from noise. Each step evaluates the model with the fate and with the
  null class in one batch and moves along `v_null + guidance * (v_fate - v_null)`.
- `invert`, the `x_start` and `project` arguments and `flow_matching_losses` belong to
  experiments in the log (inversion, SDEdit, anchored sampling, perceptual loss).

**`src/static2fate/video.py`**. `VideoUNet` is the same UNet applied to every frame of a clip
(frames folded into the batch) with a `TemporalAttention` layer after each of its seven stages:
self-attention over the frame axis, separately at every pixel. Its output projection starts at
zero, so at initialisation the video model is exactly the frame model (tested in
`tests/test_video.py`). One noise draw and one flow time per clip.

**`src/static2fate/autoencoder.py`**, **`latent.py`**. A convolutional KL autoencoder,
downsampling by 4, four latent channels. The flow model sees the posterior mean divided by its
standard deviation over the training set, and never trains the decoder.

**`scripts/34_train_latent.py`**. AdamW at 2e-4, gradient clipping at 1, EMA of the weights
(0.999), flips and 90 degree rotations applied to the code. **`scripts/35_generate_latent.py`**
encodes the 0 h frame, samples the clip, decodes, and writes the samples in one layout for
every scoring script. Its last argument forces a fate class, which is how the counterfactual
figure is made: the noise depends only on the seed and the cell order.

**`scripts/09_train_flow.py`** holds what every other script imports: `HELDOUT_WELLS`,
`split_index`, `load` and `fate_classes`. The split is by well, never by cell.

## 4. Scoring

**`src/static2fate/evaluate.py`**: `field_bootstrap` (group means with CIs over fields),
`energy_terms` (mean distance of samples to the truth, and mean distance between samples) and
`rank_of_own` (where the cell's own real future ranks among real candidates).
**`src/static2fate/features.py`**: `crop_features` measures a crop the same way for real and
generated images (Otsu nucleus nearest the centre, ring means); `dinov2_features` is the frozen
encoder. Scripts 20, 21, 22, 30, 42 and 45 each answer one question
([scripts/README.md](../scripts/README.md)) and `48_scorecard.py` draws them together.

## 5. Display

**`src/static2fate/display.py`**: one colour and intensity convention for every image of real
cells, `plate_ranges` for the plate-wide scaling, `mosaic` and `scale_bar`.

## Questions worth preparing

- **Why flow matching and not diffusion?** Straight paths, a plain regression loss, a
  deterministic sampler with few steps, and the same noise can be reused under two conditions.
- **Why is the fate an input?** To separate two problems. Given the outcome, can the model draw
  the course? That is tested here. Predicting the outcome is a different task, and section 3
  of the report shows a single early frame does not carry it.
- **How do you know the model did not memorise?** Held-out loss equals training loss
  ([log](log.md#too-small-or-short-of-cells)), and all scores are on wells the model never saw.
- **What would you change first?** The label: event times from the movie instead of a
  thresholded endpoint stain (report, section 7).
- **Why handcrafted features next to DINOv2?** They are interpretable, they can be computed on
  a generated crop, and at this scale they are at least as good.
- **What is the unit of analysis?** The field for confidence intervals, the well for the
  train/test split. With three wells per dose, well-to-well variation is not estimated.
