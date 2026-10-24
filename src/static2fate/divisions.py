"""Find cell divisions in nuclear tracks.

At 30 min per frame a HeLa mitosis (about an hour) shows up as one or two frames where the
nucleus is small and bright (condensed chromosomes), followed by a larger, dimmer nucleus again
(a daughter). Apoptosis also condenses the nucleus but does not come back. trackpy keeps one
daughter on the mother's track; the other daughter starts a new track next to it.

So a division is: a condensation frame, a recovery within RECOVER frames, and a sister track
born within SISTER_PX of the cell at the recovery frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COND_AREA = 0.6  # condensed: area below 0.6 x the cell's recent median ...
COND_HOECHST = 1.4  # ... and Hoechst mean above 1.4 x its recent median
RECOVER = 3  # recovery within 3 frames (1.5 h) of the condensation
REC_AREA = 1.5  # recovered: area 1.5 x and Hoechst 0.8 x the condensed frame
REC_HOECHST = 0.8
SISTER_PX = 70  # same as the tracking search range
SISTER_AREA = (0.5, 2.0)  # sister nucleus area relative to the daughter


def condensation_events(area: np.ndarray, hoechst: np.ndarray) -> list[int]:
    """Indices t (t >= 1) where the nucleus is condensed relative to the up to 3 frames before."""
    out = []
    for t in range(1, len(area)):
        lo = max(0, t - 3)
        ref_a, ref_h = np.nanmedian(area[lo:t]), np.nanmedian(hoechst[lo:t])
        if area[t] < COND_AREA * ref_a and hoechst[t] > COND_HOECHST * ref_h:
            out.append(t)
    return out


def recovery_after(area: np.ndarray, hoechst: np.ndarray, m: int) -> int | None:
    """First index r in m+1 .. m+RECOVER where the nucleus has decondensed, else None."""
    for r in range(m + 1, min(m + RECOVER, len(area) - 1) + 1):
        if area[r] > REC_AREA * area[m] and hoechst[r] < REC_HOECHST * hoechst[m]:
            return r
    return None


def find_divisions(field: pd.DataFrame, frames: list[int]) -> pd.DataFrame:
    """One field's per-frame nuclei (columns frame, particle, x, y, area, hoechst_nuc_mean).

    Returns one row per track with: first condensation frame, recovery frame, whether a sister
    track was born next to it, and whether the track is still condensed at the last frame.
    """
    f = field[field.frame.isin(frames)]
    born = f.groupby("particle").frame.min()
    first_pos = f.sort_values("frame").groupby("particle").first()
    rows = []
    for pid, s in f.groupby("particle"):
        s = s.set_index("frame").reindex(frames)
        area, hoe = s.area.to_numpy(float), s.hoechst_nuc_mean.to_numpy(float)
        if np.isnan(area).any():  # gaps (tracker memory): skip, the event test needs every frame
            rows.append({"particle": pid, "complete_live": False})
            continue
        ev = condensation_events(area, hoe)
        cond_frame = rec_frame = None
        sister = None
        for m in ev:
            r = recovery_after(area, hoe, m)
            if r is not None:
                cond_frame, rec_frame = frames[m], frames[r]
                # a track born between the condensation and one frame after recovery, nearby
                cand = born[(born >= frames[m]) & (born <= min(frames[r] + 1, frames[-1])) & (born.index != pid)]
                sister = None
                for q in cand.index:
                    p = first_pos.loc[q]
                    at = s.loc[p.frame] if p.frame in s.index else s.loc[frames[r]]
                    d = np.hypot(p.x - at.x, p.y - at.y)
                    ratio = p.area / at.area
                    if d < SISTER_PX and SISTER_AREA[0] < ratio < SISTER_AREA[1]:
                        sister = int(q)
                        break
                break
        # condensed relative to the start (median of the first 3 frames), and since when
        vs0 = (area < COND_AREA * np.median(area[:3])) & (hoe > COND_HOECHST * np.median(hoe[:3]))
        end_onset = None
        if vs0[-1]:
            k = len(vs0) - 1
            while k > 0 and vs0[k - 1]:
                k -= 1
            end_onset = frames[k]
        rows.append({"particle": pid, "complete_live": True,
                     "first_condensed_frame": frames[ev[0]] if ev else None,
                     "condensed_at_end": bool(vs0[-1]), "end_onset_frame": end_onset,
                     "cond_frame": cond_frame, "recovery_frame": rec_frame,
                     "mitosis_like": rec_frame is not None, "sister": sister,
                     "division": sister is not None})
    return pd.DataFrame(rows)
