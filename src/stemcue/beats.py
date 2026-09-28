"""Beat grid: beat_this beats and downbeats, numbering, bars, grid positions, and a librosa cross-check.

Numbering, bars and pos are pure functions of a beat list and a downbeat list, so corrected lists can be fed back in.
"""

import librosa
import numpy as np
import torch

from stemcue.analysis import ACTIVE_DB, FR, HOP, Levels
from stemcue.audio import SR
from stemcue.schema import Bar, Beat, BeatCheck, Pos
from stemcue.vendor.beat_this.inference import Audio2Beats
from stemcue.vendor.beat_this.model import BeatThis
from stemcue.vendor.beat_this.utils import infer_beat_numbers

CHECK_WINDOW_S = 10
CHECK_TOL_S = 0.04
TEXTURE_MIN_FRACTION = 0.5
DRUM_WEIGHT = 1.0
SUPPORT_WEIGHT = 0.5


def track(model: BeatThis, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """beat_this beats and downbeats (seconds) of a mono signal at SR, on CPU with the minimal postprocessor."""
    with torch.inference_mode():
        beats, downbeats = Audio2Beats(model, device="cpu")(mix, SR)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def median_ibi(beats: np.ndarray) -> float:
    return float(np.median(np.diff(beats)))


def bpm_median(beats: np.ndarray) -> float:
    return round(60 / median_ibi(beats), 2) if len(beats) >= 2 else 0.0  # noqa: PLR2004


def number_beats(beats: np.ndarray, downbeats: np.ndarray) -> list[Beat]:
    """Bar number, beat-in-bar, downbeat flag and local tempo of each beat."""
    n = len(beats)
    diffs = np.diff(beats)
    bpms = []
    for i in range(n):
        around = diffs[max(i - 1, 0) : i + 1]
        bpms.append(round(60 / float(np.median(around)), 2))
    if len(downbeats) == 0:
        return [Beat(t=round(float(t), 4), bar=0, beat=i + 1, downbeat=False, bpm=bpms[i]) for i, t in enumerate(beats)]
    numbers = infer_beat_numbers(beats, downbeats)
    down = set(downbeats.tolist())
    out: list[Beat] = []
    bar = 0
    for i, t in enumerate(beats):
        is_down = float(t) in down
        if is_down:
            bar += 1
        out.append(Beat(t=round(float(t), 4), bar=bar, beat=int(numbers[i]), downbeat=is_down, bpm=bpms[i]))
    return out


def _frame_slice(start: float, end: float, nf: int) -> slice:
    i0 = min(int(start * FR), nf - 1)
    return slice(i0, max(min(int(end * FR), nf), i0 + 1))


def make_bars(beats: np.ndarray, downbeats: np.ndarray, duration: float, lv: Levels) -> list[Bar]:
    """One bar per downbeat, running to the next downbeat (the last to the last beat plus one median IBI)."""
    if len(downbeats) == 0:
        return []
    last_end = min(float(beats[-1]) + median_ibi(beats), duration)
    ends = [*downbeats[1:].tolist(), last_end]
    nf = len(lv.mix_db)
    bars: list[Bar] = []
    for n, (start, end) in enumerate(zip(downbeats.tolist(), ends, strict=True), start=1):
        frames = _frame_slice(start, end, nf)
        texture = [
            s.name
            for s in lv.stems
            if not s.silent and float(np.mean(s.level_db[frames] > ACTIVE_DB)) > TEXTURE_MIN_FRACTION
        ]
        bars.append(
            Bar(
                n=n,
                start=round(start, 4),
                end=round(end, 4),
                beats=int(np.sum((beats >= start) & (beats < end))),
                texture=texture,
                mix_db=round(float(np.mean(lv.mix_db[frames])), 1),
            )
        )
    return bars


def pos(t: float, beats: np.ndarray, numbered: list[Beat]) -> Pos | None:
    """Nearest sixteenth on the beat grid, as bar.beat.sixteenth plus the offset from that grid point."""
    if len(beats) < 2:  # noqa: PLR2004
        return None
    med = median_ibi(beats)
    if t < beats[0] or t >= beats[-1] + med:
        return None
    j = int(np.searchsorted(beats, t, side="right")) - 1
    ibi = float(beats[j + 1] - beats[j]) if j + 1 < len(beats) else med
    q = round((t - float(beats[j])) / ibi * 4)
    grid_t = float(beats[j]) + q / 4 * ibi
    if q == 4:  # noqa: PLR2004
        if j + 1 < len(beats):
            j, q = j + 1, 0
            grid_t = float(beats[j])
        else:
            q = 3
            grid_t = float(beats[j]) + q / 4 * ibi
    b = numbered[j]
    return Pos(
        bar=b.bar,
        beat=b.beat,
        sixteenth=q + 1,
        label=f"{b.bar}.{b.beat}.{q + 1}",
        offset_ms=round((t - grid_t) * 1000, 1),
    )


def _nearest(ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ref, values)
    hi = np.clip(idx, 0, len(ref) - 1)
    lo = np.clip(idx - 1, 0, len(ref) - 1)
    return np.where(np.abs(ref[hi] - values) < np.abs(ref[lo] - values), ref[hi], ref[lo])


def percussive_envelope(lv: Levels, fluxes: dict[str, np.ndarray], mix_flux: np.ndarray) -> np.ndarray:
    env = np.zeros_like(mix_flux)
    for s in lv.stems:
        if s.silent or s.name not in fluxes:
            continue
        lower = s.name.lower()
        weight = DRUM_WEIGHT if "drum" in lower else SUPPORT_WEIGHT if ("bass" in lower or "perc" in lower) else 0.0
        if weight:
            env = env + weight * fluxes[s.name][: len(env)]
    return env if env.any() else mix_flux


def beat_check(beats: np.ndarray, bpm: float, perc_env: np.ndarray, duration: float) -> list[BeatCheck]:
    """Per 10 s window: how many beat_this beats have a librosa DP beat within 40 ms, and the median offset."""
    if bpm == 0:
        return []
    _, frames = librosa.beat.beat_track(
        onset_envelope=perc_env, sr=SR, hop_length=HOP, bpm=bpm, tightness=400, trim=False
    )
    lb = np.asarray(frames, dtype=float) / FR
    out: list[BeatCheck] = []
    for s in np.arange(0, duration, CHECK_WINDOW_S):
        start = float(s)
        end = min(start + CHECK_WINDOW_S, duration)
        inside = beats[(beats >= start) & (beats < end)]
        if len(inside) == 0:
            continue
        if len(lb) == 0:
            agree, offsets = 0.0, np.array([])
        else:
            d = _nearest(lb, inside) - inside
            ok = np.abs(d) <= CHECK_TOL_S
            agree, offsets = float(ok.mean()), d[ok]
        out.append(
            BeatCheck(
                start=round(start, 4),
                end=round(end, 4),
                n=len(inside),
                agree=round(agree, 3),
                median_offset_ms=round(float(np.median(offsets)) * 1000, 1) if len(offsets) else None,
            )
        )
    return out
