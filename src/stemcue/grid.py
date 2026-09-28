"""grid: rebuild the beats, bars and event positions of a cues.json from a hand-written grid fix file."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from stemcue.analysis import ACTIVE_DB
from stemcue.audio import SR, fit_length, load_mono
from stemcue.beats import TEXTURE_MIN_FRACTION, median_ibi
from stemcue.errors import InputError, UsageError
from stemcue.schema import (
    Bar,
    BarSegment,
    Beat,
    Cues,
    Event,
    FitSegment,
    FreeSegment,
    Grid,
    GridFix,
    GridReport,
    Pos,
    RawGrid,
    Segment,
    SegmentReport,
    Tempo,
    TrackSegment,
)
from stemcue.viewer import render_viewer

log = logging.getLogger("stemcue")

MIN_BEATS = 2
# A fit's last beat can land a float hair before the next segment's first beat; closer than this is an overlap.
MIN_BEAT_STEP_S = 0.001
GAP_FACTOR = 1.5
TRACK_MARGIN_S = 0.3
IBI_LO_FACTOR = 0.93
IBI_HI_FACTOR = 1.08
OUTLIER_PASSES = 3
MIN_OUTLIER_NEIGHBOURS = 4
MIN_SMOOTH_ANCHORS = 3
DOWNBEAT_TOL_S = 0.060
SIXTEENTHS = 4


@dataclass
class _Part:
    """Bars (beat times) of one segment; bpb is set for fit/track, whose full bars go into the downbeat check."""

    bars: list[list[float]]
    note: str
    bpb: int | None


def load_cues(path: Path) -> Cues:
    try:
        return Cues.model_validate(json.loads(path.read_bytes()))
    except (OSError, ValueError) as exc:
        msg = f"cannot read cues file: {path}"
        raise InputError(msg) from exc


def load_fix(path: Path) -> GridFix:
    try:
        data = path.read_bytes()
    except OSError as exc:
        msg = f"invalid grid fix {path}: cannot read file"
        raise InputError(msg) from exc
    try:
        obj = json.loads(data)
    except ValueError as exc:
        msg = f"invalid grid fix {path}: not valid JSON"
        raise InputError(msg) from exc
    try:
        return GridFix.model_validate(obj)
    except ValidationError as exc:
        err = exc.errors()[0]
        loc = ".".join(map(str, err["loc"]))
        text = err["msg"].removeprefix("Value error, ")
        detail = f"{loc}: {text}" if loc else text
        msg = f"invalid grid fix {path}: {detail}"
        raise InputError(msg) from exc


def raw_grid(cues: Cues) -> RawGrid:
    """Return the beat_this lists from cues.grid, or from the beats of a cues.json written before grids existed."""
    if cues.grid is not None:
        return cues.grid.beat_this
    return RawGrid(beats=[b.t for b in cues.beats], downbeats=[b.t for b in cues.beats if b.downbeat])


def require_audio(path: Path) -> Path:
    if not path.is_file():
        msg = f"audio file not found: {path}"
        raise InputError(msg)
    return path


def _free(_seg: FreeSegment) -> _Part:
    return _Part(bars=[], note="no beats", bpb=None)


def _bar(seg: BarSegment) -> _Part:
    step = (seg.end - seg.start) / seg.beats
    return _Part(
        bars=[[seg.start + k * step for k in range(seg.beats)]],
        note=f"{seg.beats} beats of {step:.4f} s",
        bpb=None,
    )


def _fit(seg: FitSegment) -> _Part:
    idx = np.array([h.bar for h in seg.hits], dtype=float)
    t = np.array([h.t for h in seg.hits], dtype=float)
    bar_len, t0 = (float(v) for v in np.polyfit(idx, t, 1))
    residuals = t - (bar_len * idx + t0)
    n_bars = max(1, round((seg.end - t0) / bar_len))
    bpb = seg.beats_per_bar
    bars = []
    for k in range(n_bars):
        s = t0 + k * bar_len
        bars.append([s + j * bar_len / bpb for j in range(bpb)])
    note = (
        f"bar {bar_len:.4f} s = {60 * bpb / bar_len:.2f} BPM; first bar {t0:.3f}; "
        f"hit residuals ms {[round(float(r) * 1000, 1) for r in residuals]}; ends {t0 + n_bars * bar_len:.3f}"
    )
    return _Part(bars=bars, note=note, bpb=bpb)


def _too_few_anchors(seg: TrackSegment) -> InputError:
    msg = f"track segment {seg.start:.3f}-{seg.end:.3f} s: fewer than 2 usable beat_this beats"
    return InputError(msg)


def _track_anchors(seg: TrackSegment, raw_beats: list[float]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Beat indices and times of the raw beats that sit on a steady pulse, plus the times dropped as off-line."""
    lo = seg.ibi_min if seg.ibi_min is not None else IBI_LO_FACTOR * seg.period
    hi = seg.ibi_max if seg.ibi_max is not None else IBI_HI_FACTOR * seg.period
    raw = np.array(
        sorted(t for t in raw_beats if seg.start - TRACK_MARGIN_S <= t < seg.end + TRACK_MARGIN_S), dtype=float
    )
    ibi = np.diff(raw)
    ok = (ibi > lo) & (ibi < hi)
    keep = np.zeros(len(raw), dtype=bool)
    keep[:-1] |= ok
    keep[1:] |= ok
    anc = raw[keep]
    if len(anc) < MIN_BEATS:
        raise _too_few_anchors(seg)
    idx_list = [0]
    kept_list = [float(anc[0])]
    for b in anc[1:]:
        n = round(float((b - kept_list[-1]) / seg.period))
        if n == 0:
            continue
        idx_list.append(idx_list[-1] + n)
        kept_list.append(float(b))
    idx = np.array(idx_list)
    kept = np.array(kept_list)
    dropped: list[float] = []
    for _ in range(OUTLIER_PASSES):
        bad = np.zeros(len(idx), dtype=bool)
        positions = np.arange(len(idx))
        for i in range(len(idx)):
            m = (np.abs(idx - idx[i]) <= seg.fit_window) & (positions != i)
            if m.sum() >= MIN_OUTLIER_NEIGHBOURS:
                p = np.polyfit(idx[m], kept[m], 1)
                bad[i] = abs(kept[i] - np.polyval(p, idx[i])) > seg.tol_ms / 1000
        if not bad.any():
            break
        dropped += [round(float(x), 3) for x in kept[bad]]
        idx, kept = idx[~bad], kept[~bad]
    if len(kept) < MIN_BEATS:
        raise _too_few_anchors(seg)
    return idx, kept, dropped


def _track_smooth(seg: TrackSegment, idx: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """One time per beat index from the first to the last anchor: a local straight line through nearby anchors."""
    full = np.arange(idx[0], idx[-1] + 1)
    times = np.interp(full, idx, kept)
    smooth = times.copy()
    for p, f in enumerate(full):
        m = np.abs(idx - f) <= seg.smooth_window
        if m.sum() >= MIN_SMOOTH_ANCHORS:
            smooth[p] = np.polyval(np.polyfit(idx[m], kept[m], 1), f)
    return np.asarray(smooth, dtype=float)


def _track(seg: TrackSegment, raw_beats: list[float]) -> _Part:
    idx, kept, dropped = _track_anchors(seg, raw_beats)
    smooth = _track_smooth(seg, idx, kept)
    res_ms = np.abs(kept - smooth[idx - idx[0]]) * 1000
    j0 = int(np.argmin(np.abs(smooth - seg.start)))
    rest = smooth[j0:]
    rest = rest[rest < seg.end]
    bpb = seg.beats_per_bar
    bars = [rest[k : k + bpb].tolist() for k in range(0, len(rest) - bpb + 1, bpb)]
    tail = rest[(len(rest) // bpb) * bpb :]
    if len(tail):
        bars.append(tail.tolist())
    step = np.diff(rest)
    step_min, step_max = (float(step.min()), float(step.max())) if len(step) else (0.0, 0.0)
    note = (
        f"anchors {len(kept)}, dropped {dropped}, beats {len(rest)}; "
        f"anchor residual ms med {float(np.median(res_ms)):.1f} p95 {float(np.percentile(res_ms, 95)):.1f} "
        f"max {float(res_ms.max()):.1f}; beat period {step_min:.4f}..{step_max:.4f}"
    )
    return _Part(bars=bars, note=note, bpb=bpb)


def _build(seg: Segment, raw: RawGrid) -> _Part:
    if isinstance(seg, FreeSegment):
        return _free(seg)
    if isinstance(seg, BarSegment):
        return _bar(seg)
    if isinstance(seg, FitSegment):
        return _fit(seg)
    return _track(seg, raw.beats)


def _bpms(times: np.ndarray, is_gap: np.ndarray, med: float) -> list[float]:
    """Local tempo from the (at most two) neighbouring intervals that are not gaps."""
    d = np.diff(times)
    out: list[float] = []
    for i in range(len(times)):
        near = [float(d[k]) for k in (i - 1, i) if 0 <= k < len(d) and not is_gap[k]]
        if near:
            out.append(round(60 / float(np.median(near)), 2))
        else:
            out.append(round(60 / med, 2) if med > 0 else 0.0)
    return out


def _bars(cues: Cues, bar_beats: list[list[float]], med: float) -> list[Bar]:
    env = cues.envelopes
    mix = np.asarray(env.mix, dtype=float)
    out: list[Bar] = []
    for k, beats_k in enumerate(bar_beats):
        start = beats_k[0]
        if k + 1 < len(bar_beats) and bar_beats[k + 1][0] - beats_k[-1] <= GAP_FACTOR * med:
            end = bar_beats[k + 1][0]
        elif med > 0:
            end = min(start + len(beats_k) * med, cues.duration)
        else:
            end = cues.duration
        i0 = min(int(start * env.fps), len(mix) - 1)
        i1 = max(min(int(end * env.fps), len(mix)), i0 + 1)
        texture = [
            s.name
            for s in cues.source.stems
            if not s.silent
            and float(np.mean(np.asarray(env.stems[s.name][i0:i1], dtype=float) > ACTIVE_DB)) > TEXTURE_MIN_FRACTION
        ]
        out.append(
            Bar(
                n=k + 1,
                start=round(start, 4),
                end=round(end, 4),
                beats=len(beats_k),
                texture=texture,
                mix_db=round(float(np.mean(mix[i0:i1])), 1),
            )
        )
    return out


def _pos(t: float, times: np.ndarray, numbered: list[Beat], med: float, free: list[FreeSegment]) -> Pos | None:
    """Nearest sixteenth on the grid; None in free spans, outside the grid and across gaps."""
    n = len(times)
    if n < MIN_BEATS or any(s.start <= t < s.end for s in free):
        return None
    if t < times[0] or t >= times[-1] + med:
        return None
    j = int(np.searchsorted(times, t, side="right")) - 1
    steady = j + 1 < n and times[j + 1] - times[j] <= GAP_FACTOR * med
    if steady:
        ibi = float(times[j + 1] - times[j])
    else:
        ibi = med
        if t >= times[j] + med:
            return None
    q = round((t - float(times[j])) / ibi * SIXTEENTHS)
    grid_t = float(times[j]) + q / SIXTEENTHS * ibi
    if q == SIXTEENTHS:
        if steady:
            j, q = j + 1, 0
            grid_t = float(times[j])
        else:
            q = SIXTEENTHS - 1
            grid_t = float(times[j]) + q / SIXTEENTHS * ibi
    b = numbered[j]
    return Pos(
        bar=b.bar,
        beat=b.beat,
        sixteenth=q + 1,
        label=f"{b.bar}.{b.beat}.{q + 1}",
        offset_ms=round((t - grid_t) * 1000, 1),
    )


def _downbeat_check(parts: list[_Part], raw: RawGrid) -> tuple[int, list[float]]:
    marks = np.asarray(raw.downbeats, dtype=float)
    checked = 0
    missing: list[float] = []
    for part in parts:
        for bar in part.bars:
            if part.bpb is None or len(bar) != part.bpb:
                continue
            checked += 1
            if len(marks) == 0 or float(np.min(np.abs(marks - bar[0]))) > DOWNBEAT_TOL_S:
                missing.append(round(float(bar[0]), 3))
    return checked, missing


def rebuild(cues: Cues, fix: GridFix) -> Cues:
    """Rebuild beats, bars, tempo and event positions from the fix; everything else is kept unchanged."""
    raw = raw_grid(cues)
    parts: list[_Part] = []
    reports: list[SegmentReport] = []
    for seg in fix.segments:
        part = _build(seg, raw)
        parts.append(part)
        n_beats = sum(len(b) for b in part.bars)
        reports.append(
            SegmentReport(
                type=seg.type, start=seg.start, end=seg.end, bars=len(part.bars), beats=n_beats, notes=[part.note]
            )
        )
        log.info(
            "%s %.3f-%.3f: %d bars, %d beats; %s", seg.type, seg.start, seg.end, len(part.bars), n_beats, part.note
        )

    bar_beats = [bar for part in parts for bar in part.bars]
    times = np.array([t for bar in bar_beats for t in bar], dtype=float)
    backwards = np.nonzero(np.diff(times) <= MIN_BEAT_STEP_S)[0]
    if len(backwards):
        msg = f"grid segments overlap near {float(times[backwards[0] + 1]):.3f} s"
        raise InputError(msg)

    med = median_ibi(times) if len(times) >= MIN_BEATS else 0.0
    is_gap = np.diff(times) > GAP_FACTOR * med
    bpms = _bpms(times, is_gap, med)
    numbered: list[Beat] = []
    for n, bar in enumerate(bar_beats, start=1):
        for j, t in enumerate(bar):
            i = len(numbered)
            numbered.append(Beat(t=round(float(t), 4), bar=n, beat=j + 1, downbeat=j == 0, bpm=bpms[i]))
    steady = np.diff(times)[~is_gap]
    bpm_median = round(60 / float(np.median(steady)), 2) if len(times) >= MIN_BEATS else 0.0

    free = [s for s in fix.segments if isinstance(s, FreeSegment)]
    events: list[Event] = [e.model_copy(update={"pos": _pos(e.t, times, numbered, med, free)}) for e in cues.events]
    checked, missing = _downbeat_check(parts, raw)
    log.info("downbeats with no beat_this downbeat within 60 ms: %d of %d %s", len(missing), checked, missing)

    return cues.model_copy(
        update={
            "tempo": Tempo(bpm_median=bpm_median),
            "beats": numbered,
            "bars": _bars(cues, bar_beats, med),
            "events": events,
            "grid": Grid(
                source="fix",
                beat_this=raw,
                fix=fix,
                report=GridReport(segments=reports, downbeats_checked=checked, downbeats_missing=missing),
            ),
        }
    )


def _load_mix(cues: Cues) -> np.ndarray:
    """Reload the playback signal analyze used: the recorded mix file, or the sum of the stems."""
    n = round(cues.duration * SR)
    if cues.source.mix is not None:
        return fit_length(load_mono(require_audio(Path(cues.source.mix))), n)
    stem_dir = Path(cues.source.stem_dir)
    stems = [fit_length(load_mono(require_audio(stem_dir / s.file)), n) for s in cues.source.stems]
    return np.asarray(np.sum(stems, axis=0), dtype=np.float32)


def apply_fix(cues_path: Path, fix_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Rebuild cues_path's grid from fix_path; return the resolved paths of the written cues.json and viewer.html."""
    cues = load_cues(cues_path)
    fix = load_fix(fix_path)
    if out_dir.resolve() == cues_path.resolve().parent:
        raise UsageError("refusing to overwrite the input cues.json; choose another --out")
    fixed = rebuild(cues, fix)
    mix = _load_mix(fixed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cues = (out_dir / "cues.json").resolve()
    out_viewer = (out_dir / "viewer.html").resolve()
    out_cues.write_text(fixed.model_dump_json(indent=None), encoding="utf-8")
    render_viewer(fixed, mix, out_viewer)
    return out_cues, out_viewer
