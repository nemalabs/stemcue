"""PNG figures for checking a grid by eye: band envelopes over a window (look) and every stem's level (overview)."""

import math
from pathlib import Path

import matplotlib as mpl
import numpy as np

from stemcue import analysis
from stemcue.audio import SR, fit_length, load_mono
from stemcue.errors import InputError, UsageError
from stemcue.grid import load_cues, raw_grid, require_audio
from stemcue.schema import Cues, FreeSegment, RawGrid, StemInfo

mpl.use("Agg")
import matplotlib.pyplot as plt

MAX_WINDOW_S = 60
DRUM_BANDS = ((30, 120), (180, 350), (1500, 5000), (7000, 11000))
BASS_BAND = (30, 400)
OTHER_BAND = (100, 8000)
LOOK_ROW_IN = 1.6
LOOK_DPI = 55
ROW_S = 20
OVERVIEW_ROW_IN = 2.6
OVERVIEW_DPI = 48
FLOOR_DB = 60
LOOK_TITLE = (
    "black=onset flux, grey=level, blue=beat_this (thick=downbeat), green=current grid (tall=downbeat, number=bar), "
    "red=band_hit"
)
OVERVIEW_TITLE = (
    "stem level dB (vs mix p99), 50 fps; cyan = grid beats, white = downbeats (bar number), grey = free tempo"
)


def _literal(text: str) -> str:
    # matplotlib parses text between two "$" as mathtext (an invalid expression fails at savefig); "\$" is a plain "$".
    return text.replace("$", r"\$")


def _look_rows(cues: Cues) -> list[tuple[StemInfo, int, int]]:
    rows: list[tuple[StemInfo, int, int]] = []
    for s in cues.source.stems:
        if s.silent:
            continue
        lower = s.name.lower()
        bands = DRUM_BANDS if "drum" in lower else (BASS_BAND,) if "bass" in lower else (OTHER_BAND,)
        rows += [(s, lo, hi) for lo, hi in bands]
    return rows


def _save(fig: plt.Figure, out: Path, dpi: int) -> Path:
    try:
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
    finally:
        plt.close(fig)
    return out.resolve()


def _look_overlays(
    ax: plt.Axes, raw: RawGrid, grid_beats: list[float], grid_downs: list[float], hits: list[float]
) -> None:
    for x in raw.beats:
        ax.axvline(x, color="tab:blue", lw=0.8, alpha=0.7)
    for x in raw.downbeats:
        ax.axvline(x, color="tab:blue", lw=2.2, alpha=0.7)
    for x in grid_beats:
        ax.plot([x, x], [1.02, 1.12], color="tab:green", lw=2.5, clip_on=False)
    for x in grid_downs:
        ax.plot([x, x], [1.02, 1.25], color="tab:green", lw=2.5, clip_on=False)
    for x in hits:
        ax.plot([x, x], [-0.12, -0.02], color="tab:red", lw=2.5, clip_on=False)


def look(cues_path: Path, start: float, end: float, out: Path) -> Path:
    """Band flux of each non-silent stem over [start, end) with beat_this, the current grid and band hits on top."""
    cues = load_cues(cues_path)
    if not (0 <= start < end <= cues.duration) or end - start > MAX_WINDOW_S:
        msg = f"bad window {start}-{end} s: need 0 <= start < end <= {cues.duration} and at most 60 s"
        raise UsageError(msg)
    rows = _look_rows(cues)
    if not rows:
        raise InputError("no non-silent stems to plot")
    stem_dir = Path(cues.source.stem_dir)
    n = round(cues.duration * SR)
    window = slice(int(start * SR), int(end * SR))
    signals: dict[str, np.ndarray] = {}
    for s, _, _ in rows:
        if s.name not in signals:
            signals[s.name] = fit_length(load_mono(require_audio(stem_dir / s.file)), n)[window]

    def inside(xs: list[float]) -> list[float]:
        return [x for x in xs if start <= x < end]

    full_raw = raw_grid(cues)
    raw = RawGrid(beats=inside(full_raw.beats), downbeats=inside(full_raw.downbeats))
    grid_beats = inside([b.t for b in cues.beats])
    grid_downs = [(b.t, b.bar) for b in cues.beats if b.downbeat and start <= b.t < end]
    hits = inside([e.t for e in cues.events if e.kind == "band_hit"])
    fps = cues.envelopes.fps

    fig, axes = plt.subplots(len(rows), 1, figsize=(28, LOOK_ROW_IN * len(rows)), sharex=True, squeeze=False)
    for ax, (stem, lo, hi) in zip(axes[:, 0], rows, strict=True):
        f = analysis.flux(signals[stem.name], lo, hi)
        f = f / (f.max() + 1e-9)
        env = np.asarray(cues.envelopes.stems.get(stem.name, []), dtype=float)
        k = np.arange(math.ceil(start * fps), min(len(env), math.ceil(end * fps)))
        ax.fill_between(k / fps, 0, (env[k] + FLOOR_DB) / FLOOR_DB, color="0.85", lw=0)
        ax.plot(start + np.arange(len(f)) * analysis.HOP / SR, f, color="k", lw=0.6)
        ax.set_ylabel(_literal(f"{stem.name} {lo}-{hi}"), rotation=0, ha="right", fontsize=9)
        _look_overlays(ax, raw, grid_beats, [x for x, _ in grid_downs], hits)
        ax.set_ylim(-0.15, 1.3)
    top = axes[0, 0]
    for x, bar in grid_downs:
        top.text(x, 1.3, str(bar), fontsize=9)
    top.set_title(f"{start}-{end} s  {LOOK_TITLE}")
    bottom = axes[-1, 0]
    bottom.set_xlim(start, end)
    bottom.set_xticks(np.arange(math.ceil(start), end, 1.0))
    bottom.set_xticks(np.arange(math.ceil(start * 4) / 4, end, 0.25), minor=True)
    return _save(fig, out, LOOK_DPI)


def overview(cues_path: Path, out: Path) -> Path:
    """Every stem's level over the whole song, 20 s per row, with the current grid and the free-tempo spans."""
    cues = load_cues(cues_path)
    names = [s.name for s in cues.source.stems]
    series = [cues.envelopes.stems.get(name, []) for name in names]
    n_frames = min((len(v) for v in series), default=0)
    matrix = np.array([v[:n_frames] for v in series], dtype=float).reshape(len(names), n_frames)
    fps = cues.envelopes.fps
    beats = [b.t for b in cues.beats]
    downs = [(b.t, b.bar) for b in cues.beats if b.downbeat]
    fix = cues.grid.fix if cues.grid is not None else None
    free = [s for s in fix.segments if isinstance(s, FreeSegment)] if fix is not None else []
    rows = max(1, math.ceil(cues.duration / ROW_S))

    fig, axes = plt.subplots(rows, 1, figsize=(30, OVERVIEW_ROW_IN * rows), squeeze=False)
    for r, ax in enumerate(axes[:, 0]):
        a = ROW_S * r
        b = min(ROW_S * (r + 1), n_frames / fps)
        i0, i1 = int(a * fps), int(b * fps)
        if i1 > i0:
            ax.imshow(
                matrix[:, i0:i1],
                aspect="auto",
                cmap="magma",
                vmin=-FLOOR_DB,
                vmax=0,
                extent=(a, b, len(names), 0),
                interpolation="nearest",
            )
        ax.set_ylim(len(names), 0)
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_yticklabels([_literal(name) for name in names], fontsize=8)
        ax.set_xlim(a, a + ROW_S)
        ax.set_xticks(np.arange(a, a + ROW_S + 0.01, 1.0))
        for x in beats:
            if a <= x < b:
                ax.axvline(x, color="cyan", lw=0.5, alpha=0.6)
        for x, bar in downs:
            if a <= x < b:
                ax.axvline(x, color="white", lw=1.2, alpha=0.8)
                ax.text(x, -0.2, str(bar), fontsize=7, clip_on=False)
        for s in free:
            if s.start < b and s.end > a:
                ax.axvspan(max(s.start, a), min(s.end, b), color="grey", alpha=0.25)
    axes[0, 0].set_title(OVERVIEW_TITLE)
    return _save(fig, out, OVERVIEW_DPI)
