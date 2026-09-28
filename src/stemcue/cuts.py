"""cuts: cut-time candidates of a cues.json (downbeats, section starts, strong hits) quantised to a frame rate."""

import math
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from stemcue.errors import UsageError
from stemcue.grid import load_cues
from stemcue.schema import Bar, Beat, Cues, Event, _Model

CUTS_SCHEMA: Final = "stemcue.cuts/1"
SECTION_WINDOW_S = 0.35
MIX_JUMP_DB = 6.0
MIN_TEXTURE_CHANGES = 2
FRAME_EPS = 1e-6
MAX_FPS = 240.0
BAR_MATCH_S = 0.001
LATE_SIXTEENTH = 3

CutKind = Literal["bar", "section", "hit"]
_KIND_ORDER: Final = {"bar": 0, "section": 0, "hit": 1}


class Cut(_Model):
    kind: CutKind
    t: float
    frame: int
    frame_t: float
    bar: int | None
    pos: str | None
    offset_ms: float | None
    strength: float | None
    stems: list[str] | None
    snap_t: float | None
    anticipation: bool
    free: bool
    reason: list[str]


class CutList(_Model):
    schema_id: Literal["stemcue.cuts/1"] = Field(default=CUTS_SCHEMA, alias="schema")
    cues: str
    fps: float
    min_gap_s: float
    min_strength: float
    grid_source: Literal["beat_this", "fix"]
    candidates: list[Cut]


def _validate(fps: float, min_gap: float, min_strength: float) -> None:
    if not (math.isfinite(fps) and 0 < fps <= MAX_FPS):
        raise UsageError("--fps must be in (0, 240]")
    if not (math.isfinite(min_gap) and min_gap >= 0):
        raise UsageError("--min-gap must be >= 0")
    if not (math.isfinite(min_strength) and min_strength >= 0):
        raise UsageError("--min-strength must be >= 0")


def _frame(t: float, fps: float) -> tuple[int, float]:
    frame = math.floor(t * fps + FRAME_EPS)
    return frame, round(frame / fps, 4)


def _near(events: list[Event], kind: str, t: float) -> list[Event]:
    return [e for e in events if e.kind == kind and abs(e.t - t) <= SECTION_WINDOW_S]


def _strongest(events: list[Event], t: float) -> float | None:
    if not events:
        return None
    return min(events, key=lambda e: (-e.strength, abs(e.t - t), e.t)).t


def _downbeat_cut(cues: Cues, beat: Beat, bar: Bar | None, prev: Bar | None, fps: float) -> Cut:
    entered = {e.source for e in _near(cues.events, "enter", beat.t)}
    exited = {e.source for e in _near(cues.events, "exit", beat.t)}
    # A stem with both an enter and an exit here is a dropout or a short burst, not a texture change.
    flicker = entered & exited
    entering = sorted(entered - flicker)
    exiting = sorted(exited - flicker)
    texture = len(entering) + len(exiting) >= MIN_TEXTURE_CHANGES
    delta = bar.mix_db - prev.mix_db if bar is not None and prev is not None else None
    jump = delta is not None and abs(delta) >= MIX_JUMP_DB
    after_stop = any(
        e.kind == "stop" and e.end is not None and abs(e.end - beat.t) <= SECTION_WINDOW_S for e in cues.events
    )
    t = round(beat.t, 4)
    frame, frame_t = _frame(t, fps)
    section = texture or jump or after_stop
    reason: list[str] = []
    snap_t: float | None = None
    if section:
        if entering:
            reason.append(f"enter: {', '.join(entering)}")
        if exiting:
            reason.append(f"exit: {', '.join(exiting)}")
        if jump and delta is not None:
            reason.append(f"mix {delta:+.1f} dB")
        if after_stop:
            reason.append("after stop")
        onsets = [e for e in _near(cues.events, "onset", beat.t) if e.source in entering]
        snap_t = _strongest(onsets, beat.t)
        if snap_t is None:
            snap_t = _strongest(_near(cues.events, "band_hit", beat.t), beat.t)
    else:
        reason.append("downbeat")
    return Cut(
        kind="section" if section else "bar",
        t=t,
        frame=frame,
        frame_t=frame_t,
        bar=beat.bar,
        pos=f"{beat.bar}.{beat.beat}.1",
        offset_ms=None,
        strength=None,
        stems=sorted(set(entering) | set(exiting)) if section else None,
        snap_t=snap_t,
        anticipation=False,
        free=False,
        reason=reason,
    )


def _thin(hits: list[Event], min_gap: float) -> list[Event]:
    """Strongest first; a hit is kept only if it is at least min_gap from every hit already kept."""
    kept: list[Event] = []
    for e in sorted(hits, key=lambda e: (-e.strength, e.t)):
        if all(abs(e.t - k.t) >= min_gap for k in kept):
            kept.append(e)
    return kept


def _hit_cut(e: Event, last_beat: dict[int, int], fps: float) -> Cut:
    p = e.pos
    anticipation = p is not None and p.sixteenth >= LATE_SIXTEENTH and p.beat == last_beat.get(p.bar)
    free = p is None
    reason = [f"band_hit strength {e.strength}"]
    if anticipation:
        reason.append("anticipation: cut at t, before the bar")
    if free:
        reason.append("free tempo: cut at t")
    t = round(e.t, 4)
    frame, frame_t = _frame(t, fps)
    return Cut(
        kind="hit",
        t=t,
        frame=frame,
        frame_t=frame_t,
        bar=p.bar if p is not None else None,
        pos=p.label if p is not None else None,
        offset_ms=p.offset_ms if p is not None else None,
        strength=e.strength,
        stems=e.stems,
        snap_t=None,
        anticipation=anticipation,
        free=free,
        reason=reason,
    )


def _matching_bar(cues: Cues, t: float) -> Bar | None:
    return next((b for b in cues.bars if abs(b.start - t) <= BAR_MATCH_S), None)


def cut_candidates(cues_path: Path, fps: float, min_gap: float, min_strength: float) -> CutList:
    """Downbeats (bar or section) and thinned strong band hits, each with its frame at fps (truncated)."""
    _validate(fps, min_gap, min_strength)
    cues = load_cues(cues_path)
    candidates: list[Cut] = []
    prev: Bar | None = None
    for beat in (b for b in cues.beats if b.downbeat):
        bar = _matching_bar(cues, beat.t)
        candidates.append(_downbeat_cut(cues, beat, bar, prev, fps))
        prev = bar
    last_beat: dict[int, int] = {}
    for b in cues.beats:
        last_beat[b.bar] = max(last_beat.get(b.bar, b.beat), b.beat)
    hits = [e for e in cues.events if e.kind == "band_hit" and e.strength >= min_strength]
    candidates += [_hit_cut(e, last_beat, fps) for e in _thin(hits, min_gap)]
    candidates.sort(key=lambda c: (c.t, _KIND_ORDER[c.kind]))
    return CutList(
        cues=str(cues_path.resolve()),
        fps=fps,
        min_gap_s=min_gap,
        min_strength=min_strength,
        grid_source=cues.grid.source if cues.grid is not None else "beat_this",
        candidates=candidates,
    )
