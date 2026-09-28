"""diagnose: checks of a beat/bar grid that point at where to look (half/double time, odd IBIs and bars)."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
from pydantic import Field

from stemcue.beats import median_ibi
from stemcue.errors import InputError
from stemcue.grid import load_cues
from stemcue.schema import Cues, _Model

DIAGNOSE_SCHEMA: Final = "stemcue.diagnose/1"
ODD_IBI_TOL = 0.08
OCTAVE_LO, OCTAVE_HI = 1.8, 2.2
MIN_RUN_IBIS = 4
LOW_AGREE = 0.6
LOOK_PAD_S = 3.0
LOOK_MAX_S = 12.0
MIN_BEATS = 2
MIN_BAR_MISMATCH = 2
BAR_MISMATCH_FRACTION = 0.1

TempoKind = Literal["x2_slower", "x2_faster"]


class OddIbi(_Model):
    t: float
    ibi_s: float
    ratio: float


class TempoRun(_Model):
    start: float
    end: float
    kind: TempoKind
    n_ibis: int
    median_ibi_s: float


class OddBar(_Model):
    n: int
    start: float
    beats: int


class LowAgreement(_Model):
    start: float
    end: float
    agree: float
    median_offset_ms: float | None


class LookWindow(_Model):
    start: float
    end: float
    why: list[str]


class Diagnosis(_Model):
    schema_id: Literal["stemcue.diagnose/1"] = Field(default=DIAGNOSE_SCHEMA, alias="schema")
    cues: str
    source_kind: Literal["stems", "file"]
    grid_source: Literal["beat_this", "fix"]
    duration: float
    beats: int
    bars: int
    bpm_median: float
    median_ibi_s: float | None
    expected_bars: int | None
    bar_lengths: dict[str, int]
    odd_bars: list[OddBar]
    odd_ibis: list[OddIbi]
    tempo_runs: list[TempoRun]
    low_agreement: list[LowAgreement]
    downbeats_missing: list[float]
    look_windows: list[LookWindow]
    warnings: list[str]


@dataclass
class _Span:
    start: float
    end: float
    why: list[str] = field(default_factory=list)


def _octave(ratio: float) -> TempoKind | None:
    if OCTAVE_LO <= ratio <= OCTAVE_HI:
        return "x2_slower"
    if 1 / OCTAVE_HI <= ratio <= 1 / OCTAVE_LO:
        return "x2_faster"
    return None


def _tempo_runs(t: np.ndarray, ibis: np.ndarray, med: float) -> list[TempoRun]:
    """Maximal runs of at least MIN_RUN_IBIS consecutive IBIs in the same tempo octave off the median."""
    kinds = [_octave(float(ibi / med)) for ibi in ibis]
    runs: list[TempoRun] = []
    i = 0
    while i < len(kinds):
        kind = kinds[i]
        j = i
        while j < len(kinds) and kinds[j] == kind:
            j += 1
        if kind is not None and j - i >= MIN_RUN_IBIS:
            runs.append(
                TempoRun(
                    start=round(float(t[i]), 4),
                    end=round(float(t[j]), 4),
                    kind=kind,
                    n_ibis=j - i,
                    median_ibi_s=round(float(np.median(ibis[i:j])), 4),
                )
            )
        i = j
    return runs


def _common_bar_length(lengths: Counter[int]) -> int | None:
    if not lengths:
        return None
    top = max(lengths.values())
    return min(k for k, v in lengths.items() if v == top)


def _pad(t: float, why: str) -> _Span:
    return _Span(t - LOOK_PAD_S, t + LOOK_PAD_S, [why])


def _look_windows(spans: list[_Span], duration: float) -> list[LookWindow]:
    """Clamp, merge overlapping or touching spans (keeping every reason once) and cut long ones into pieces."""
    clamped = [_Span(max(0.0, s.start), min(duration, s.end), s.why) for s in spans]
    ordered = sorted((s for s in clamped if s.end > s.start), key=lambda s: s.start)
    merged: list[_Span] = []
    for s in ordered:
        if merged and s.start <= merged[-1].end:
            last = merged[-1]
            last.end = max(last.end, s.end)
            last.why += [w for w in s.why if w not in last.why]
        else:
            merged.append(_Span(s.start, s.end, list(s.why)))
    out: list[LookWindow] = []
    for s in merged:
        start = s.start
        while True:
            end = min(start + LOOK_MAX_S, s.end)
            out.append(LookWindow(start=round(start, 4), end=round(end, 4), why=list(s.why)))
            if end >= s.end:
                break
            start = end
    return out


def _warnings(cues: Cues, expected_bars: int | None, runs: int, odd_bars: int, low: int) -> list[str]:
    bars = len(cues.bars)
    out: list[str] = []
    if runs:
        out.append(f"{runs} tempo-octave run(s): beat_this may have switched to half or double time")
    if expected_bars and abs(bars - expected_bars) > max(MIN_BAR_MISMATCH, BAR_MISMATCH_FRACTION * expected_bars):
        out.append(f"{bars} bars but about {expected_bars} expected from the median beat")
    if odd_bars:
        out.append(f"{odd_bars} bar(s) with an unusual number of beats")
    if low:
        out.append(
            f"{low} beat_check window(s) below {LOW_AGREE} agreement "
            "(librosa can lock onto off-beats; look before trusting either)"
        )
    if len(cues.beats) < MIN_BEATS:
        out.append("fewer than 2 beats: no grid to check")
    return out


@dataclass
class _Findings:
    median_ibi_s: float | None = None
    expected_bars: int | None = None
    odd_bars: list[OddBar] = field(default_factory=list)
    odd_ibis: list[OddIbi] = field(default_factory=list)
    tempo_runs: list[TempoRun] = field(default_factory=list)
    low_agreement: list[LowAgreement] = field(default_factory=list)
    downbeats_missing: list[float] = field(default_factory=list)
    look_windows: list[LookWindow] = field(default_factory=list)


def _findings(cues: Cues, lengths: Counter[int], cues_path: Path) -> _Findings:
    if len(cues.beats) < MIN_BEATS:
        return _Findings()
    t = np.asarray([b.t for b in cues.beats], dtype=float)
    med = median_ibi(t)
    if not med > 0:
        msg = f"cannot diagnose {cues_path}: the median beat interval is {med}"
        raise InputError(msg)
    ibis = np.diff(t)
    odd_ibis = [
        OddIbi(t=round(float(t[i]), 4), ibi_s=round(float(ibi), 4), ratio=round(float(ibi / med), 3))
        for i, ibi in enumerate(ibis)
        if abs(ibi / med - 1) > ODD_IBI_TOL
    ]
    runs = _tempo_runs(t, ibis, med)
    common = _common_bar_length(lengths)
    odd_bars = [OddBar(n=b.n, start=b.start, beats=b.beats) for b in cues.bars if b.beats != common]
    low = [
        LowAgreement(start=w.start, end=w.end, agree=w.agree, median_offset_ms=w.median_offset_ms)
        for w in cues.beat_check
        if w.agree < LOW_AGREE
    ]
    report = cues.grid.report if cues.grid is not None else None
    missing = list(report.downbeats_missing) if report is not None else []

    spans = [_pad(x.t, f"odd ibi at {x.t}") for x in odd_ibis]
    for r in runs:
        spans += [_pad(r.start, f"tempo {r.kind} from {r.start}"), _pad(r.end, f"tempo {r.kind} until {r.end}")]
    spans += [_pad(b.start, f"bar {b.n} has {b.beats} beats") for b in odd_bars]
    spans += [_Span(w.start, w.end, [f"beat_check agree {w.agree}"]) for w in low]
    spans += [_pad(x, f"no beat_this downbeat near {x}") for x in missing]
    return _Findings(
        median_ibi_s=round(med, 4),
        expected_bars=round(cues.duration / (4 * med)),
        odd_bars=odd_bars,
        odd_ibis=odd_ibis,
        tempo_runs=runs,
        low_agreement=low,
        downbeats_missing=missing,
        look_windows=_look_windows(spans, cues.duration),
    )


def diagnose(cues_path: Path) -> Diagnosis:
    """Grid statistics and suspicious places of a cues.json, with time windows worth checking with look."""
    cues = load_cues(cues_path)
    lengths = Counter(b.beats for b in cues.bars)
    f = _findings(cues, lengths, cues_path)
    return Diagnosis(
        cues=str(cues_path.resolve()),
        source_kind=cues.source.kind,
        grid_source=cues.grid.source if cues.grid is not None else "beat_this",
        duration=cues.duration,
        beats=len(cues.beats),
        bars=len(cues.bars),
        bpm_median=cues.tempo.bpm_median,
        median_ibi_s=f.median_ibi_s,
        expected_bars=f.expected_bars,
        bar_lengths={str(k): lengths[k] for k in sorted(lengths)},
        odd_bars=f.odd_bars,
        odd_ibis=f.odd_ibis,
        tempo_runs=f.tempo_runs,
        low_agreement=f.low_agreement,
        downbeats_missing=f.downbeats_missing,
        look_windows=f.look_windows,
        warnings=_warnings(cues, f.expected_bars, len(f.tempo_runs), len(f.odd_bars), len(f.low_agreement)),
    )
