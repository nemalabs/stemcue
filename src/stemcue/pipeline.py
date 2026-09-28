"""analyze: stems (+ optional mix) or one song file -> cues.json and viewer.html. The CLI's only entry into analysis."""

import logging
from pathlib import Path
from typing import Final, Literal

import numpy as np

from stemcue import analysis, beats, weights
from stemcue.analysis import HOP, RawEvent
from stemcue.audio import SR, Stem, discover_stems, fit_length, load_mono, single_file_stem
from stemcue.errors import InputError, UsageError
from stemcue.schema import (
    AnalysisParams,
    Beat,
    BeatCheck,
    Cues,
    Envelopes,
    Event,
    Grid,
    ModelInfo,
    RawGrid,
    Source,
    StemInfo,
    Tempo,
)
from stemcue.viewer import render_viewer

log = logging.getLogger("stemcue")

DRUM_KINDS: Final = frozenset({"kick", "snare", "cymbal"})


def _to_event(e: RawEvent, beat_arr: np.ndarray, numbered: list[Beat]) -> Event:
    return Event(
        t=round(e.t, 4),
        end=round(e.end, 4) if e.end is not None else None,
        kind=e.kind,
        source=e.source,
        strength=round(e.strength, 3),
        stems=e.stems,
        pos=beats.pos(e.t, beat_arr, numbered) if numbered else None,
    )


def _inputs(input_path: Path, mix: Path | None) -> tuple[list[Stem], Path, Literal["stems", "file"]]:
    """Stems, the directory their file names are relative to, and the source kind of a stem folder or a song file."""
    path = input_path.resolve()
    if not path.exists():
        msg = f"no such file or directory: {path}"
        raise InputError(msg)
    if path.is_file():
        if mix is not None:
            raise UsageError("--mix applies only to a stem folder; a single file is its own mix")
        return [single_file_stem(path)], path.parent, "file"
    return discover_stems(path), path, "stems"


def analyze(input_path: Path, out_dir: Path, mix: Path | None, checkpoint: str, weights_dir: Path) -> tuple[Path, Path]:
    """Analyze a stem folder or one song file; return the resolved paths of the written cues.json and viewer.html."""
    weights.validate_name(checkpoint)
    stems, stem_dir, kind = _inputs(input_path, mix)
    mix_path = mix.resolve() if mix is not None else None
    if mix_path is not None and not mix_path.is_file():
        msg = f"mix file not found: {mix_path}"
        raise InputError(msg)

    log.info("loading %d stems from %s", len(stems), stem_dir)
    signals = {s.name: load_mono(s.path) for s in stems}
    n = max(len(y) for y in signals.values())
    signals = {name: fit_length(y, n) for name, y in signals.items()}
    mix_signal = fit_length(load_mono(mix_path), n) if mix_path is not None else np.sum(list(signals.values()), axis=0)
    duration = n / SR

    weights.ensure_installed(checkpoint, weights_dir)
    model, source_sha = weights.load_model(checkpoint, weights_dir)

    log.info("measuring levels and onsets")
    lv = analysis.levels(mix_signal, signals)
    stem_result = analysis.stem_events(signals, lv, mix_signal, duration)
    raw_events = stem_result.events + analysis.stop_events(lv, duration)
    if kind == "file":
        log.info("splitting percussion from the mix (HPSS)")
        # stem_events adds drum bands when the file name contains "drum"; the HPSS events replace them.
        raw_events = [e for e in raw_events if e.kind not in DRUM_KINDS]
        raw_events += analysis.percussive_events(signals[stems[0].name])

    log.info("tracking beats with beat_this (%s)", checkpoint)
    beat_arr, downbeat_arr = beats.track(model, mix_signal)
    if len(beat_arr) < 2:  # noqa: PLR2004
        beat_arr, downbeat_arr = np.array([]), np.array([])
    numbered = beats.number_beats(beat_arr, downbeat_arr) if len(beat_arr) else []
    bpm = beats.bpm_median(beat_arr)
    bars = beats.make_bars(beat_arr, downbeat_arr, duration, lv) if len(beat_arr) else []
    check: list[BeatCheck] = []
    if bpm:
        log.info("cross-checking the beats with librosa")
        perc_env = beats.percussive_envelope(lv, stem_result.flux, analysis.flux(mix_signal))
        check = beats.beat_check(beat_arr, bpm, perc_env, duration)

    events = sorted((_to_event(e, beat_arr, numbered) for e in raw_events), key=lambda e: e.t)
    cues = Cues(
        source=Source(
            kind=kind,
            stem_dir=str(stem_dir),
            mix=str(mix_path) if mix_path is not None else None,
            stems=[
                StemInfo(name=s.name, file=s.path.name, silent=sl.silent, active_ratio=round(sl.active_ratio, 4))
                for s, sl in zip(stems, lv.stems, strict=True)
            ],
        ),
        duration=round(duration, 4),
        analysis=AnalysisParams(sr=SR, hop=HOP),
        model=ModelInfo(checkpoint=checkpoint, source_sha256=source_sha),
        tempo=Tempo(bpm_median=bpm),
        beats=numbered,
        bars=bars,
        beat_check=check,
        events=events,
        envelopes=Envelopes(
            fps=analysis.ENVELOPE_FPS,
            mix=analysis.envelope(lv.mix_level_db),
            stems={sl.name: analysis.envelope(sl.level_db) for sl in lv.stems},
        ),
        grid=Grid(
            source="beat_this",
            beat_this=RawGrid(
                beats=[round(float(t), 4) for t in beat_arr],
                downbeats=[round(float(t), 4) for t in downbeat_arr],
            ),
            fix=None,
            report=None,
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    cues_path = (out_dir / "cues.json").resolve()
    viewer_path = (out_dir / "viewer.html").resolve()
    cues_path.write_text(cues.model_dump_json(indent=None), encoding="utf-8")
    log.info("rendering the viewer")
    render_viewer(cues, mix_signal, viewer_path)
    return cues_path, viewer_path
