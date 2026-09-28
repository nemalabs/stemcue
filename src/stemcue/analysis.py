"""librosa analysis of stems and mix: levels, activity spans, onsets, drum bands, band hits, full stops, envelopes."""

import math
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from stemcue.audio import SR
from stemcue.schema import EventKind

HOP = 220
FR = SR / HOP
N_FFT = 2048
ACTIVE_DB = -38.0
SILENT_RATIO = 0.005
ENVELOPE_FPS = 50
ENVELOPE_FLOOR_DB = -60.0
MIN_SPAN_S = 0.3
BAND_HIT_WINDOW_S = 0.03
ATTACK_BEFORE_S = 0.030
ATTACK_AFTER_S = 0.100
ATTACK_SMOOTH_S = 0.003
ATTACK_LEVEL = 0.5
BANDPASS_ORDER = 4
CYMBAL_HI_HZ = min(11000.0, 0.95 * SR / 2)


@dataclass
class RawEvent:
    t: float
    kind: EventKind
    source: str
    strength: float
    end: float | None = None
    stems: list[str] | None = None


@dataclass
class StemLevels:
    name: str
    level_db: np.ndarray
    active_ratio: float
    silent: bool


@dataclass
class Levels:
    mix_db: np.ndarray
    mix_level_db: np.ndarray
    stems: list[StemLevels] = field(default_factory=list)


def _rms(y: np.ndarray) -> np.ndarray:
    return np.asarray(librosa.feature.rms(y=y, hop_length=HOP)[0])


def _to_db(rms: np.ndarray, ref: float) -> np.ndarray:
    return np.asarray(librosa.amplitude_to_db(rms + 1e-9, ref=ref))


def levels(mix: np.ndarray, stems: dict[str, np.ndarray]) -> Levels:
    """Levels in dB relative to the mix's 99th-percentile RMS; stems and mix smoothed over 5 frames."""
    mix_rms = _rms(mix)
    ref = float(np.percentile(mix_rms, 99))
    mix_db = _to_db(mix_rms, ref)
    out = Levels(mix_db=mix_db, mix_level_db=uniform_filter1d(mix_db, 5))
    for name, y in stems.items():
        level_db = uniform_filter1d(_to_db(_rms(y), ref), 5)
        ratio = float(np.mean(level_db > ACTIVE_DB))
        out.stems.append(StemLevels(name, level_db, ratio, ratio < SILENT_RATIO))
    return out


def spans(active: np.ndarray, min_on: float = 0.15, min_off: float = 0.12) -> list[list[float]]:
    """On-spans of a boolean frame series: short gaps closed, short blips dropped."""
    a = active.copy()
    i = 0
    n = len(a)
    while i < n:
        if not a[i]:
            j = i
            while j < n and not a[j]:
                j += 1
            if i > 0 and j < n and (j - i) / FR < min_off:
                a[i:j] = True
            i = j
        else:
            i += 1
    out = []
    i = 0
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if (j - i) / FR >= min_on:
                out.append([round(i / FR, 3), round(j / FR, 3)])
            i = j
        else:
            i += 1
    return out


def _stft_mag(y: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))


def _band_flux(mag: np.ndarray, lo: float, hi: float) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    band = np.log1p(10 * mag[(freqs >= lo) & (freqs < hi)])
    d = np.maximum(0, np.diff(band, axis=1, prepend=band[:, :1])).sum(axis=0)
    return np.asarray(d / (np.percentile(d, 99.5) + 1e-9))


def flux(y: np.ndarray, lo: float = 20, hi: float = 11000) -> np.ndarray:
    """Positive log-spectral flux in [lo, hi) Hz, normalised by its 99.5th percentile."""
    return _band_flux(_stft_mag(y), lo, hi)


def pick(env: np.ndarray, height: float, gap_s: float, prominence: float) -> list[tuple[float, float]]:
    idx, props = find_peaks(env, height=height, distance=max(1, int(gap_s * FR)), prominence=prominence)
    return [(round(i / FR, 4), round(float(h), 3)) for i, h in zip(idx, props["peak_heights"], strict=True)]


def attack_envelope(x: np.ndarray) -> np.ndarray:
    """Hilbert amplitude envelope of x (at SR), smoothed over 3 ms."""
    n = next_fast_len(len(x))
    env = np.abs(hilbert(x, N=n))[: len(x)]
    return np.asarray(uniform_filter1d(env, size=int(ATTACK_SMOOTH_S * SR)), dtype=np.float32)


def refine_attack(env: np.ndarray, tp: float) -> float:
    """Time where env first reaches half-way from its pre-peak floor to the peak near the flux peak tp.

    The centred 2048-point STFT flux peaks some 30 ms before an attack; the envelope locates it on the sample grid.
    """
    lo = max(0, math.ceil((tp - ATTACK_BEFORE_S) * SR))
    hi = min(len(env), math.floor((tp + ATTACK_AFTER_S) * SR) + 1)
    if lo >= hi:
        return tp
    window = env[lo:hi]
    i_peak = int(np.argmax(window))
    peak = float(window[i_peak])
    rising = window[: i_peak + 1]
    floor = float(np.min(rising))
    if peak - floor < 1e-6:  # noqa: PLR2004
        return tp
    # Search from the dip, not the window start: a sustained earlier note can already exceed the threshold there.
    i_floor = int(np.argmin(rising))
    first = i_floor + int(np.argmax(rising[i_floor:] >= floor + ATTACK_LEVEL * (peak - floor)))
    return round((lo + first) / SR, 4)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass, so the attack is not delayed."""
    sos = butter(BANDPASS_ORDER, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return np.asarray(sosfiltfilt(sos, x))


def _merge(events: list[RawEvent]) -> list[RawEvent]:
    """One event per (kind, source, t): refinement can move two flux peaks onto the same attack."""
    best: dict[tuple[str, str, float], RawEvent] = {}
    for e in events:
        key = (e.kind, e.source, e.t)
        if key not in best or e.strength > best[key].strength:
            best[key] = e
    return list(best.values())


def _refined(kind: EventKind, source: str, peaks: list[tuple[float, float]], env: np.ndarray) -> list[RawEvent]:
    return _merge([RawEvent(t=refine_attack(env, t), kind=kind, source=source, strength=h) for t, h in peaks])


def _drum_events(name: str, y: np.ndarray, mag: np.ndarray) -> list[RawEvent]:
    kick = _band_flux(mag, 30, 110)
    snare = 0.5 * _band_flux(mag, 180, 320) + 0.5 * _band_flux(mag, 1800, 5000)
    cymbal = _band_flux(mag, 6000, 11000)
    kick_env = attack_envelope(bandpass(y, 30, 110))
    snare_env = attack_envelope(bandpass(y, 180, 320) + bandpass(y, 1800, 5000))
    cymbal_env = attack_envelope(bandpass(y, 6000, CYMBAL_HI_HZ))
    return (
        _refined("kick", name, pick(kick, 0.35, 0.15, 0.2), kick_env)
        + _refined("snare", name, pick(snare, 0.35, 0.15, 0.2), snare_env)
        + _refined("cymbal", name, pick(cymbal, 0.45, 0.2, 0.25), cymbal_env)
    )


def percussive_events(y: np.ndarray) -> list[RawEvent]:
    """Kick, snare and cymbal events from the HPSS percussive part of a full mix."""
    _, perc = librosa.effects.hpss(y)
    return _drum_events("percussive", perc, _stft_mag(perc))


def _clamped_spans(active: np.ndarray, duration: float, min_on: float, min_off: float) -> list[tuple[float, float]]:
    return [(min(s, duration), min(e, duration)) for s, e in spans(active, min_on=min_on, min_off=min_off)]


def _span_events(name: str, level_db: np.ndarray, duration: float) -> list[RawEvent]:
    out: list[RawEvent] = []
    for start, end in _clamped_spans(level_db > ACTIVE_DB, duration, 0.15, 0.12):
        if end - start >= MIN_SPAN_S:
            out.append(RawEvent(t=start, kind="enter", source=name, strength=1.0))
            out.append(RawEvent(t=end, kind="exit", source=name, strength=1.0))
    return out


def stop_events(lv: Levels, duration: float) -> list[RawEvent]:
    """Full stops: the mix falls far below its running (4 s median) level."""
    run = median_filter(lv.mix_db, size=int(4 * FR))
    quiet = lv.mix_db < np.minimum(run - 18, -40)
    return [
        RawEvent(t=start, end=end, kind="stop", source="mix", strength=1.0)
        for start, end in _clamped_spans(quiet, duration, 0.12, 0.05)
    ]


@dataclass
class StemAnalysis:
    events: list[RawEvent]
    flux: dict[str, np.ndarray]


def stem_events(stems: dict[str, np.ndarray], lv: Levels, mix: np.ndarray, duration: float) -> StemAnalysis:
    """Onset, drum-band, enter/exit and band_hit events of the non-silent stems; attack times refined."""
    silent = {s.name for s in lv.stems if s.silent}
    events: list[RawEvent] = []
    fluxes: dict[str, np.ndarray] = {}
    onsets: dict[str, list[float]] = {}
    for sl in lv.stems:
        if sl.silent:
            continue
        y = stems[sl.name]
        mag = _stft_mag(y)
        fluxes[sl.name] = _band_flux(mag, 20, 11000)
        stem_onsets = _refined("onset", sl.name, pick(fluxes[sl.name], 0.3, 0.12, 0.15), attack_envelope(y))
        onsets[sl.name] = [e.t for e in stem_onsets]
        events += stem_onsets
        if "drum" in sl.name.lower():
            events += _drum_events(sl.name, y, mag)
        events += _span_events(sl.name, sl.level_db, duration)
    if fluxes:
        stack = np.sum([np.clip(f, 0, 1) for f in fluxes.values()], axis=0).astype(float)
        peaks = pick(stack / (np.percentile(stack, 99.8) + 1e-9), 0.45, 0.25, 0.3)
        for hit in _refined("band_hit", "mix", peaks, attack_envelope(mix)):
            hit.stems = [
                s.name
                for s in lv.stems
                if s.name not in silent and any(abs(o - hit.t) <= BAND_HIT_WINDOW_S for o in onsets[s.name])
            ]
            events.append(hit)
    return StemAnalysis(events=events, flux=fluxes)


def envelope(level_db: np.ndarray) -> list[float]:
    """level_db resampled to ENVELOPE_FPS by nearest frame, clipped to [-60, 0] dB."""
    nf = len(level_db)
    count = int(nf * ENVELOPE_FPS / FR)
    idx = [min(round(k * FR / ENVELOPE_FPS), nf - 1) for k in range(count)]
    return [round(float(v), 1) for v in np.clip(level_db[idx], ENVELOPE_FLOOR_DB, 0.0)]
