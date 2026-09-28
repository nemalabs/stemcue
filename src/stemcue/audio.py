"""Stem discovery and audio loading (soundfile + soxr only; no ffmpeg fallback)."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
import soxr

from stemcue.errors import InputError

SR = 22050
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"})
UNNUMBERED = 10**9

_NUMBERED = re.compile(r"^(\d+)\s+(.+)$")


@dataclass(frozen=True)
class Stem:
    name: str
    path: Path


def discover_stems(stem_dir: Path) -> list[Stem]:
    """Audio files directly inside stem_dir, ordered by their leading number ("3 Guitar.wav") then name."""
    if not stem_dir.is_dir():
        msg = f"not a directory: {stem_dir}"
        raise InputError(msg)
    found: list[tuple[int, str, Path]] = []
    for path in stem_dir.iterdir():
        if path.name.startswith(".") or path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
            continue
        match = _NUMBERED.match(path.stem)
        if match:
            found.append((int(match.group(1)), match.group(2), path))
        else:
            found.append((UNNUMBERED, path.stem, path))
    if not found:
        msg = f"no audio files in {stem_dir}"
        raise InputError(msg)
    found.sort(key=lambda item: (item[0], item[1]))
    stems: list[Stem] = []
    seen: dict[str, int] = {}
    for _, name, path in found:
        count = seen.get(name, 0) + 1
        seen[name] = count
        stems.append(Stem(name if count == 1 else f"{name} ({count})", path))
    return stems


def single_file_stem(path: Path) -> Stem:
    """Return a whole-song audio file as the one stem of an analysis."""
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        msg = f"unsupported audio file: {path}"
        raise InputError(msg)
    return Stem(name=path.stem, path=path)


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Mono float32 samples of path at sr."""
    try:
        data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    except (soundfile.LibsndfileError, RuntimeError) as exc:
        msg = f"cannot read audio: {path}"
        raise InputError(msg) from exc
    y = np.asarray(data.mean(axis=1), dtype=np.float32)
    if rate != sr:
        y = np.asarray(soxr.resample(y, rate, sr, quality="HQ"), dtype=np.float32)
    return y


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or truncate y to n samples."""
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))
