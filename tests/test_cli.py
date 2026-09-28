"""Black-box tests of the stemcue CLI (one CLI invocation per test)."""

import hashlib
import itertools
import json
import math
import os
import pickle
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors import safe_open

from stemcue.vendor.beat_this.model import BeatThis

ROOT = Path(__file__).resolve().parents[1]
FINAL0_CKPT = ROOT / ".cache" / "stemcue" / "download" / "final0.ckpt"
FINAL0_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
HOSTILE_NAME = "<img src=x onerror=alert(1)>"


def run_cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "stemcue", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        # matplotlib writes its font cache here instead of the user's home directory.
        env={**os.environ, "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib")},
    )


@pytest.fixture(scope="session")
def final0_ckpt() -> Path:
    if not FINAL0_CKPT.is_file():
        pytest.fail(f"missing {FINAL0_CKPT}; download it from the beat_this public directory first")
    return FINAL0_CKPT


@pytest.fixture(scope="session")
def weights_dir(tmp_path_factory: pytest.TempPathFactory, final0_ckpt: Path) -> Path:
    wdir = tmp_path_factory.mktemp("weights")
    r = run_cli("weights", "import", str(final0_ckpt), "--name", "final0", "--weights-dir", str(wdir))
    if r.returncode != 0:
        pytest.fail(f"weights import failed ({r.returncode}): {r.stderr}")
    return wdir


def write_synthetic_stems(stem_dir: Path) -> None:
    """16 s at 120 BPM: kick on every beat from 0.5 s, snare on beats 2 and 4, bass on each bar,
    one brass hit at 9.25 s, a silent vocal stem and a quiet stem whose file name is hostile HTML."""
    sr = 44100
    n = 16 * sr
    rng = np.random.default_rng(0)
    drums = np.zeros(n)
    bass = np.zeros(n)
    brass = np.zeros(n)
    hostile = np.zeros(n)
    beats = 0.5 + 0.5 * np.arange(30)
    for i, b in enumerate(beats):
        s = int(b * sr)
        k = np.arange(int(0.1 * sr))
        drums[s : s + k.size] += 0.8 * np.sin(2 * np.pi * 60 * k / sr) * np.exp(-k / (0.06 * sr))
        if i % 2 == 1:
            drums[s : s + k.size] += 0.4 * rng.standard_normal(k.size) * np.exp(-k / (0.08 * sr))
        if i % 4 == 0:
            m = np.arange(int(0.45 * sr))
            bass[s : s + m.size] += 0.3 * np.sin(2 * np.pi * 55 * m / sr)
            hostile[s : s + m.size] += 0.05 * np.sin(2 * np.pi * 880 * m / sr)
    s = int(9.25 * sr)
    m = np.arange(int(0.3 * sr))
    env = np.minimum(1.0, m / (0.005 * sr)) * np.exp(-m / (0.2 * sr))
    brass[s : s + m.size] += 0.3 * env * sum(np.sin(2 * np.pi * 440 * h * m / sr) / h for h in range(1, 6))
    sf.write(stem_dir / "0 Lead Vocals.wav", np.zeros(n), sr)
    sf.write(stem_dir / "1 Drums.wav", drums, sr)
    sf.write(stem_dir / "2 Bass.wav", bass, sr)
    sf.write(stem_dir / "9 Brass.wav", brass, sr)
    sf.write(stem_dir / f"11 {HOSTILE_NAME}.wav", hostile, sr)


def write_synthetic_song(path: Path) -> None:
    """The synthetic stems summed into one file, as a song without stems would arrive."""
    stem_dir = path.parent / "stems-for-song"
    stem_dir.mkdir()
    write_synthetic_stems(stem_dir)
    parts = [sf.read(p)[0] for p in sorted(stem_dir.iterdir())]
    sf.write(path, np.sum(parts, axis=0), 44100)


def test_analyze_single_file_song_finds_percussive_hits(tmp_path: Path, weights_dir: Path) -> None:
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    song = song_dir / "song mix.wav"
    write_synthetic_song(song)
    out = tmp_path / "out"

    r = run_cli("analyze", str(song), "--out", str(out), "--weights-dir", str(weights_dir), timeout=600)

    assert r.returncode == 0, r.stderr
    cues_path = (out / "cues.json").resolve()
    viewer_path = (out / "viewer.html").resolve()
    assert r.stdout == f"cues\t{cues_path}\nviewer\t{viewer_path}\n"
    cues = json.loads(cues_path.read_text(encoding="utf-8"))
    assert cues["source"]["kind"] == "file"
    assert cues["source"]["stem_dir"] == str(song_dir.resolve())
    assert cues["source"]["mix"] is None
    assert [(s["name"], s["file"]) for s in cues["source"]["stems"]] == [("song mix", "song mix.wav")]
    assert abs(cues["tempo"]["bpm_median"] - 120) <= 3
    kicks = [e for e in cues["events"] if e["kind"] == "kick"]
    snares = [e for e in cues["events"] if e["kind"] == "snare"]
    assert len(kicks) >= 8
    assert len(snares) >= 4
    assert {e["source"] for e in kicks + snares} == {"percussive"}


def test_diagnose_reports_grid_statistics_as_json(analyzed: Path) -> None:
    r = run_cli("diagnose", str(analyzed))

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    cues = json.loads(analyzed.read_text(encoding="utf-8"))
    d = json.loads(r.stdout)
    assert d["schema"] == "stemcue.diagnose/1"
    assert d["cues"] == str(analyzed.resolve())
    assert d["source_kind"] == "stems"
    assert d["grid_source"] == "beat_this"
    assert d["beats"] == len(cues["beats"])
    assert d["bars"] == len(cues["bars"])
    assert sum(d["bar_lengths"].values()) == len(cues["bars"])
    assert abs(d["median_ibi_s"] - 0.5) <= 0.02
    assert d["tempo_runs"] == []
    assert isinstance(d["look_windows"], list)
    assert isinstance(d["warnings"], list)


def test_cuts_quantises_candidates_to_frames(analyzed: Path) -> None:
    r = run_cli("cuts", str(analyzed), "--fps", "24")

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    cues = json.loads(analyzed.read_text(encoding="utf-8"))
    c = json.loads(r.stdout)
    assert c["schema"] == "stemcue.cuts/1"
    assert c["fps"] == 24
    cands = c["candidates"]
    assert cands
    assert [x["t"] for x in cands] == sorted(x["t"] for x in cands)
    for x in cands:
        assert x["frame"] == math.floor(x["t"] * 24 + 1e-6)
        assert x["frame_t"] == round(x["frame"] / 24, 4)
    downbeats = [b["t"] for b in cues["beats"] if b["downbeat"]]
    bars = [x for x in cands if x["kind"] in {"bar", "section"}]
    assert [x["t"] for x in bars] == downbeats
    hits = [x for x in cands if x["kind"] == "hit"]
    assert all(x["strength"] >= 0.6 for x in hits)
    hit_t = sorted(x["t"] for x in hits)
    assert all(b - a >= 0.25 for a, b in itertools.pairwise(hit_t))


def _edge(kind: str, source: str, t: float) -> dict[str, object]:
    return {"t": t, "end": None, "kind": kind, "source": source, "strength": 1.0, "stems": None, "pos": None}


def test_cuts_section_needs_two_stems_changing_not_one_stem_flickering(tmp_path: Path, analyzed: Path) -> None:
    cues = json.loads(analyzed.read_text(encoding="utf-8"))
    cues["events"] = [e for e in cues["events"] if e["kind"] not in {"enter", "exit", "stop"}]
    for bar in cues["bars"]:
        bar["mix_db"] = -10.0
    downbeats = [b["t"] for b in cues["beats"] if b["downbeat"]]
    flicker, entry = downbeats[2], downbeats[4]
    cues["events"] += [
        _edge("exit", "Brass", flicker - 0.1),
        _edge("enter", "Brass", flicker + 0.1),
        _edge("enter", "Bass", entry - 0.05),
        _edge("enter", "Brass", entry + 0.05),
    ]
    edited = tmp_path / "cues.json"
    edited.write_text(json.dumps(cues), encoding="utf-8")

    r = run_cli("cuts", str(edited), "--fps", "24")

    assert r.returncode == 0, r.stderr
    by_t = {x["t"]: x for x in json.loads(r.stdout)["candidates"] if x["kind"] in {"bar", "section"}}
    assert by_t[flicker]["kind"] == "bar"
    assert by_t[entry]["kind"] == "section"
    assert by_t[entry]["stems"] == ["Bass", "Brass"]
    assert by_t[entry]["reason"] == ["enter: Bass, Brass"]
    assert [x["t"] for x in by_t.values() if x["kind"] == "section"] == [entry]


def test_analyze_synthetic_stems_writes_cues_and_viewer(tmp_path: Path, weights_dir: Path) -> None:
    stem_dir = tmp_path / "stems"
    stem_dir.mkdir()
    write_synthetic_stems(stem_dir)
    out = tmp_path / "out"

    r = run_cli("analyze", str(stem_dir), "--out", str(out), "--weights-dir", str(weights_dir), timeout=600)

    assert r.returncode == 0, r.stderr
    cues_path = (out / "cues.json").resolve()
    viewer_path = (out / "viewer.html").resolve()
    assert r.stdout == f"cues\t{cues_path}\nviewer\t{viewer_path}\n"

    cues = json.loads(cues_path.read_text(encoding="utf-8"))
    assert cues["schema"] == "stemcue.cues/1"
    assert cues["model"] == {"checkpoint": "final0", "source_sha256": FINAL0_SHA256}
    stems = {s["name"]: s for s in cues["source"]["stems"]}
    assert list(stems) == ["Lead Vocals", "Drums", "Bass", "Brass", HOSTILE_NAME]
    assert stems["Lead Vocals"]["silent"] is True
    assert stems["Drums"]["silent"] is False
    assert abs(cues["tempo"]["bpm_median"] - 120) <= 3
    assert any(b["downbeat"] for b in cues["beats"])
    brass_hits = [e for e in cues["events"] if e["kind"] == "onset" and e["source"] == "Brass"]
    assert any(abs(e["t"] - 9.25) <= 0.03 for e in brass_hits)
    hit = min(brass_hits, key=lambda e: abs(e["t"] - 9.25))
    assert hit["pos"] is not None
    assert re.fullmatch(r"\d+\.\d+\.[1-4]", hit["pos"]["label"])
    assert len([e for e in cues["events"] if e["kind"] == "kick"]) >= 25
    assert not any(e["source"] == "Lead Vocals" for e in cues["events"])
    assert cues["envelopes"]["fps"] == 50
    assert abs(len(cues["envelopes"]["mix"]) - 16 * 50) <= 2

    html = viewer_path.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "innerHTML" not in html
    assert "<img src=x" not in html
    assert "data:audio/mpeg;base64," in html
    embedded = html.split('<script id="stemcue-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == cues


def test_import_real_checkpoint_converts_to_safetensors(tmp_path: Path, final0_ckpt: Path) -> None:
    wdir = tmp_path / "weights"

    r = run_cli("weights", "import", str(final0_ckpt), "--name", "final0", "--weights-dir", str(wdir))

    assert r.returncode == 0, r.stderr
    target = (wdir / "final0.safetensors").resolve()
    assert r.stdout == f"final0\t{target}\n"
    assert target.is_file()
    assert not list(wdir.glob("*.part"))
    with safe_open(str(target), framework="numpy") as f:
        meta = f.metadata()
        keys = list(f.keys())
    assert meta["stemcue_format"] == "1"
    assert meta["checkpoint"] == "final0"
    assert meta["source_sha256"] == FINAL0_SHA256
    assert json.loads(meta["hyper_parameters"])["transformer_dim"] > 0
    assert keys
    assert not any(k.startswith("model.") for k in keys)


class _RunsShell:
    def __init__(self, command: str) -> None:
        self.command = command

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, (self.command,))


def test_import_poisoned_checkpoint_is_refused_without_running_it(tmp_path: Path) -> None:
    sentinel = tmp_path / "pwned"
    evil = tmp_path / "evil.ckpt"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("evil/data.pkl", pickle.dumps({"state_dict": _RunsShell(f"touch '{sentinel}'")}, protocol=2))
        z.writestr("evil/byteorder", b"little")
        z.writestr("evil/version", b"3\n")
    sha = hashlib.sha256(evil.read_bytes()).hexdigest()
    wdir = tmp_path / "weights"

    r = run_cli("weights", "import", str(evil), "--name", "evil", "--sha256", sha, "--weights-dir", str(wdir))

    assert r.returncode == 3, r.stderr
    assert "disallowed global posix.system" in r.stderr
    assert r.stdout == ""
    assert not sentinel.exists()
    assert not wdir.exists() or not any(wdir.iterdir())


def _save_checkpoint(path: Path, state: dict[str, torch.Tensor], hparams: dict[str, object]) -> str:
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}, "hyper_parameters": hparams}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_refuses_storage_type_rewritten_by_the_pickle(tmp_path: Path) -> None:
    torch.manual_seed(0)
    state = BeatThis().state_dict()
    plain = tmp_path / "plain.ckpt"
    _save_checkpoint(plain, state, {})
    marker = b"ctorch\nFloatStorage\n"
    # BUILD with state ("<i4",) right after the global: rewrites the storage stand-in to int32 (same itemsize).
    rewrite = b"X\x03\x00\x00\x00<i4\x85b"
    evil = tmp_path / "rewritten.ckpt"
    with zipfile.ZipFile(plain) as src, zipfile.ZipFile(evil, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename.endswith("/data.pkl"):
                assert marker in data
                data = data.replace(marker, marker + rewrite, 1)
            dst.writestr(info.filename, data)
    sha = hashlib.sha256(evil.read_bytes()).hexdigest()
    wdir = tmp_path / "weights"

    r = run_cli("weights", "import", str(evil), "--name", "rewritten", "--sha256", sha, "--weights-dir", str(wdir))

    assert r.returncode == 3, r.stderr
    assert r.stdout == ""
    assert "Traceback" not in r.stderr
    assert not wdir.exists() or not any(wdir.iterdir())


def test_import_refuses_hyper_parameters_larger_than_the_tensors(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.ckpt"
    sha = _save_checkpoint(ckpt, {"x": torch.zeros(1)}, {"n_layers": 64})
    wdir = tmp_path / "weights"

    r = run_cli("weights", "import", str(ckpt), "--name", "tiny", "--sha256", sha, "--weights-dir", str(wdir))

    assert r.returncode == 3, r.stderr
    assert "hyper-parameters describe a model larger than the size limit" in r.stderr
    assert r.stdout == ""
    assert not wdir.exists() or not any(wdir.iterdir())


def test_import_refuses_name_that_matches_a_pin_ignoring_case(tmp_path: Path, final0_ckpt: Path) -> None:
    wdir = tmp_path / "weights"

    r = run_cli(
        "weights", "import", str(final0_ckpt), "--name", "FINAL0", "--sha256", FINAL0_SHA256, "--weights-dir", str(wdir)
    )

    assert r.returncode == 2, r.stderr
    assert "invalid checkpoint name: FINAL0" in r.stderr
    assert r.stdout == ""
    assert not wdir.exists()


GRID_FIX = {
    "schema": "stemcue.gridfix/1",
    "segments": [
        {"type": "free", "start": 0.0, "end": 2.5},
        {"type": "bar", "start": 2.5, "end": 5.5, "beats": 6},
        {"type": "fit", "start": 5.5, "end": 9.5, "hits": [{"t": 5.5, "bar": 0}, {"t": 7.5, "bar": 1}]},
        {"type": "track", "start": 9.5, "end": 16.0, "period": 0.5},
    ],
}


@pytest.fixture(scope="session")
def analyzed(tmp_path_factory: pytest.TempPathFactory, weights_dir: Path) -> Path:
    base = tmp_path_factory.mktemp("analyzed")
    stem_dir = base / "stems"
    stem_dir.mkdir()
    write_synthetic_stems(stem_dir)
    r = run_cli("analyze", str(stem_dir), "--out", str(base / "out"), "--weights-dir", str(weights_dir), timeout=600)
    if r.returncode != 0:
        pytest.fail(f"analyze failed ({r.returncode}): {r.stderr}")
    return base / "out" / "cues.json"


@pytest.fixture(scope="session")
def fix_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("fix") / "fix.json"
    path.write_text(json.dumps(GRID_FIX), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def gridded(
    tmp_path_factory: pytest.TempPathFactory, analyzed: Path, fix_file: Path
) -> tuple[subprocess.CompletedProcess[str], Path]:
    out = tmp_path_factory.mktemp("grid1") / "out"
    r = run_cli("grid", str(analyzed), "--fix", str(fix_file), "--out", str(out), timeout=300)
    return r, out


def test_grid_rebuilds_beats_bars_and_positions(
    analyzed: Path, gridded: tuple[subprocess.CompletedProcess[str], Path]
) -> None:
    r, out = gridded

    assert r.returncode == 0, r.stderr
    cues_path = (out / "cues.json").resolve()
    viewer_path = (out / "viewer.html").resolve()
    assert r.stdout == f"cues\t{cues_path}\nviewer\t{viewer_path}\n"
    before = json.loads(analyzed.read_text(encoding="utf-8"))
    cues = json.loads(cues_path.read_text(encoding="utf-8"))
    assert before["grid"]["source"] == "beat_this"
    assert before["grid"]["beat_this"]["beats"] == [b["t"] for b in before["beats"]]
    assert cues["grid"]["source"] == "fix"
    assert cues["grid"]["beat_this"] == before["grid"]["beat_this"]
    assert [s["type"] for s in cues["grid"]["report"]["segments"]] == ["free", "bar", "fit", "track"]

    bar1 = [b for b in cues["beats"] if b["bar"] == 1]
    assert [b["t"] for b in bar1] == [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert [b["beat"] for b in bar1] == [1, 2, 3, 4, 5, 6]
    assert [b["downbeat"] for b in bar1] == [True, False, False, False, False, False]
    bars = cues["bars"]
    assert [b["n"] for b in bars[:4]] == [1, 2, 3, 4]
    assert [b["beats"] for b in bars[:4]] == [6, 4, 4, 4]
    assert bars[0]["start"] == 2.5
    assert bars[1]["start"] == 5.5
    assert bars[2]["start"] == 7.5
    assert [b["t"] for b in cues["beats"] if b["bar"] == 3] == [7.5, 8.0, 8.5, 9.0]
    assert abs(bars[3]["start"] - 9.5) <= 0.03

    assert [(e["t"], e["kind"], e["source"]) for e in cues["events"]] == [
        (e["t"], e["kind"], e["source"]) for e in before["events"]
    ]
    brass = min(
        (e for e in cues["events"] if e["kind"] == "onset" and e["source"] == "Brass"), key=lambda e: abs(e["t"] - 9.25)
    )
    assert brass["pos"]["label"] == "3.4.3"
    early = [e for e in cues["events"] if e["t"] < 2.5]
    assert early
    assert all(e["pos"] is None for e in early)

    html = viewer_path.read_text(encoding="utf-8")
    embedded = html.split('<script id="stemcue-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == cues


def test_grid_on_its_own_output_gives_the_same_result(
    tmp_path: Path, gridded: tuple[subprocess.CompletedProcess[str], Path], fix_file: Path
) -> None:
    first_run, first = gridded
    assert first_run.returncode == 0, first_run.stderr
    out = tmp_path / "again"

    r = run_cli("grid", str(first / "cues.json"), "--fix", str(fix_file), "--out", str(out), timeout=300)

    assert r.returncode == 0, r.stderr
    again = json.loads((out / "cues.json").read_text(encoding="utf-8"))
    assert again == json.loads((first / "cues.json").read_text(encoding="utf-8"))


def test_look_writes_a_png(tmp_path: Path, analyzed: Path) -> None:
    png = tmp_path / "figs" / "look.png"

    r = run_cli("look", str(analyzed), "--start", "8", "--end", "11", "--out", str(png), timeout=300)

    assert r.returncode == 0, r.stderr
    assert r.stdout == f"png\t{png.resolve()}\n"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
