"""Pinned beat_this weights: download, verify, convert to safetensors, and load."""

import hashlib
import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, TypedDict

import safetensors
import safetensors.numpy
import torch

from stemcue.errors import InputError, NetworkError, UsageError, WeightsIntegrityError
from stemcue.safe_ckpt import MAX_TOTAL_BYTES, describe, read_checkpoint
from stemcue.vendor.beat_this.model import BeatThis

BASE_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
ALLOWED_HOST = "cloud.cp.jku.at"
PINNED = {
    "final0": "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331",
    "final1": "365b553f43750717c907f32fbc42910f3b264d616654583df6e44472c93ead80",
    "final2": "8f810c0a44d3a979b0372b00fa61b4e3f83a8f2a6d963c7791b9d4852cdd1abc",
}
MAX_CHECKPOINT_BYTES = 134217728
DEFAULT_WEIGHTS_DIR = Path("~/.cache/stemcue/weights")
WEIGHTS_DIR_ENV = "STEMCUE_WEIGHTS_DIR"
FORMAT_VERSION = "1"
TIMEOUT_S = 60
DOWNLOAD_DEADLINE_S = 600
CHUNK_BYTES = 1 << 20
MAX_REDIRECTS = 3
HP_INT_MAX = 4096
HTTP_OK = 200

_NAME_RE = re.compile(r"[a-z0-9_-]{1,64}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_INT_KEYS = ("spect_dim", "transformer_dim", "ff_mult", "n_layers", "head_dim", "stem_dim")
_BOOL_KEYS = ("sum_head", "partial_transformers")

log = logging.getLogger("stemcue")


class Dropout(TypedDict):
    frontend: float
    transformer: float


class HyperParameters(TypedDict):
    spect_dim: int
    transformer_dim: int
    ff_mult: int
    n_layers: int
    head_dim: int
    stem_dim: int
    dropout: Dropout
    sum_head: bool
    partial_transformers: bool


def _defaults() -> dict[str, object]:
    return {
        "spect_dim": 128,
        "transformer_dim": 512,
        "ff_mult": 4,
        "n_layers": 6,
        "head_dim": 32,
        "stem_dim": 32,
        "dropout": {"frontend": 0.1, "transformer": 0.2},
        "sum_head": True,
        "partial_transformers": True,
    }


def validate_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name):
        msg = f"invalid checkpoint name: {name}"
        raise UsageError(msg)
    return name


def resolve_hparams(raw: Mapping[str, object]) -> HyperParameters:
    """Keep the BeatThis init parameters, fill absent ones with BeatThis defaults, and validate them."""
    hp = {key: raw.get(key, default) for key, default in _defaults().items()}
    ints: dict[str, int] = {}
    for key in _INT_KEYS:
        value = hp[key]
        if type(value) is not int or not 1 <= value <= HP_INT_MAX:
            msg = f"bad hyper-parameter {key}: {describe(value)}"
            raise WeightsIntegrityError(msg)
        ints[key] = value
    if ints["transformer_dim"] % ints["head_dim"] != 0:
        raise WeightsIntegrityError("bad hyper-parameters: transformer_dim is not a multiple of head_dim")
    dropout = hp["dropout"]
    if not (isinstance(dropout, dict) and set(dropout) == {"frontend", "transformer"}):
        msg = f"bad hyper-parameter dropout: {describe(dropout)}"
        raise WeightsIntegrityError(msg)
    rates: dict[str, float] = {}
    for key, value in dropout.items():
        if type(value) not in (int, float) or not 0 <= value <= 1:
            msg = f"bad hyper-parameter dropout.{key}: {describe(value)}"
            raise WeightsIntegrityError(msg)
        rates[key] = float(value)
    flags: dict[str, bool] = {}
    for key in _BOOL_KEYS:
        value = hp[key]
        if type(value) is not bool:
            msg = f"bad hyper-parameter {key}: {describe(value)}"
            raise WeightsIntegrityError(msg)
        flags[key] = value
    return HyperParameters(
        spect_dim=ints["spect_dim"],
        transformer_dim=ints["transformer_dim"],
        ff_mult=ints["ff_mult"],
        n_layers=ints["n_layers"],
        head_dim=ints["head_dim"],
        stem_dim=ints["stem_dim"],
        dropout=Dropout(frontend=rates["frontend"], transformer=rates["transformer"]),
        sum_head=flags["sum_head"],
        partial_transformers=flags["partial_transformers"],
    )


def _new_model(hp: HyperParameters) -> BeatThis:
    try:
        return BeatThis(
            spect_dim=hp["spect_dim"],
            transformer_dim=hp["transformer_dim"],
            ff_mult=hp["ff_mult"],
            n_layers=hp["n_layers"],
            head_dim=hp["head_dim"],
            stem_dim=hp["stem_dim"],
            dropout=dict(hp["dropout"]),
            sum_head=hp["sum_head"],
            partial_transformers=hp["partial_transformers"],
        )
    except (AssertionError, ValueError, TypeError, RuntimeError) as exc:
        msg = f"hyper-parameters rejected by BeatThis: {exc}"
        raise WeightsIntegrityError(msg) from exc


def _check_shapes(hp: HyperParameters, state: dict[str, torch.Tensor]) -> None:
    """Compare size, names, shapes and dtypes on the meta device, so no real memory is allocated for a bad file."""
    with torch.device("meta"):
        shape_model = _new_model(hp)
    size = sum(t.numel() * t.element_size() for t in [*shape_model.parameters(), *shape_model.buffers()])
    if size > MAX_TOTAL_BYTES:
        raise WeightsIntegrityError("hyper-parameters describe a model larger than the size limit")
    expected = shape_model.state_dict()
    missing = expected.keys() - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        msg = f"weights do not match the model: {len(missing)} missing and {len(unexpected)} unexpected tensors"
        raise WeightsIntegrityError(msg)
    for key, want in expected.items():
        got = state[key]
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            msg = f"weights do not match the model: {describe(key)}"
            raise WeightsIntegrityError(msg)


def _build_model(hp: HyperParameters, state: dict[str, torch.Tensor]) -> BeatThis:
    _check_shapes(hp, state)
    model = _new_model(hp)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        msg = f"weights do not match the model: {exc}"
        raise WeightsIntegrityError(msg) from exc
    for key, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            msg = f"tensor {key} has non-finite values"
            raise WeightsIntegrityError(msg)
    return model.eval()


def _convert(data: bytes, name: str, sha: str, weights_dir: Path) -> Path:
    """Parse checkpoint bytes with the restricted reader, prove they load strictly, then write <name>.safetensors."""
    tensors, raw_hp = read_checkpoint(data)
    hp = resolve_hparams(raw_hp)
    _build_model(hp, {k: torch.from_numpy(v) for k, v in tensors.items()})
    target = weights_dir / f"{name}.safetensors"
    part = weights_dir / f"{name}.safetensors.part"
    metadata = {
        "stemcue_format": FORMAT_VERSION,
        "checkpoint": name,
        "source_sha256": sha,
        "hyper_parameters": json.dumps(hp, sort_keys=True),
    }
    try:
        weights_dir.mkdir(parents=True, exist_ok=True)
        try:
            safetensors.numpy.save_file(tensors, part, metadata=metadata)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    # safetensors reports I/O failures as SafetensorError, which is not an OSError.
    except (OSError, safetensors.SafetensorError) as exc:
        msg = f"cannot write weights directory {weights_dir}: {exc}"
        raise InputError(msg) from exc
    return target.resolve()


def import_checkpoint(file: Path, name: str, sha256: str | None, weights_dir: Path) -> Path:
    """Verify a checkpoint already on disk against its expected SHA-256 and convert it."""
    validate_name(name)
    given = sha256.lower() if sha256 is not None else None
    if given is not None and not _SHA_RE.fullmatch(given):
        msg = f"invalid --sha256: expected 64 hex characters, got {sha256}"
        raise UsageError(msg)
    pinned = PINNED.get(name)
    if pinned is None and given is None:
        msg = f"{name} is not pinned; pass --sha256"
        raise UsageError(msg)
    if pinned is not None and given is not None and given != pinned:
        msg = f"--sha256 differs from the pinned hash for {name}"
        raise UsageError(msg)
    expected = pinned or given
    if not file.is_file():
        msg = f"cannot read checkpoint file: {file}"
        raise InputError(msg)
    try:
        with file.open("rb") as fh:
            data = fh.read(MAX_CHECKPOINT_BYTES + 1)
    except (FileNotFoundError, IsADirectoryError, PermissionError) as exc:
        msg = f"cannot read checkpoint file: {file}"
        raise InputError(msg) from exc
    if len(data) > MAX_CHECKPOINT_BYTES:
        msg = f"checkpoint file exceeds {MAX_CHECKPOINT_BYTES} bytes"
        raise WeightsIntegrityError(msg)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        msg = f"sha256 mismatch for {name}: expected {expected}, got {actual}"
        raise WeightsIntegrityError(msg)
    log.info("verified %s (sha256 %s); converting", file, actual)
    # The bytes that were hashed are the bytes that get parsed; FILE is not opened again.
    return _convert(data, name, actual, weights_dir)


class _PinnedRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(  # noqa: PLR0913, PLR0917 - signature fixed by urllib
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or target.hostname != ALLOWED_HOST:
            reason = f"refused redirect to {newurl}"
            raise urllib.error.URLError(reason)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(name: str) -> tuple[bytes, str]:
    """Read BASE_URL/<name>.ckpt into memory; return the body and its SHA-256."""
    url = f"{BASE_URL}/{name}.ckpt"
    opener = urllib.request.build_opener(_PinnedRedirectHandler())
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
        with opener.open(url, timeout=TIMEOUT_S) as resp:
            if resp.status != HTTP_OK:
                msg = f"download failed for {name}: HTTP status {resp.status}"
                raise NetworkError(msg)
            # read1 returns whatever has arrived, so a server trickling bytes cannot hold one call past the deadline.
            while chunk := resp.read1(CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_CHECKPOINT_BYTES:
                    msg = f"download failed for {name}: body exceeds {MAX_CHECKPOINT_BYTES} bytes"
                    raise NetworkError(msg)
                digest.update(chunk)
                chunks.append(chunk)
                if time.monotonic() > deadline:
                    msg = f"download failed for {name}: not finished within {DOWNLOAD_DEADLINE_S} s"
                    raise NetworkError(msg)
    except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
        msg = f"download failed for {name}: {exc}"
        raise NetworkError(msg) from exc
    return b"".join(chunks), digest.hexdigest()


def _fetch_one(name: str, weights_dir: Path) -> Path:
    target = weights_dir / f"{name}.safetensors"
    if target.is_file():
        return target.resolve()
    log.info("fetching weights %s from %s", name, ALLOWED_HOST)
    body, actual = _download(name)
    if actual != PINNED[name]:
        msg = f"sha256 mismatch for {name}: expected {PINNED[name]}, got {actual}"
        raise WeightsIntegrityError(msg)
    return _convert(body, name, actual, weights_dir)


def fetch(names: list[str], weights_dir: Path) -> list[tuple[str, Path]]:
    """Install each pinned checkpoint that is not installed yet; return (name, safetensors path) pairs."""
    for name in names:
        validate_name(name)
        if name not in PINNED:
            msg = f"unknown checkpoint {name}; pinned: {', '.join(PINNED)}"
            raise UsageError(msg)
    return [(name, _fetch_one(name, weights_dir)) for name in names]


def ensure_installed(name: str, weights_dir: Path) -> None:
    """Fetch a pinned checkpoint when missing; an unpinned one must have been imported."""
    validate_name(name)
    if (weights_dir / f"{name}.safetensors").is_file():
        return
    if name not in PINNED:
        msg = f"weights not installed: {name} (run: stemcue weights import FILE --name {name} --sha256 HEX)"
        raise InputError(msg)
    _fetch_one(name, weights_dir)


def load_model(name: str, weights_dir: Path) -> tuple[BeatThis, str]:
    """Load <name>.safetensors into an eval-mode BeatThis; return it with the source checkpoint's SHA-256."""
    validate_name(name)
    path = weights_dir / f"{name}.safetensors"
    try:
        with safetensors.safe_open(str(path), framework="pt") as fh:
            meta = fh.metadata() or {}
            state = {key: fh.get_tensor(key) for key in fh.keys()}  # noqa: SIM118 - safe_open is not a Mapping
    except safetensors.SafetensorError as exc:
        msg = f"malformed weights file {path}: {exc}"
        raise WeightsIntegrityError(msg) from exc
    if meta.get("stemcue_format") != FORMAT_VERSION or meta.get("checkpoint") != name:
        msg = f"weights file {path} is not a stemcue {name} file"
        raise WeightsIntegrityError(msg)
    sha = meta.get("source_sha256", "")
    if name in PINNED and sha != PINNED[name]:
        msg = f"weights file {path} was not converted from the pinned {name} checkpoint"
        raise WeightsIntegrityError(msg)
    try:
        raw_hp = json.loads(meta.get("hyper_parameters", ""))
    except ValueError as exc:
        msg = f"weights file {path} has malformed hyper-parameters: {exc}"
        raise WeightsIntegrityError(msg) from exc
    if not isinstance(raw_hp, dict):
        msg = f"weights file {path} has malformed hyper-parameters"
        raise WeightsIntegrityError(msg)
    model = _build_model(resolve_hparams(raw_hp), state)
    return model, sha


def default_weights_dir() -> Path:
    """$STEMCUE_WEIGHTS_DIR when set and not empty, else ~/.cache/stemcue/weights."""
    return Path(os.environ.get(WEIGHTS_DIR_ENV) or DEFAULT_WEIGHTS_DIR).expanduser()
