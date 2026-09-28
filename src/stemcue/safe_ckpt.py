"""Restricted reader for PyTorch zip checkpoints that never executes code from the file.

Only the four globals a plain tensor checkpoint needs are resolved, each to a local stand-in;
tensors are rebuilt as numpy arrays from the raw storage blobs. torch is deliberately not imported here.
"""

import collections
import io
import math
import pickle
import zipfile
import zlib
from collections.abc import Mapping
from typing import Any, TypeGuard

import numpy as np

from stemcue.errors import WeightsIntegrityError

MAX_ENTRIES = 4096
MAX_ENTRY_BYTES = 268435456
MAX_TOTAL_BYTES = 536870912
MAX_PICKLE_BYTES = 4194304
MAX_DIMS = 8
MAX_STORAGE_KEY_CHARS = 20
MAX_MESSAGE_CHARS = 300
MAX_DESCRIBED_CHARS = 80
MAX_DESCRIBED_INT = 10**18
STATE_DICT_PREFIX = "model."

# find_class hands out these immutable strings instead of objects: a pickle BUILD cannot rewrite a str, so the dtype
# used for a storage is always one of these two.
_STORAGE_DTYPES: dict[str, np.dtype[Any]] = {
    "torch.FloatStorage": np.dtype("<f4"),
    "torch.LongStorage": np.dtype("<i8"),
}
_ALLOWED_DTYPES = (np.dtype(np.float32), np.dtype(np.int64))

_MALFORMED = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    pickle.UnpicklingError,
    EOFError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    OverflowError,
    RecursionError,
    UnicodeDecodeError,
    zlib.error,
    NotImplementedError,
    RuntimeError,
)


def describe(value: object) -> str:
    """Bounded text for a value taken from a weight file, safe to put into an error message."""
    if type(value) is bool or type(value) is float:
        return repr(value)
    if type(value) is int and abs(value) <= MAX_DESCRIBED_INT:
        return repr(value)
    if type(value) is str:
        return repr(value[:MAX_DESCRIBED_CHARS])
    return type(value).__name__


def _c_strides(size: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]
    return tuple(strides)


def _is_int_tuple(value: object) -> TypeGuard[tuple[int, ...]]:
    return isinstance(value, tuple) and all(type(v) is int and v >= 0 for v in value)


class _Unpickler(pickle.Unpickler):
    def __init__(self, data: bytes, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(io.BytesIO(data))
        self._archive = archive
        self._prefix = prefix
        self._storages: dict[str, np.ndarray] = {}
        self._storage_ids: set[int] = set()
        self._rebuilt_bytes = 0

    def find_class(self, module: str, name: str) -> object:
        match (module, name):
            case ("collections", "OrderedDict"):
                return collections.OrderedDict
            case ("torch._utils", "_rebuild_tensor_v2"):
                return self._rebuild_tensor_v2
            case ("torch", "FloatStorage"):
                return "torch.FloatStorage"
            case ("torch", "LongStorage"):
                return "torch.LongStorage"
        # Escaped so control characters from the file (e.g. ANSI sequences) never reach the terminal raw.
        shown = [s[:MAX_DESCRIBED_CHARS].encode("unicode_escape").decode("ascii") for s in (module, name)]
        msg = f"disallowed global {shown[0]}.{shown[1]}"
        raise WeightsIntegrityError(msg)

    def persistent_load(self, pid: object) -> np.ndarray:
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):  # noqa: PLR2004
            raise WeightsIntegrityError("unexpected persistent id")
        storage_type, key, numel = pid[1], pid[2], pid[4]
        if not (type(storage_type) is str and storage_type in _STORAGE_DTYPES):
            raise WeightsIntegrityError("unexpected storage type")
        if not (isinstance(key, str) and len(key) <= MAX_STORAGE_KEY_CHARS and key.isascii() and key.isdigit()):
            raise WeightsIntegrityError("unexpected storage key")
        if type(numel) is not int or numel < 0:
            raise WeightsIntegrityError("unexpected storage size")
        dtype = _STORAGE_DTYPES[storage_type]
        cached = self._storages.get(key)
        if cached is not None:
            if cached.dtype != dtype or cached.size != numel:
                msg = f"storage {key} referenced with conflicting type or size"
                raise WeightsIntegrityError(msg)
            return cached
        entry = f"{self._prefix}data/{key}"
        try:
            raw = self._archive.read(entry)
        except KeyError:
            msg = f"missing storage blob {describe(entry)}"
            raise WeightsIntegrityError(msg) from None
        if len(raw) != numel * dtype.itemsize:
            msg = f"storage blob {describe(entry)} has {len(raw)} bytes, expected {numel * dtype.itemsize}"
            raise WeightsIntegrityError(msg)
        array = np.frombuffer(raw, dtype=dtype)
        self._storages[key] = array
        self._storage_ids.add(id(array))
        return array

    def _rebuild_tensor_v2(  # noqa: PLR0913, PLR0917 - called positionally by the pickle, like torch's
        self,
        storage: object,
        storage_offset: object,
        size: object,
        stride: object,
        requires_grad: object,
        backward_hooks: object,
        metadata: object = None,
    ) -> np.ndarray:
        del metadata
        if not (isinstance(storage, np.ndarray) and id(storage) in self._storage_ids):
            raise WeightsIntegrityError("tensor storage is not a checkpoint storage")
        if type(storage_offset) is not int or storage_offset < 0:
            raise WeightsIntegrityError("bad tensor storage offset")
        if not (_is_int_tuple(size) and _is_int_tuple(stride)):
            raise WeightsIntegrityError("bad tensor size or stride")
        if len(size) != len(stride) or len(size) > MAX_DIMS:
            raise WeightsIntegrityError("bad tensor rank")
        if type(requires_grad) is not bool:
            raise WeightsIntegrityError("bad tensor requires_grad")
        if not (isinstance(backward_hooks, Mapping) and len(backward_hooks) == 0):
            raise WeightsIntegrityError("tensor has backward hooks")
        expected = _c_strides(size)
        if any(n > 1 and s != e for n, s, e in zip(size, stride, expected, strict=True)):
            raise WeightsIntegrityError("non-contiguous tensor stride")
        n = math.prod(size)
        if storage_offset + n > storage.size:
            raise WeightsIntegrityError("tensor exceeds its storage")
        # Many tensors may view one large storage; cap the total copied so a small pickle cannot exhaust memory.
        self._rebuilt_bytes += n * storage.dtype.itemsize
        if self._rebuilt_bytes > MAX_TOTAL_BYTES:
            raise WeightsIntegrityError("tensors exceed the size limit")
        return storage[storage_offset : storage_offset + n].reshape(size).astype(storage.dtype.newbyteorder("="))


def _pickle_entry(archive: zipfile.ZipFile) -> str:
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise WeightsIntegrityError("checkpoint has too many entries")
    if any(i.file_size > MAX_ENTRY_BYTES for i in infos):
        raise WeightsIntegrityError("checkpoint entry too large")
    if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
        raise WeightsIntegrityError("checkpoint too large")
    pkl = [i for i in infos if i.filename.endswith("/data.pkl") and i.filename.count("/") == 1]
    if len(pkl) != 1:
        raise WeightsIntegrityError("checkpoint must contain exactly one <name>/data.pkl")
    if pkl[0].file_size > MAX_PICKLE_BYTES:
        raise WeightsIntegrityError("checkpoint data.pkl too large")
    return pkl[0].filename


def _validate(result: object) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not isinstance(result, dict) or "state_dict" not in result or "hyper_parameters" not in result:
        raise WeightsIntegrityError("checkpoint lacks state_dict or hyper_parameters")
    raw_state, hparams = result["state_dict"], result["hyper_parameters"]
    if not isinstance(raw_state, Mapping) or not isinstance(hparams, dict):
        raise WeightsIntegrityError("checkpoint state_dict or hyper_parameters has the wrong type")
    tensors: dict[str, np.ndarray] = {}
    for key, value in raw_state.items():
        if not (isinstance(key, str) and key.startswith(STATE_DICT_PREFIX)):
            msg = f"unexpected state_dict key {describe(key)}"
            raise WeightsIntegrityError(msg)
        if not isinstance(value, np.ndarray):
            msg = f"state_dict entry {describe(key)} is not a tensor"
            raise WeightsIntegrityError(msg)
        if value.dtype not in _ALLOWED_DTYPES:
            msg = f"state_dict entry {describe(key)} has dtype {value.dtype}"
            raise WeightsIntegrityError(msg)
        if value.dtype == np.float32 and not np.isfinite(value).all():
            msg = f"state_dict entry {describe(key)} has non-finite values"
            raise WeightsIntegrityError(msg)
        tensors[key.removeprefix(STATE_DICT_PREFIX)] = value
    return tensors, {str(k): v for k, v in hparams.items()}


def read_checkpoint(data: bytes) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return (state_dict without the "model." prefix, hyper_parameters) of a PyTorch zip checkpoint's bytes."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            pkl_name = _pickle_entry(archive)
            prefix = pkl_name.removesuffix("data.pkl")
            names = set(archive.namelist())
            if f"{prefix}byteorder" not in names or archive.read(f"{prefix}byteorder") != b"little":
                raise WeightsIntegrityError("checkpoint byteorder is missing or not little-endian")
            result = _Unpickler(archive.read(pkl_name), archive, prefix).load()
        return _validate(result)
    except _MALFORMED as exc:
        msg = f"malformed checkpoint: {type(exc).__name__}: {str(exc)[:MAX_MESSAGE_CHARS]}"
        raise WeightsIntegrityError(msg) from exc
