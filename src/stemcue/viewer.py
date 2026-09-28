"""Single-file HTML viewer: the cues JSON and an MP3 of the mix embedded into viewer_template.html."""

import base64
import html
import io
import re
from importlib import resources
from pathlib import Path

import numpy as np
import soundfile

from stemcue.audio import SR
from stemcue.schema import Cues

_BACKSLASH = chr(92)
# Applied to the JSON text in this order; afterwards it holds no "<", ">" or "&", so no tag or "</script>" can
# appear inside the <script> element, and U+2028/U+2029 cannot break the script either.
_JSON_ESCAPES = (
    ("&", _BACKSLASH + "u0026"),
    ("<", _BACKSLASH + "u003c"),
    (">", _BACKSLASH + "u003e"),
    (chr(0x2028), _BACKSLASH + "u2028"),
    (chr(0x2029), _BACKSLASH + "u2029"),
)
_PLACEHOLDER = re.compile(r"__STEMCUE_(DATA|AUDIO|TITLE)__")


def _mp3_data_url(mix: np.ndarray) -> str:
    buf = io.BytesIO()
    soundfile.write(buf, np.clip(mix, -1.0, 1.0), SR, format="MP3", subtype="MPEG_LAYER_III")
    return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _script_safe_json(cues: Cues) -> str:
    text = cues.model_dump_json()
    for raw, escaped in _JSON_ESCAPES:
        text = text.replace(raw, escaped)
    return text


def render_viewer(cues: Cues, mix: np.ndarray, out_path: Path) -> None:
    template = resources.files("stemcue").joinpath("viewer_template.html").read_text(encoding="utf-8")
    source = cues.source
    title = source.stems[0].name if source.kind == "file" and source.stems else Path(source.stem_dir).name
    values = {
        "DATA": _script_safe_json(cues),
        "AUDIO": _mp3_data_url(mix),
        "TITLE": html.escape(title),
    }
    found = [m.group(1) for m in _PLACEHOLDER.finditer(template)]
    if sorted(found) != sorted(values):
        msg = f"viewer template placeholders are {found}, expected each of {sorted(values)} once"
        raise RuntimeError(msg)
    # One pass, so text inserted for one placeholder is never scanned for another placeholder.
    page = _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
    out_path.write_text(page, encoding="utf-8")
