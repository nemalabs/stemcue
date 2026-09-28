"""cues.json (schema stemcue.cues/1)."""

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_ID: Final = "stemcue.cues/1"

EventKind = Literal["onset", "kick", "snare", "cymbal", "band_hit", "enter", "exit", "stop"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True, validate_by_name=True)


class StemInfo(_Model):
    name: str
    file: str
    silent: bool
    active_ratio: float


class Source(_Model):
    kind: Literal["stems", "file"] = "stems"
    stem_dir: str
    mix: str | None
    stems: list[StemInfo]


class AnalysisParams(_Model):
    sr: int
    hop: int


class ModelInfo(_Model):
    checkpoint: str
    source_sha256: str


class Tempo(_Model):
    bpm_median: float


class Beat(_Model):
    t: float
    bar: int
    beat: int
    downbeat: bool
    bpm: float


class Bar(_Model):
    n: int
    start: float
    end: float
    beats: int
    texture: list[str]
    mix_db: float


class BeatCheck(_Model):
    start: float
    end: float
    n: int
    agree: float
    median_offset_ms: float | None


class Pos(_Model):
    bar: int
    beat: int
    sixteenth: int
    label: str
    offset_ms: float


class Event(_Model):
    t: float
    end: float | None
    kind: EventKind
    source: str
    strength: float
    stems: list[str] | None
    pos: Pos | None


class Envelopes(_Model):
    fps: int
    mix: list[float]
    stems: dict[str, list[float]]


class RawGrid(_Model):
    beats: list[float]
    downbeats: list[float]


class FixHit(_Model):
    t: float
    bar: int


class FreeSegment(_Model):
    type: Literal["free"]
    start: float
    end: float


class BarSegment(_Model):
    type: Literal["bar"]
    start: float
    end: float
    beats: int = Field(ge=1, le=16)


class FitSegment(_Model):
    type: Literal["fit"]
    start: float
    end: float
    beats_per_bar: int = Field(default=4, ge=1, le=16)
    hits: list[FixHit] = Field(min_length=2)


class TrackSegment(_Model):
    type: Literal["track"]
    start: float
    end: float
    beats_per_bar: int = Field(default=4, ge=1, le=16)
    period: float = Field(gt=0.1, lt=2.0)
    ibi_min: float | None = None
    ibi_max: float | None = None
    tol_ms: float = Field(default=45.0, gt=0)
    fit_window: int = Field(default=6, ge=2)
    smooth_window: int = Field(default=4, ge=1)


Segment = Annotated[FreeSegment | BarSegment | FitSegment | TrackSegment, Field(discriminator="type")]

_OVERLAP_TOL_S = 1e-6


class GridFix(_Model):
    schema_id: Literal["stemcue.gridfix/1"] = Field(default="stemcue.gridfix/1", alias="schema")
    segments: list[Segment] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_segments(self) -> Self:
        for i, seg in enumerate(self.segments):
            if not seg.start < seg.end:
                msg = f"segment {i}: start must be before end"
                raise ValueError(msg)
            if isinstance(seg, FitSegment) and len({h.bar for h in seg.hits}) < 2:  # noqa: PLR2004
                msg = f"segment {i}: hits need at least two different bar numbers"
                raise ValueError(msg)
            if (
                isinstance(seg, TrackSegment)
                and seg.ibi_min is not None
                and seg.ibi_max is not None
                and not seg.ibi_min < seg.ibi_max
            ):
                msg = f"segment {i}: ibi_min must be below ibi_max"
                raise ValueError(msg)
            if i > 0 and self.segments[i - 1].end > seg.start + _OVERLAP_TOL_S:
                msg = f"segment {i} starts before segment {i - 1} ends"
                raise ValueError(msg)
        return self


class SegmentReport(_Model):
    type: Literal["free", "bar", "fit", "track"]
    start: float
    end: float
    bars: int
    beats: int
    notes: list[str]


class GridReport(_Model):
    segments: list[SegmentReport]
    downbeats_checked: int
    downbeats_missing: list[float]


class Grid(_Model):
    source: Literal["beat_this", "fix"]
    beat_this: RawGrid
    fix: GridFix | None
    report: GridReport | None


class Cues(_Model):
    schema_id: Literal["stemcue.cues/1"] = Field(default=SCHEMA_ID, alias="schema")
    source: Source
    duration: float
    analysis: AnalysisParams
    model: ModelInfo
    tempo: Tempo
    beats: list[Beat]
    bars: list[Bar]
    beat_check: list[BeatCheck]
    events: list[Event]
    envelopes: Envelopes
    grid: Grid | None = None
