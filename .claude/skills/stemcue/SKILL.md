---
name: stemcue
description: Find 音ハメ (cut-to-music) timings in a Suno stem folder or a single song file with the stemcue CLI (beat_this + librosa), then check and hand-correct the beat/bar grid (free-tempo spans, odd-meter bars, line fits to band hits, cleaned beat_this beats). Use when the user hands over Suno stems or a song to sync video cuts to, asks where the hits / downbeats / entries are, or when cues.json bars look wrong.
---

# stemcue — hit timings and a trustworthy bar grid from Suno stems or a single song file

stemcue is a CLI (`stemcue …` once installed with `uv tool install`; inside a clone of its repository use
`uv run stemcue …` instead). It reads a folder of Suno stems (plus the full mix if available) or one audio file of the whole song, finds every attack per stem with librosa, tracks beats and downbeats
with a hardened copy of beat_this, and writes `cues.json` plus a self-contained `viewer.html`. The raw beat_this grid is **not** trustworthy on real songs
(on the reference song it reported 121 downbeats where the true grid has 75). This skill is the procedure for turning the raw output
into a grid you can cut to, and for choosing the cut times.

Work from evidence: every grid decision is backed by a number from the tool or by a figure you looked at. Write down
which bars you verified by eye and which you inferred.

## 0. Setup (once)

```bash
stemcue weights fetch            # pinned final0; SHA-256 checked before parsing
```
- Weights come only through `stemcue weights fetch` / `stemcue weights import FILE --name NAME --sha256 HEX`.
  Never load a `.ckpt` with `torch.load`, never install upstream `beat_this` to "compare": the point of this tool is that
  checkpoint files are parsed by a restricted reader that cannot execute code.
- Install the CLI first if `stemcue --help` fails: `uv tool install git+https://github.com/nemalabs/stemcue`.
- Weights are installed into `~/.cache/stemcue/weights` by default (`--weights-dir` to change it).
- matplotlib writes a font cache into `MPLCONFIGDIR`; when running inside a sandbox or a project that forbids writes to
  the home directory, set `MPLCONFIGDIR` to a directory inside the project for figure commands.

## 1. Analyze

```bash
stemcue analyze "STEM_DIR" --mix MIX.wav --out OUT
```
- Pass `--mix` whenever the full song exists: beat_this was trained on mixes, and the stem sum is not the same signal.
- Output: `OUT/cues.json`, `OUT/viewer.html` (open it in a browser; it plays the mix and draws every lane).

Without stems, pass the song file itself:
```bash
stemcue analyze SONG.wav --out OUT
```
- The file is its own mix, so `--mix` is refused (exit 2). `source.kind` in cues.json is `"file"` (`"stems"` for a
  folder).
- kick / snare / cymbal come from a percussive split of the mix (librosa HPSS), with `source: "percussive"`. They are
  less precise than a drum stem: other instruments' attacks leak into the percussive part.
- Per-instrument onsets, enter / exit and multi-stem `band_hit` are not available: onsets and enter / exit describe the
  whole song, and `band_hit` is an attack peak of the full mix.

What is in `cues.json` (schema `stemcue.cues/1`):
- `events[]`: `kind` ∈ onset (per stem), kick / snare / cymbal (drum stem bands, or the percussive split of a single
  file), band_hit (several stems at once),
  enter / exit (a stem starts / stops sounding), stop (the whole band stops). `t` is the refined attack time (50 % rise of
  the envelope), not the spectral-flux peak. `strength` is the flux peak.
- `beats[]`, `bars[]` and each event's `pos` (`bar.beat.sixteenth` + `offset_ms` from that grid point).
- `beat_check[]`: per 10 s, the share of beat_this beats that a librosa dynamic-programming tracker agrees with (±40 ms).
- `grid`: the raw beat_this beats/downbeats (kept forever, even after corrections) and, after `grid`, the fix and a report.
- `envelopes`: per-stem level in dB at 50 fps.

## 2. Find what is wrong with the raw grid

Look before fixing. In this order:

1. **Numbers.** Print the diagnosis (JSON on stdout, schema `stemcue.diagnose/1`; nothing is written):
   ```bash
   stemcue diagnose OUT/cues.json
   ```
   - `median_ibi_s`, `expected_bars` (= duration / (4 × median IBI)) next to `bars`, and `bar_lengths` (beats per bar →
     count).
   - `odd_ibis`: beat intervals more than 8 % off the median (`t` = the interval's first beat, `ratio` = IBI / median).
   - `tempo_runs`: 4 or more consecutive IBIs at 1.8–2.2× (`x2_slower`) or 1/2.2–1/1.8× (`x2_faster`) the median —
     beat_this switched to half or double time there.
   - `odd_bars`: bars whose beat count differs from the most common one.
   - `low_agreement`: `beat_check` windows below 0.6 agreement.
   - `downbeats_missing`: after `grid`, the grid downbeats with no beat_this downbeat within 60 ms.
   - `look_windows`: time windows (at most 12 s, with the reasons) around everything above — the windows to open with
     `look` in step 3.
   - `warnings`: one line per problem found.

   Warning signs: far more bars than expected (beat_this calls extra downbeats), IBIs near half or double the median
   (double-time / lost beats), `beat_check` windows with low agreement or a median offset near ±half a beat, bars of
   1–3 beats scattered through the song.
2. **The map.** `stemcue overview OUT/cues.json --out OUT/overview.png` — every stem's level, 20 s per row,
   current grid in cyan (downbeats white with bar numbers). Read the song's structure first: where the band enters,
   breaks, drops out, which stems carry the accents.
3. **Zoom.** `stemcue look OUT/cues.json --start A --end B --out OUT/look_A.png` (≤ 60 s; 6–12 s windows are the
   most readable). Black = onset flux per band, grey = level, blue = beat_this beats (thick = downbeat), green ticks on
   top = current grid (tall = downbeat, number = bar), red ticks at the bottom = band hits. Judge which grid lands on the
   real hits, entries and drop-outs. When statistics are weak, the figure decides.

Always look at: the intro (free tempo is common), every break, every place the texture changes, and every window
flagged in step 1.

## 3. Write a fix file

A fix is a list of non-overlapping segments in time order (`stemcue.gridfix/1`):

```json
{
  "schema": "stemcue.gridfix/1",
  "segments": [
    {"type": "free", "start": 0.0, "end": 16.333},
    {"type": "fit", "start": 16.333, "end": 27.5, "beats_per_bar": 4,
     "hits": [{"t": 16.333, "bar": 0}, {"t": 18.189, "bar": 1}, {"t": 20.054, "bar": 2}, {"t": 23.766, "bar": 4}, {"t": 25.632, "bar": 5}]},
    {"type": "bar", "start": 27.5, "end": 30.38, "beats": 6},
    {"type": "track", "start": 30.38, "end": 159.92, "beats_per_bar": 4, "period": 0.4638, "ibi_min": 0.43, "ibi_max": 0.50}
  ]
}
```
(This is the fix of the reference song; it reproduces its hand-made grid — see section 6.)

- `free`: no beats; events inside get `pos: null`. Use for rubato intros, fermatas, held endings, stop-time hits that no
  steady pulse explains. Evidence: counting beats between hits gives non-integers; extrapolating the later grid puts the
  hits between grid points; beat_this checkpoints disagree.
- `fit`: a straight line through chosen bar-start hits (`bar` = bar index within the segment, gaps allowed). Use where
  beat_this is unreliable but the band plays strict bar-start hits. Choose hits that start bars; skip anticipations
  (an 8th early) and irregular fills. Residuals should be a few ms.
- `bar`: one bar of N equal beats (6-beat break bars, 2-beat pickups, 3/4 bars). An odd bar shifts every later
  downbeat — the usual reason why the start and the end of a song disagree on bar phase.
- `track`: beat_this beats, cleaned and smoothed: beats whose neighbour intervals fall outside `[ibi_min, ibi_max]` are
  dropped as anchors, anchors are numbered by `round(gap / period)`, anchors more than `tol_ms` (45) off the local line over
  ±`fit_window` (6) beats are dropped (up to 3 passes), then every beat is placed on a local line over ±`smooth_window` (4)
  beats. Bars start at the beat nearest `start` — so `start` sets the downbeat phase. `period` = the median beat period of
  the section (measure it; do not guess).

```bash
stemcue grid OUT/cues.json --fix fix.json --out OUT2
```
`grid` always rebuilds from the raw beat_this lists stored in cues.json, so re-running it on its own output with a new
fix is safe. It refuses to write into the input's own directory. Report lines (stderr) per segment, then the count of
grid downbeats with no beat_this downbeat within 60 ms. Fix-file mistakes (overlapping segments, a fit with a single bar
number, beats of two segments within 1 ms of each other) stop with exit 4 and a message naming the segment.

## 4. Check the corrected grid

- Track segments: anchor residual median ≲ 10 ms, p95 ≲ 20 ms, max ≲ 30 ms is good (reference song: 6.7 / 16.3 / 28.4). A long
  "dropped" list in one region means beat_this lost the beat there — look at it.
- Fit segments: hit residuals within ±5 ms; the predicted end should meet the next segment's start (reference song: 27.489 vs 27.5).
- Downbeat check: grid downbeats with no beat_this downbeat within 60 ms. A few misses in one place = look there. Many =
  wrong phase (move the track `start` by one beat) or a missing odd bar before it.
- `look` again over every boundary and every flagged window: section entries, band re-entries and drop-outs should sit on
  green downbeats. `overview` again for the whole song.
- Beat period drift of ±50 ms between sections is normal for these songs; a single straight line over the whole song is
  not (the first, wrong hand-made grid of the reference song was one line: 0.7 s = 1.5 beats off in one section).
- librosa's DP tracker (`beat_check`) can lock onto off-beat percussion and sit half a beat off (reference song 110–130 s:
  agreement 0.00, median −201 ms) while beat_this is right. Low agreement is a reason to look, not a verdict.

## 5. Choosing cut times (音ハメ)

Start from the candidates (JSON on stdout, schema `stemcue.cuts/1`; nothing is written):
```bash
stemcue cuts OUT/cues.json --fps 24        # --min-gap 0.25 --min-strength 0.6 by default
```
- One candidate per downbeat of the current grid: `kind: "section"` when at least two stems enter or exit within
  ±0.35 s, not counting a stem that has both an enter and an exit in that window (in either order: a dropout or a
  short burst), the bar's `mix_db` jumps by 6 dB or more against the previous bar, or a full stop ends within ±0.35 s
  (`reason` says which); `kind: "bar"` otherwise. `snap_t` (sections only) is the strongest onset within ±0.35 s of a
  stem that enters there without also exiting there, else the strongest band_hit there, else null.
- `kind: "hit"`: band hits with `strength` ≥ `--min-strength`, thinned strongest first to at least `--min-gap` seconds
  apart. `anticipation: true` = the hit sits on the bar's last beat at sixteenth 3 or 4 (it lands before the bar);
  `free: true` = no grid position (free-tempo span or outside the grid). Cut both at `t`, not at a grid point.
- `frame` = the frame index at `--fps`, truncated (floor), and `frame_t` = its start time. `t` is the unquantised time.
- The command does not apply the 0.2 ms nudge below, and it does not know about the opening or about stem leakage: the
  rules below still apply to its output.

- Section cuts go on downbeats of the corrected grid. Before using a strong hit as a section cut, read its `pos` label:
  many strong hits are anticipations or off-beats (a label like `N.4.4` or `N.4.3` = late in the bar's last beat), which
  are great for accents but land before the bar.
- Anticipations, off-beat hits and anything in a free span: cut at the event's raw `t`, not at a grid point.
- Quantise every cue to the video frame rate (24 fps: 1/24 s) and compare boundaries only after quantising. When writing
  start times into a composition, truncate rather than round (rounding up can hide the first frame); nudge frame-exact
  events 0.2 ms earlier.
- Differences of 25–30 ms between trackers are under one frame at 24 fps; do not chase them.
- Thin drum-driven cues to ≥ 0.25 s spacing to avoid flicker; snap section bounds to the strongest hit of the entering
  stem within ±0.35 s.
- The opening does not have to follow the music: an opening with impact can matter more than one locked to the beat.
- Suno stems are separated roughly: instruments leak into each other (on the reference song, guitar and brass were mixed together).
  Use the stems as evidence, not as truth; confirm an entry on the mix envelope too.

## 6. Reference song

The hand-made grid of the reference song (a 160 s Suno song; 301 beats, 75 bars of 3, 4 or 6 beats) was built bar by bar:
- 0–16.333 s free (solo piano rubato, stop-time hits at 5.93 / 6.27 / 6.71, a free build),
- 16.333–27.5 s: six 4/4 bars fitted to five band hits (bar 1.8595 s = 129.07 BPM, residuals ≤ 4 ms),
- 27.5–30.38 s: one 6-beat break bar (moves the downbeat phase by two beats),
- 30.38 s–end: beat_this final0 beats cleaned (45 ms / ±6 beats) and smoothed (±4 beats); beat_this lost the beat around
  53.6–58.7 s (±40 ms there).
The fix in section 3 reproduces it with stemcue; comparison numbers: see `NOTES.md` next to this file.

## Rules

- Never trust the raw downbeats; always run steps 2–4 before handing cut times to anyone.
- Keep the raw analysis directory; write each correction to a new `--out` directory.
- State in your report which boundaries you checked with `look` and which you did not.
