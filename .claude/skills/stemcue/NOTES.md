# stemcue skill — reference measurements

Verified = produced by a command in the session that built this skill (2026-09-28) and checked there.
Reported = taken from the notes of the earlier project that built the hand-made reference grid, not re-measured.

The reference song is a 160 s Suno song (stems plus the full mix) with a rubato piano intro, a 6-beat break bar and a
band section at about 129 BPM. Its grid was first built by hand, bar by bar, in an earlier project.

## The fix in SKILL.md section 3 against the hand-made grid (verified)

Commands:
```bash
stemcue analyze "SONG Stems" --mix SONG.wav --out raw
stemcue grid raw/cues.json --fix fix.json --out fixed      # fix.json = the fix in SKILL.md section 3
```

| | raw beat_this (final0, mix) | after the fix | hand-made grid |
|---|---|---|---|
| beats | 380 | 301 | 301 |
| bars / downbeats | 121 | 75 | 75 |
| beats per bar | mixed | 4 × 73, 6 × 1, 3 × 1 (same order as the hand-made grid) | same |
| beat time difference to the hand-made grid | — | median 0.0 ms, max 0.1 ms | — |
| bar start difference to the hand-made grid | — | max 0.0 ms | — |

`grid` report lines (stderr), in this order:
- free 0.000–16.333: no beats
- fit 16.333–27.500: `bar 1.8595 s = 129.07 BPM; first bar 16.332; hit residuals ms [1.0, -2.5, 3.0, -4.0, 2.5]; ends 27.489`
- bar 27.500–30.380: `6 beats of 0.4800 s`
- track 30.380–159.920: `anchors 249, dropped [37.24, 37.72, 38.18, 54.08, 54.56, 55.6, 35.88, 53.14, 53.62], beats 271;
  anchor residual ms med 6.7 p95 16.3 max 28.4; beat period 0.4208..0.5018`
- downbeats with no beat_this downbeat within 60 ms: 1 of 73 `[21.91]` (a bar start that the hand-made grid confirmed by
  eye; beat_this simply did not call a downbeat there — a miss in a fit segment is not by itself an error)

Checked by eye in `look` 26–32 s: the 6-beat bar (bar 7) starts on the keyboard entry at 27.5 s and bar 8 on the drum and
bass re-entry at 30.38 s. In `overview`: bar 15 at 43.4 s (brass/keyboard hit), bar 23 at 58.32 s (brass entry),
bar 53 at 113.97 s (band re-entry) — the same bar numbers as the song's structure notes.

In the viewer (browser run): the playhead label reads `-` inside the free intro, `7.2.1` at 28.0 s (second beat of the
6-beat bar) and `16.1.1` at 45.3 s (the 45.26 s re-entry). The viewer's label and the event `pos.label` differ for 2 of
3903 events, both on a sixteenth boundary (beat times are stored rounded to 4 decimals).

## What the dropped anchors mean (reported)
- 35.88 / 37.24–38.18: gap-based bridging failed here in the first hand-made attempt; these beat_this beats sit off the
  local line.
- 53.14–55.6: beat_this picked double time around 55–57 s and loses the beat; the grid there is interpolated
  (about ±40 ms per beat).
