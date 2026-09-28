# stemcue

[日本語](README.md) | English

A command-line tool that finds where to cut when you edit video to music (cutting on the beat, called 音ハメ in
Japanese).

Give it a song and it lists, with times:

- the beats (the steady pulse of the song) and the downbeats (the first beat of each bar)
- the moments the kick, snare and cymbals hit
- the moments several instruments hit hard together
- where each instrument starts and stops playing, and where the whole band stops

Everything is placed on a beat-and-bar ruler (the "grid" below). You can see that a hit falls "just before beat 3 of
bar 12", which makes it easy to decide whether a cut belongs on the bar or on the hit.

## Input

The main target is **stems** exported from Suno, the AI music service. Stems are the song split into one audio file per
instrument (vocals, drums, bass, ...). With stems, stemcue can also tell which instrument came in when.

A song without stems works too: pass the one audio file of the whole song. Accuracy is lower; how much lower was
measured on a real song (see [Stems vs a single file](#stems-vs-a-single-file)).

## What it uses

- **Beats and downbeats**: [beat_this](https://github.com/CPJKU/beat_this), an AI model published by a research group at
  JKU Linz (Johannes Kepler University, Austria) that finds beats and downbeats in music.
- **Everything else** (drum hits, instruments coming in and out, ...): [librosa](https://librosa.org/), the standard
  Python library for audio analysis.

stemcue keeps its own copy of the beat_this code and takes simple security measures when loading beat_this's trained
data (the "weights"): it checks the file's hash first (see [About the weights](#about-the-weights-and-their-hashes)), reads it with a
dedicated reader that extracts numbers only, and converts it to the safetensors format before use.

## What you get

- `cues.json`: everything found, with times. Meant for programs and AI agents.
- `viewer.html`: open it in a browser to play the song with the hits and beats drawn in one row per instrument. It is a
  single self-contained file and makes no network connections.
- `diagnose` and `cuts` print their results as JSON (a text format programs read easily), for handing the work to an AI
  agent.

A "skill" for Claude Code (Anthropic's AI coding tool) is included too. A skill is a written procedure: it teaches Claude
how to run stemcue, how to find and fix mistakes in the grid, and how to choose cut times.

## Install

You need [uv](https://docs.astral.sh/uv/) (a tool that installs and runs Python programs) and git. If Python 3.12 is
missing, uv fetches it. librosa, PyTorch (which runs the AI model) and the other libraries are installed by uv on first
run; PyTorch is large, so the first run takes a while to download.

### Inside the project that uses it (recommended)

In the top folder of the project where you want to use stemcue:

```bash
git clone https://github.com/nemalabs/stemcue tools/stemcue
mkdir -p .claude/skills
cp -R tools/stemcue/.claude/skills/stemcue .claude/skills/     # if you also want the Claude Code skill
```

Run the commands from the project's top folder like this:

```bash
uv run --project tools/stemcue stemcue --help
```

`--project tools/stemcue` means "use the stemcue in tools/stemcue"; it does not change the folder the command runs in,
so give song files relative to the project's top folder. stemcue's Python environment is created in
`tools/stemcue/.venv` and does not affect other projects.

The rest of this README writes the command as `stemcue …`; inside a project, read it as
`uv run --project tools/stemcue stemcue …`.

If you do not want `tools/stemcue/` in your project's git, add it to `.gitignore`.

### For the whole computer

To type `stemcue` from any folder:

```bash
uv tool install git+https://github.com/nemalabs/stemcue
```

If the shell then says `stemcue` is not found, the folder where uv puts commands (`uv tool dir --bin` shows it) is not on
your PATH. Run `uv tool update-shell` and open a new terminal.

### Getting the weights

```bash
stemcue weights fetch          # downloads the beat_this weights (final0) and converts them
```

You can skip this: the first `analyze` downloads them automatically.

The weights folder is chosen in this order:

1. `--weights-dir` on the command
2. the environment variable `STEMCUE_WEIGHTS_DIR`
3. otherwise `~/.cache/stemcue/weights` (inside your home folder)

To keep the weights inside a project, set `STEMCUE_WEIGHTS_DIR`. For example, `STEMCUE_WEIGHTS_DIR=.cache/stemcue/weights`
puts them in `.cache/stemcue/weights` under the folder the command runs in.

With Claude Code, put it in the project's `.claude/settings.json`; it then applies to every command Claude runs:

```json
{
  "env": {
    "STEMCUE_WEIGHTS_DIR": ".cache/stemcue/weights"
  }
}
```

### About the weights and their hashes

stemcue itself contains no weights. `stemcue weights fetch` downloads them (three versions: `final0`, `final1`,
`final2`) from the beat_this authors' public server (`cloud.cp.jku.at`). A downloaded file whose SHA-256 hash differs
from the value recorded in `src/stemcue/weights.py` in even one character is not used.

**The recorded hashes were computed from the files as downloaded on 2026-09-28. They are not hashes published by the
beat_this authors.** A match only shows that you received the same file as was downloaded on 2026-09-28; it does not
guarantee that the file is the authors' intended release. If the authors replace a file on their server, the hash no
longer matches and `fetch` stops. If you obtain weights another way, check them yourself, then register them with their
hash:

```bash
stemcue weights import FILE --name NAME --sha256 HEX
```

The weights are subject to the beat_this authors' terms; this repository's license does not cover them.

## Usage

### 1. Analyze the song

```bash
# A folder of stems. If you also have the finished song with all instruments mixed (the mix), pass it with --mix:
# beat_this was trained on mixes, so it finds the beat more reliably in the mix than in the sum of the stems.
stemcue analyze "My Song Stems" --mix "My Song.wav" --out out

# No stems: pass the one audio file of the song
stemcue analyze "My Song.wav" --out out
```

`out` then holds `cues.json` and `viewer.html`. Open `viewer.html` in a browser (on macOS: `open out/viewer.html`).

### 2. Check the grid and fix it

On real songs the grid from beat_this often cannot be used as is. Typical mistakes:

- too many downbeats
- from some point on, the tempo is taken as half the real one
- a steady beat forced onto an intro whose tempo drifts

Find and fix them with:

```bash
stemcue diagnose out/cues.json                          # suspicious spots (spans at half or double tempo, bars with the wrong number of beats, time spans worth a look)
stemcue overview out/cues.json --out out/overview.png   # one picture of the whole song: every instrument's level over time, with the grid
stemcue look out/cues.json --start 28 --end 40 --out out/look_28.png   # a zoomed picture of one time span (up to 60 s)
stemcue grid out/cues.json --fix fix.json --out fixed   # rebuild the grid according to a fix file
```

The fix file (`fix.json`) is JSON that says how to place beats in each span of the song, for example "0–16 s has a free
tempo, place no beats", "the one bar from 27.5 s has 6 beats", "from 30 s use beat_this's beats, drop the outliers and
smooth them". How to write it, how to read the pictures and how to decide are in the skill's procedure,
[`SKILL.md`](.claude/skills/stemcue/SKILL.md). It reads fine as a manual without Claude Code.

### 3. Get cut candidates

```bash
stemcue cuts fixed/cues.json --fps 24
```

Give the video frame rate (frames per second; 24 here) and you get cut candidates with frame numbers, of three kinds:

- downbeats
- section starts (a downbeat where several instruments come in or drop out, the level changes a lot, or the band
  resumes after a stop)
- strong hits (with where they fall in the bar)

### Output and exit codes

Every command explains itself with `--help`. Results go to standard output: `analyze`, `grid`, `look`, `overview` and
`weights` print "name, tab, value" lines, `diagnose` and `cuts` print JSON. Progress goes to standard error.

| Exit code | Meaning |
|---|---|
| 0 | success |
| 2 | the command was used wrongly |
| 3 | the weights failed verification |
| 4 | bad input (audio, `cues.json`, fix file) |
| 5 | network failure |

## Stems vs a single file

Without stems there is no file that says which sound is the drums. stemcue splits the song into "sustained" and
"struck" sounds (librosa's HPSS) and looks for kick, snare and cymbals in the struck part.

The loss of accuracy was measured on one Suno song (147.8 s): the song analyzed from its stems, against the same stems
summed into one file. Two detections within 30 ms of each other count as the same.

| found | from stems | from one file | stem results also found in the one file | found only in the one file (likely false) |
|---|---|---|---|---|
| beats | 328 | 322 | 322 | 0 |
| kick | 407 | 502 | 399 | 103 |
| snare | 428 | 522 | 422 | 100 |
| cymbal | 249 | 252 | 156 | 96 |
| several instruments hitting together | 181 | 333 | 126 | 207 |
| an instrument starts | 102 | 3 | — | — |
| an instrument stops | 102 | 3 | — | — |

- Beats land in nearly the same places.
- Kicks and snares are almost all found, with about 20 % extra detections. Nearly 40 % of the cymbals are missed.
- One file cannot tell which instrument came in or dropped out, so `cuts` can judge section starts only from level
  changes and stops, and finds fewer of them.
- Only one song was measured: treat the numbers as a rough guide.

## Why librosa alone is not enough

librosa is good at finding the moment a sound starts and at measuring the level of the low and high parts of the sound,
and stemcue uses it for that. For beats and bars on its own, it falls short:

- **It does not know where bars start.** librosa's beat finder (`beat_track`) returns beat positions only, not which
  beat is the first of a bar. Cutting on bar starts needs exactly that.
- **It easily takes the tempo as half or double.** On the first 30 s of the song in the table above, librosa reported
  94 BPM (94 beats per minute) with 0.63 s between beats. beat_this has 0.32 s between beats, so librosa marked every
  other beat only.
- **Its beats drift.** In the same 30 s, 9 of librosa's 48 beats were 40 ms or more (up to 162 ms) away from the
  nearest beat_this beat. At 24 fps one frame is about 42 ms, so that is up to almost 4 frames.

beat_this is not perfect either: it can switch to half tempo in the middle of a song. That is why stemcue is meant to be
used by finding suspicious spots with `diagnose` and fixing them with a fix file. librosa's beats are kept in
`cues.json` (`beat_check`) as a cross-check that points at spans where the two disagree.

## Claude Code skill

If you copied `.claude/skills/stemcue` as in [Inside the project that uses it](#inside-the-project-that-uses-it-recommended),
the skill is available when you open that project in Claude Code. With a whole-computer install, copy it to
`~/.claude/skills/` to use it in every project.

Give Claude Code a stem folder or a song file and ask, for example, "find the hits" or "I want to cut on bar starts"; it
follows the skill's procedure. The skill uses `tools/stemcue` when it exists, else a `stemcue` installed for the whole
computer; if there is neither, it tells you instead of installing anything. Unless `STEMCUE_WEIGHTS_DIR` is set, it keeps
the weights in the project's `.cache/stemcue/weights`.

## For developers

```bash
git clone https://github.com/nemalabs/stemcue && cd stemcue
uv sync
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
```

The tests run the `stemcue` command itself and check its results. They compare the converted weights with the original
file, so the original `final0.ckpt` must be at `.cache/stemcue/download/final0.ckpt`:

```bash
mkdir -p .cache/stemcue/download
curl -fL -o .cache/stemcue/download/final0.ckpt \
  https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt
uv run pytest
```

## License

stemcue is MIT-licensed ([`LICENSE`](LICENSE)). The beat_this code copied into `src/stemcue/vendor/beat_this/` is
MIT-licensed by the Institute of Computational Perception, JKU Linz (the `LICENSE` in that folder). The weights are not
in this repository and have their own terms (see [About the weights](#about-the-weights-and-their-hashes)).
