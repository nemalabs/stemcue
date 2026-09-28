"""stemcue command line: argument parsing, output lines and exit codes around the core functions."""

import argparse
import logging
import sys
from pathlib import Path

from stemcue import figures, weights
from stemcue.cuts import cut_candidates
from stemcue.diagnose import diagnose
from stemcue.errors import InputError, NetworkError, UsageError, WeightsIntegrityError
from stemcue.grid import apply_fix
from stemcue.pipeline import analyze

EXIT_USAGE = 2
EXIT_WEIGHTS = 3
EXIT_INPUT = 4
EXIT_NETWORK = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stemcue", description="Find hit points in Suno stems or a single song file.")
    sub = parser.add_subparsers(dest="command", required=True)

    an = sub.add_parser("analyze", help="analyze a stem folder or one song file into cues.json and viewer.html")
    an.add_argument("input", metavar="INPUT", help="a stem folder, or a single audio file of the whole song")
    an.add_argument("--out", required=True, help="output directory (created with parents)")
    an.add_argument(
        "--mix", help="audio for beat tracking and playback when INPUT is a stem folder (default: sum of the stems)"
    )
    an.add_argument("--checkpoint", default="final0", help="weights name (default: final0)")
    an.add_argument("--weights-dir", default=str(weights.DEFAULT_WEIGHTS_DIR), help="default: %(default)s")

    wt = sub.add_parser("weights", help="install beat_this weights")
    wsub = wt.add_subparsers(dest="weights_command", required=True)
    fe = wsub.add_parser("fetch", help="download pinned checkpoints and convert them to safetensors")
    fe.add_argument("names", nargs="*", metavar="NAME", default=["final0"])
    fe.add_argument("--weights-dir", default=str(weights.DEFAULT_WEIGHTS_DIR), help="default: %(default)s")
    im = wsub.add_parser("import", help="verify and convert a checkpoint file already on disk")
    im.add_argument("file", metavar="FILE")
    im.add_argument("--name", required=True)
    im.add_argument("--sha256", help="expected SHA-256 (64 hex); required for names that are not pinned")
    im.add_argument("--weights-dir", default=str(weights.DEFAULT_WEIGHTS_DIR), help="default: %(default)s")

    gr = sub.add_parser("grid", help="rebuild beats, bars and positions from a grid fix file")
    gr.add_argument("cues", metavar="CUES")
    gr.add_argument("--fix", required=True, help="grid fix JSON (schema stemcue.gridfix/1)")
    gr.add_argument("--out", required=True, help="output directory (created with parents)")

    lk = sub.add_parser("look", help="plot band envelopes and grids over a time window to PNG")
    lk.add_argument("cues", metavar="CUES")
    lk.add_argument("--start", type=float, required=True, help="window start in seconds")
    lk.add_argument("--end", type=float, required=True, help="window end in seconds (at most 60 s after --start)")
    lk.add_argument("--out", required=True, help="PNG path (parent created with parents)")

    ov = sub.add_parser("overview", help="plot every stem's level over the whole song to PNG")
    ov.add_argument("cues", metavar="CUES")
    ov.add_argument("--out", required=True, help="PNG path (parent created with parents)")

    dg = sub.add_parser("diagnose", help="check the beat/bar grid of a cues.json and print the findings as JSON")
    dg.add_argument("cues", metavar="CUES", help="cues.json written by analyze or grid")

    ct = sub.add_parser(
        "cuts",
        help="print cut-time candidates (downbeats, section starts, strong hits) quantised to a frame rate as JSON",
    )
    ct.add_argument("cues", metavar="CUES", help="cues.json written by analyze or grid")
    ct.add_argument("--fps", type=float, required=True, help="video frame rate, e.g. 24 or 29.97")
    ct.add_argument(
        "--min-gap", type=float, default=0.25, help="minimum seconds between kept hit candidates (default: %(default)s)"
    )
    ct.add_argument(
        "--min-strength", type=float, default=0.6, help="band_hit strength threshold (default: %(default)s)"
    )
    return parser


def _setup_logging() -> None:
    logger = logging.getLogger("stemcue")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("stemcue: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _run(args: argparse.Namespace) -> None:
    if args.command == "grid":
        cues_path, viewer_path = apply_fix(Path(args.cues), Path(args.fix), Path(args.out))
        sys.stdout.write(f"cues\t{cues_path}\nviewer\t{viewer_path}\n")
        return
    if args.command == "look":
        png = figures.look(Path(args.cues), args.start, args.end, Path(args.out))
        sys.stdout.write(f"png\t{png}\n")
        return
    if args.command == "overview":
        png = figures.overview(Path(args.cues), Path(args.out))
        sys.stdout.write(f"png\t{png}\n")
        return
    if args.command == "diagnose":
        sys.stdout.write(diagnose(Path(args.cues)).model_dump_json(indent=2) + "\n")
        return
    if args.command == "cuts":
        cuts = cut_candidates(Path(args.cues), args.fps, args.min_gap, args.min_strength)
        sys.stdout.write(cuts.model_dump_json(indent=2) + "\n")
        return
    # Only analyze and weights define --weights-dir.
    weights_dir = Path(args.weights_dir).expanduser()
    if args.command == "analyze":
        cues_path, viewer_path = analyze(
            Path(args.input),
            Path(args.out),
            Path(args.mix) if args.mix is not None else None,
            args.checkpoint,
            weights_dir,
        )
        sys.stdout.write(f"cues\t{cues_path}\nviewer\t{viewer_path}\n")
    elif args.weights_command == "fetch":
        for name, path in weights.fetch(args.names, weights_dir):
            sys.stdout.write(f"{name}\t{path}\n")
    else:
        path = weights.import_checkpoint(Path(args.file), args.name, args.sha256, weights_dir)
        sys.stdout.write(f"{args.name}\t{path}\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _setup_logging()
    try:
        _run(args)
    except UsageError as exc:
        code = EXIT_USAGE
        message = str(exc)
    except WeightsIntegrityError as exc:
        code = EXIT_WEIGHTS
        message = str(exc)
    except InputError as exc:
        code = EXIT_INPUT
        message = str(exc)
    except NetworkError as exc:
        code = EXIT_NETWORK
        message = str(exc)
    else:
        return 0
    sys.stderr.write(f"stemcue: error: {message}\n")
    return code
