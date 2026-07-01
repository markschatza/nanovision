from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from nanovision_dataset.inspect import audit_run, export_contact_sheet, export_html_viewer, export_snippet
from nanovision_dataset.minatar_source import DEFAULT_GAMES, MinAtarSource
from nanovision_dataset.pgx_source import PgxBaselineSource
from nanovision_dataset.writer import artifact_to_dict, write_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanovision-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a MinAtar grayscale frame backlog.")
    generate.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    generate.add_argument("--episodes", type=int, default=1)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--max-steps", type=int, default=1000)
    generate.add_argument("--policy", choices=["random", "pgx-baseline"], default="random")
    generate.add_argument("--baseline-dir", type=str, default="artifacts/pgx-baselines")
    generate.add_argument("--jax-cache-dir", type=str, default="artifacts/jax-cache")
    generate.add_argument("--batch-size", type=int, default=16)
    generate.add_argument("--out", type=Path, required=True)
    generate.set_defaults(func=_generate)

    audit = subparsers.add_parser("audit", help="Audit a saved dataset run.")
    audit.add_argument("run_dir", type=Path)
    audit.set_defaults(func=_audit)

    html = subparsers.add_parser("export-html", help="Export a self-contained HTML playback viewer.")
    html.add_argument("run_dir", type=Path)
    html.add_argument("--out", type=Path, required=True)
    html.add_argument("--max-frames", type=int, default=512)
    html.set_defaults(func=_export_html)

    snippet = subparsers.add_parser("export-snippet", help="Export an animated GIF snippet.")
    snippet.add_argument("run_dir", type=Path)
    snippet.add_argument("--out", type=Path, required=True)
    snippet.add_argument("--max-frames", type=int, default=64)
    snippet.add_argument("--scale", type=int, default=16)
    snippet.set_defaults(func=_export_snippet)

    sheet = subparsers.add_parser("export-contact-sheet", help="Export a PNG contact sheet.")
    sheet.add_argument("run_dir", type=Path)
    sheet.add_argument("--out", type=Path, required=True)
    sheet.add_argument("--max-frames", type=int, default=64)
    sheet.add_argument("--columns", type=int, default=8)
    sheet.add_argument("--scale", type=int, default=16)
    sheet.set_defaults(func=_export_contact_sheet)
    return parser


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if args.policy == "pgx-baseline":
        source = PgxBaselineSource(
            max_steps=args.max_steps,
            baseline_dir=args.baseline_dir,
            jax_cache_dir=args.jax_cache_dir,
            batch_size=args.batch_size,
        )
    else:
        source = MinAtarSource(max_steps=args.max_steps, policy_source="random")
    records = source.rollout_games(args.games, episodes=args.episodes, seed=args.seed)
    artifact = write_run(
        args.out,
        records,
        policy_source=source.policy_source,
        settings={
            "action_space": "pgx-minatar" if args.policy == "pgx-baseline" else "minatar-native",
            "baseline_dir": args.baseline_dir if args.policy == "pgx-baseline" else None,
            "batch_size": args.batch_size if args.policy == "pgx-baseline" else None,
            "games": args.games,
            "episodes": args.episodes,
            "jax_cache_dir": args.jax_cache_dir if args.policy == "pgx-baseline" else None,
            "policy": args.policy,
            "seed": args.seed,
            "max_steps": args.max_steps,
        },
    )
    return {"artifact": artifact_to_dict(artifact)}


def _audit(args: argparse.Namespace) -> dict[str, object]:
    return audit_run(args.run_dir)


def _export_html(args: argparse.Namespace) -> dict[str, object]:
    path = export_html_viewer(args.run_dir, args.out, max_frames=args.max_frames)
    return {"output": str(path)}


def _export_snippet(args: argparse.Namespace) -> dict[str, object]:
    path = export_snippet(args.run_dir, args.out, max_frames=args.max_frames, scale=args.scale)
    return {"output": str(path)}


def _export_contact_sheet(args: argparse.Namespace) -> dict[str, object]:
    path = export_contact_sheet(
        args.run_dir,
        args.out,
        max_frames=args.max_frames,
        columns=args.columns,
        scale=args.scale,
    )
    return {"output": str(path)}


if __name__ == "__main__":
    raise SystemExit(main())
