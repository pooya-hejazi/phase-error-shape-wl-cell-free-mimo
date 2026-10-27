"""Command-line interface for validated, manifest-producing experiment runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigurationError, load_run_config
from .experiment import run_experiment


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase-graph-cme")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_project_root(),
        help="project root; defaults to the installed source checkout",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--acknowledge-locked-preregistration",
        action="store_true",
        help="required in addition to the hash-locked file for FINAL_CAMPAIGN",
    )
    return parser


def _resolve_config(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    config_path = _resolve_config(root, args.config)
    try:
        config = load_run_config(config_path)
        if args.command == "validate-config":
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "experiment_id": config.experiment_id,
                        "stage": config.stage,
                        "evidence_eligible": config.evidence_eligible,
                        "config_path": str(config_path),
                    },
                    indent=2,
                )
            )
            return 0
        if (
            config.stage == "FINAL_CAMPAIGN"
            and not args.acknowledge_locked_preregistration
        ):
            raise RuntimeError(
                "FINAL_CAMPAIGN requires --acknowledge-locked-preregistration"
            )
        summary = run_experiment(
            config, project_root=root, config_path=config_path
        )
        print(json.dumps(summary, indent=2))
        return 0
    except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
