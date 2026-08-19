"""Command line interface for the DeSide PyTorch/Lightning 2.x line.

Subcommands
-----------

``train``
    Train a DeSide model from scratch using a YAML configuration file that
    follows the VAEDecon layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DESIDE_ROOT = Path(__file__).resolve().parents[2]
if str(DESIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(DESIDE_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deside",
        description="DeSide (PyTorch/Lightning 2.x): deconvolution by deep learning and single-cell references.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a DeSide model from scratch using a YAML configuration file."
    )
    train_parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to a DeSide YAML configuration file (e.g., deside/configs/example_config.yaml).",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for training.output_dir in the YAML.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from deside.configs import DeSideConfig
        from deside.workflow import train_from_config

        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        config = DeSideConfig.from_yaml(config_path)
        if args.output_dir:
            config.training = dict(config.training)
            config.training["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        train_from_config(config, config_file_path=config_path)
        print(f"DeSide training completed. Model saved under model_dir={config.model_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
