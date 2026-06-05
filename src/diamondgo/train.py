from __future__ import annotations

import argparse
import json

from diamondgo.config import ExperimentConfig
from diamondgo.model import PolicyValueNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DiamondGo policy/value checkpoints.")
    parser.add_argument("--print-config", action="store_true", help="Print the default config as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig()
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return

    model = PolicyValueNet(config.board.size, config.model)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Initialized {config.board.size}x{config.board.size} model with {params:,} parameters.")


if __name__ == "__main__":
    main()
