from __future__ import annotations

import argparse
import json

from diamondgo.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DiamondGo self-play data.")
    parser.add_argument("--print-config", action="store_true", help="Print the default config as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig()
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return
    print("Self-play loop is scaffolded. Next step: implement PUCT rollouts.")


if __name__ == "__main__":
    main()
