#!/usr/bin/env python3
"""Run inference comparison between different agent setups.

Examples:
    # Run comparison with 3 games per setup
    python compare.py --games 3

    # Run comparison with 5 games per setup
    python compare.py --games 5
"""

import argparse

from src.compare import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare Slither.io agent performance")
    parser.add_argument(
        "--games",
        type=int,
        default=3,
        help="Number of games per setup (default: 3)",
    )

    args = parser.parse_args()
    run_comparison(num_games=args.games)


if __name__ == "__main__":
    main()
