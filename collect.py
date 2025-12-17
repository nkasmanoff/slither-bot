#!/usr/bin/env python3
"""Collect trajectories for training.

Examples:
    # Collect human-played trajectories
    python collect.py --mode human --episodes 5

    # Collect auto trajectories using rules-based policy
    python collect.py --mode auto --episodes 10

    # Custom output directory
    python collect.py --mode auto --episodes 5 --output-dir my_trajectories
"""

import argparse

from src.collect import collect_human_trajectories
from src.pretrain import collect_auto_trajectories


def main():
    parser = argparse.ArgumentParser(description="Collect Slither.io trajectories")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["human", "auto"],
        help="Collection mode: human (manual play) or auto (rules-based)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to collect (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trajectories",
        help="Directory to save trajectory files (default: trajectories)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Sampling interval in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode (default: 2000)",
    )

    args = parser.parse_args()

    if args.mode == "human":
        collect_human_trajectories(
            num_episodes=args.episodes,
            output_dir=args.output_dir,
            poll_interval=args.poll_interval,
            max_steps=args.max_steps,
        )
    else:
        collect_auto_trajectories(
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            poll_interval=args.poll_interval,
        )


if __name__ == "__main__":
    main()
