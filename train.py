#!/usr/bin/env python3
"""Train RL agents on Slither.io.

Examples:
    # Train with A2C (recommended)
    python train.py --algorithm a2c --episodes 50

    # Train with REINFORCE
    python train.py --algorithm reinforce --episodes 50

    # Train with pre-trained model
    python train.py --pretrained models/pretrained.pt

    # Record training videos
    python train.py --record-video
"""

import argparse

from src.environment import train_agent, train_agent_a2c


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on Slither.io")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="a2c",
        choices=["reinforce", "a2c"],
        help="RL algorithm to use (default: a2c)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (default: 100)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="N-step update interval for A2C (default: 64)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record gameplay videos",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.15,
        help="Delay between actions in seconds (default: 0.15)",
    )

    args = parser.parse_args()

    if args.algorithm == "a2c":
        train_agent_a2c(
            num_episodes=args.episodes,
            record_video=args.record_video,
            pretrained_model_path=args.pretrained,
            n_steps=args.n_steps,
            action_delay=args.action_delay,
        )
    else:
        train_agent(
            num_episodes=args.episodes,
            record_video=args.record_video,
            pretrained_model_path=args.pretrained,
            action_delay=args.action_delay,
        )


if __name__ == "__main__":
    main()
