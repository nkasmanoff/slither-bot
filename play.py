#!/usr/bin/env python3
"""Play Slither.io with trained models or rules-based policy.

Examples:
    # Play with best A2C model
    python play.py --agent a2c --model models/best_model_a2c.pt

    # Play with rules-based policy
    python play.py --agent rules

    # Play with random weights (baseline)
    python play.py --agent a2c

    # Record gameplay videos
    python play.py --agent a2c --model models/best_model_a2c.pt --record-video
"""

import argparse

from src.environment import load_and_play
from src.rules_policy import run_rules_based_policy


def main():
    parser = argparse.ArgumentParser(description="Play Slither.io with trained agents")
    parser.add_argument(
        "--agent",
        type=str,
        default="a2c",
        choices=["reinforce", "a2c", "rules"],
        help="Agent type to use (default: a2c)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (not used for rules-based)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of games to play (default: 5)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record gameplay videos",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.1,
        help="Delay between actions in seconds (default: 0.1)",
    )

    args = parser.parse_args()

    if args.agent == "rules":
        run_rules_based_policy(num_games=args.games, record_video=args.record_video)
    else:
        load_and_play(
            model_path=args.model,
            num_games=args.games,
            record_video=args.record_video,
            agent_type=args.agent,
            action_delay=args.action_delay,
        )


if __name__ == "__main__":
    main()
