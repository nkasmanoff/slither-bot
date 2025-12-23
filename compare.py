#!/usr/bin/env python3
"""Run inference comparison between different agent setups.

Examples:
    # Run comparison with 3 games per setup for default agents
    python compare.py --games 3

    # Run comparison for specific agents
    python compare.py --agents rules,a2c_last,ppo_last --games 10
"""

import argparse

from src.compare import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare Slither.io agent performance")
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of games per setup (default: 5)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="rules,a2c_random,a2c_last,ppo_last,reinforce_last",
        help="Comma-separated list of agents to compare (default: rules,a2c_random,a2c_last,ppo_last,reinforce_last)",
    )

    args = parser.parse_args()
    
    # Convert comma-separated string to list
    agent_list = [a.strip() for a in args.agents.split(",")]
    
    run_comparison(
        num_games=args.games,
        agents=agent_list
    )


if __name__ == "__main__":
    main()
