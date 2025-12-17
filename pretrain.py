#!/usr/bin/env python3
"""Pre-train policy network on collected trajectories.

Examples:
    # Pre-train on existing trajectories
    python pretrain.py --data-dir trajectories --epochs 50

    # Collect auto trajectories then pre-train
    python pretrain.py --auto --auto-episodes 10 --epochs 50

    # Custom output path
    python pretrain.py --data-dir trajectories --output models/my_pretrained.pt
"""

import argparse

from src.pretrain import collect_auto_trajectories, train_supervised


def main():
    parser = argparse.ArgumentParser(description="Pre-train Slither.io policy network")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="trajectories",
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pretrained.pt",
        help="Path to save the pre-trained model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Collect trajectories using rules-based policy first",
    )
    parser.add_argument(
        "--auto-episodes",
        type=int,
        default=5,
        help="Number of episodes to collect in auto mode (default: 5)",
    )

    args = parser.parse_args()

    if args.auto:
        print("=== Collecting trajectories with rules-based policy ===")
        collect_auto_trajectories(
            output_dir=args.data_dir,
            num_episodes=args.auto_episodes,
        )

    print("\n=== Training Policy Network ===")
    train_supervised(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
