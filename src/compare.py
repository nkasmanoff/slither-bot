"""Compare different agent setups for inference.

Runs comparison between:
1. Rules-based policy (heuristic)
2. A2C with random weights
3. A2C with trained model weights
"""

import csv
import glob
import io
import os
from datetime import datetime

import numpy as np

from .environment import load_and_play
from .rules_policy import run_rules_based_policy


def find_latest_final_model(models_dir="models"):
    """Find the most recent final_model_a2c_*.pt file."""
    os.makedirs(models_dir, exist_ok=True)
    pattern = os.path.join(models_dir, "final_model_a2c_*.pt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No final_model_a2c_*.pt files found in {models_dir}")

    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def calculate_statistics(data):
    """Calculate statistics for a list of values."""
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def run_comparison(num_games=3):
    """Run inference comparison between different agent setups.

    Args:
        num_games: Number of games per setup.

    Returns:
        List of (name, metrics) tuples.
    """
    print("=" * 70)
    print("Agent Inference Comparison")
    print("=" * 70)

    all_metrics = []

    # Setup 1: Rules-based policy
    print("\n" + "=" * 70)
    print("Setup 1: Rules-Based Policy")
    print("=" * 70)
    metrics = run_rules_based_policy(num_games=num_games, record_video=False)
    if metrics:
        all_metrics.append(("Rules-Based Policy", metrics))

    # Setup 2: A2C with random weights
    print("\n" + "=" * 70)
    print("Setup 2: A2C with Random Weights")
    print("=" * 70)
    metrics = load_and_play(
        model_path=None, num_games=num_games, record_video=False, agent_type="a2c"
    )
    if metrics:
        all_metrics.append(("A2C Random Weights", metrics))

    # Setup 3: A2C with final model
    print("\n" + "=" * 70)
    print("Setup 3: A2C with Final Model")
    print("=" * 70)
    try:
        final_model_path = find_latest_final_model()
        print(f"Using: {final_model_path}")
        metrics = load_and_play(
            model_path=final_model_path,
            num_games=num_games,
            record_video=False,
            agent_type="a2c",
        )
        if metrics:
            all_metrics.append(("A2C Final Model", metrics))
    except FileNotFoundError as e:
        print(f"Skipping: {e}")

    # Setup 4: A2C with best model
    print("\n" + "=" * 70)
    print("Setup 4: A2C with Best Model")
    print("=" * 70)
    best_model_path = "models/best_model_a2c.pt"
    if os.path.exists(best_model_path):
        print(f"Using: {best_model_path}")
        metrics = load_and_play(
            model_path=best_model_path,
            num_games=num_games,
            record_video=False,
            agent_type="a2c",
        )
        if metrics:
            all_metrics.append(("A2C Best Model", metrics))
    else:
        print(f"Skipping: {best_model_path} not found")

    # Generate statistics
    if all_metrics:
        _save_comparison_stats(all_metrics)

    print("\n" + "=" * 70)
    print("Comparison complete! Check 'inference_logs/' for detailed metrics.")
    print("=" * 70)

    return all_metrics


def _save_comparison_stats(all_metrics):
    """Save comparison statistics to CSV."""
    print("\n" + "=" * 70)
    print("Statistics Summary")
    print("=" * 70)

    header = [
        "Run",
        "Length_Mean",
        "Length_Std",
        "Length_Max",
        "Steps_Mean",
        "Steps_Max",
        "Reward_Mean",
        "Reward_Max",
    ]
    csv_rows = [header]

    for run_name, metrics_data in all_metrics:
        length_stats = calculate_statistics(metrics_data.get("game_snake_lengths", []))
        steps_stats = calculate_statistics(metrics_data.get("game_steps", []))
        reward_stats = calculate_statistics(metrics_data.get("game_rewards", []))

        row = [
            run_name,
            f"{length_stats['mean']:.2f}",
            f"{length_stats['std']:.2f}",
            f"{length_stats['max']:.2f}",
            f"{steps_stats['mean']:.2f}",
            f"{steps_stats['max']:.2f}",
            f"{reward_stats['mean']:.2f}",
            f"{reward_stats['max']:.2f}",
        ]
        csv_rows.append(row)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(csv_rows)
    print(output.getvalue())

    csv_dir = "inference_logs"
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"comparison_stats_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nStatistics saved to: {csv_path}")
