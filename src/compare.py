"""Compare different agent setups for inference.

Runs comparison between various agent types and models.
"""

import csv
import glob
import io
import os
from datetime import datetime

import numpy as np

from .environment import load_and_play
from .rules_policy import run_rules_based_policy


def find_latest_final_model(models_dir="models", agent_type="a2c"):
    """Find the most recent final_model_{agent_type}_*.pt file."""
    os.makedirs(models_dir, exist_ok=True)
    pattern = os.path.join(models_dir, f"final_model_{agent_type}_*.pt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No final_model_{agent_type}_*.pt files found in {models_dir}"
        )

    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def calculate_statistics(data):
    """Calculate statistics for a list of values."""
    if not data:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
        }

    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def run_comparison(num_games=3, agents=None):
    """Run inference comparison between different agent setups.

    Args:
        num_games: Number of games per setup.
        agents: List of agent strings to compare. Supported:
                'rules', 'a2c_random', 'a2c_last', 'a2c_best',
                'ppo_random', 'ppo_last', 'ppo_best',
                'reinforce_random', 'reinforce_last', 'reinforce_best'

    Returns:
        List of (name, metrics) tuples.
    """
    if agents is None:
        agents = ["rules", "a2c_random", "a2c_last", "ppo_last", "reinforce_last"]

    print("=" * 70)
    print(f"Agent Inference Comparison (n={num_games})")
    print("=" * 70)

    all_metrics = []

    for agent_id in agents:
        print("\n" + "=" * 70)
        print(f"Running: {agent_id}")
        print("=" * 70)

        try:
            if agent_id == "rules":
                metrics = run_rules_based_policy(num_games=num_games, record_video=False)
                if metrics:
                    all_metrics.append(("Rules-Based Policy", metrics))

            elif agent_id.endswith("_random"):
                agent_type = agent_id.replace("_random", "")
                metrics = load_and_play(
                    model_path=None,
                    num_games=num_games,
                    record_video=False,
                    agent_type=agent_type,
                )
                if metrics:
                    all_metrics.append((f"{agent_type.upper()} Random", metrics))

            elif agent_id.endswith("_last"):
                agent_type = agent_id.replace("_last", "")
                model_path = find_latest_final_model(agent_type=agent_type)
                print(f"Using: {model_path}")
                metrics = load_and_play(
                    model_path=model_path,
                    num_games=num_games,
                    record_video=False,
                    agent_type=agent_type,
                )
                if metrics:
                    all_metrics.append((f"{agent_type.upper()} Last", metrics))

            elif agent_id.endswith("_best"):
                agent_type = agent_id.replace("_best", "")
                model_path = f"models/best_model_{agent_type}.pt"
                if os.path.exists(model_path):
                    print(f"Using: {model_path}")
                    metrics = load_and_play(
                        model_path=model_path,
                        num_games=num_games,
                        record_video=False,
                        agent_type=agent_type,
                    )
                    if metrics:
                        all_metrics.append((f"{agent_type.upper()} Best", metrics))
                else:
                    print(f"Skipping: {model_path} not found")
            else:
                print(f"Unknown agent type: {agent_id}")

        except Exception as e:
            print(f"Error running {agent_id}: {e}")

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

    # Define the statistics we want to collect
    stats_to_collect = [
        "Mean",
        "Median",
        "Std",
        "Q1",
        "Q3",
        "Max",
        "Min",
    ]
    metrics_to_track = ["Length", "Steps", "Reward"]

    header = ["Run"]
    for metric in metrics_to_track:
        for stat in stats_to_collect:
            header.append(f"{metric}_{stat}")

    csv_rows = [header]

    for run_name, metrics_data in all_metrics:
        stats_data = {
            "Length": calculate_statistics(metrics_data.get("game_snake_lengths", [])),
            "Steps": calculate_statistics(metrics_data.get("game_steps", [])),
            "Reward": calculate_statistics(metrics_data.get("game_rewards", [])),
        }

        row = [run_name]
        for metric in metrics_to_track:
            m_stats = stats_data[metric]
            row.extend(
                [
                    f"{m_stats['mean']:.2f}",
                    f"{m_stats['median']:.2f}",
                    f"{m_stats['std']:.2f}",
                    f"{m_stats['q1']:.2f}",
                    f"{m_stats['q3']:.2f}",
                    f"{m_stats['max']:.2f}",
                    f"{m_stats['min']:.2f}",
                ]
            )
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
