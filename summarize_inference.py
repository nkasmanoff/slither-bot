#!/usr/bin/env python3
"""Summarize all inference logs in the inference_logs/ directory.

This script reads all JSON files in inference_logs/, calculates statistics
for each run, and generates a combined summary CSV, similar to compare.py.
"""

import csv
import glob
import io
import json
import os
from datetime import datetime

import numpy as np


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


def summarize_all_logs():
    """Read all JSON files in inference_logs and generate summary."""
    log_dir = "inference_logs"
    json_files = glob.glob(os.path.join(log_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {log_dir}")
        return

    print(f"Found {len(json_files)} inference log files.")
    
    all_metrics = []

    for file_path in sorted(json_files):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Use filename as run name
            run_name = os.path.basename(file_path).replace(".json", "")
            
            all_metrics.append((run_name, data))
            print(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_metrics:
        return

    save_summary_stats(all_metrics)


def save_summary_stats(all_metrics):
    """Save comparison statistics to CSV, identical logic to src/compare.py."""
    print("\n" + "=" * 70)
    print("Global Inference Summary")
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

    # Print to console
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(csv_rows)
    print(output.getvalue())

    # Save to file
    csv_dir = "inference_logs"
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"summary_inference_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nSummary saved to: {csv_path}")


if __name__ == "__main__":
    summarize_all_logs()

