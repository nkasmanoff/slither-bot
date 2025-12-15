"""Supervised pre-training of the Slither.io policy network on human trajectories.

This script trains the same `PolicyNetwork` architecture used in `slither_rl.py`
using the JSON trajectory files produced by `slither_human_collect.py`.

The resulting checkpoint is saved in the **same format** as
`REINFORCEAgent.save_model`, so it can be loaded directly via
`REINFORCEAgent.load_model` to warm-start RL.

Typical workflow:

1. Collect human trajectories:

       python slither_human_collect.py --episodes 5 --output-dir human_trajectories

2. Pre-train the policy network on those trajectories:

       python slither_human_pretrain.py \
           --data-dir human_trajectories \
           --epochs 5 \
           --batch-size 64 \
           --output models/human_pretrained.pt

3. Warm-start RL by pointing `slither_rl.train_agent` at the checkpoint
   (see comments in `slither_rl.py` or set `SLITHER_PRETRAINED_MODEL`).
"""

import argparse
import glob
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from slither_rl import PolicyNetwork


class HumanTrajectoryDataset(Dataset):
    """Dataset over (observation, action_index) pairs from human episodes.

    Expects JSON files produced by `slither_human_collect.py` with structure:

    {
        "metadata": {...},
        "steps": [
            {
                "observation": [...],  # length 11
                "action_index": int | null,
                ...
            },
            ...
        ]
    }
    """

    def __init__(self, data_dir: str, max_files: int | None = None):
        self.observations: List[List[float]] = []
        self.actions: List[int] = []

        pattern = os.path.join(data_dir, "*.json")
        all_paths = sorted(glob.glob(pattern))
        if max_files is not None:
            all_paths = all_paths[:max_files]

        if not all_paths:
            raise ValueError(f"No JSON files found in data_dir='{data_dir}'")

        for path in all_paths:
            with open(path, "r") as f:
                data = json.load(f)

            for step in data.get("steps", []):
                obs = step.get("observation")
                action = step.get("action_index")

                if obs is None or action is None:
                    continue

                # Basic sanity checks
                if not isinstance(obs, list):
                    continue
                if not isinstance(action, int):
                    continue

                self.observations.append(obs)
                self.actions.append(action)

        if not self.observations:
            raise ValueError(
                f"No valid (observation, action_index) pairs found in '{data_dir}'. "
                "Check that your human_trajectories were recorded correctly."
            )

        # Infer dimensions
        self.state_dim = len(self.observations[0])
        self.num_actions = int(max(self.actions)) + 1

        print(
            f"Loaded {len(self.observations)} samples from {len(all_paths)} file(s). "
            f"State dim = {self.state_dim}, num_actions = {self.num_actions}."
        )

        # Optional: print a rough class distribution
        counts = np.bincount(self.actions, minlength=self.num_actions)
        total = counts.sum()
        dist_str = ", ".join(
            f"a{idx}={count} ({count / total:.1%})" for idx, count in enumerate(counts)
        )
        print(f"Action distribution: {dist_str}")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        act = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, act


def train_supervised(
    data_dir: str,
    output_path: str = "models/human_pretrained.pt",
    batch_size: int = 64,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    val_split: float = 0.1,
    max_files: int | None = None,
    seed: int = 42,
) -> str:
    """Train PolicyNetwork by behavior cloning from human data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = HumanTrajectoryDataset(data_dir=data_dir, max_files=max_files)

    state_dim = dataset.state_dim
    action_dim = dataset.num_actions

    # Split into train/val
    if 0.0 < val_split < 1.0 and len(dataset) > 10:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"Dataset split: {train_size} train, {val_size} val samples.")
    else:
        train_ds, val_ds = dataset, None
        print(f"Using all {len(dataset)} samples for training (no val split).")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Since PolicyNetwork outputs probabilities (softmax), use NLLLoss on log-probs
    criterion = nn.NLLLoss()

    def evaluate(loader: DataLoader) -> Tuple[float, float]:
        policy.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)

                probs = policy(obs_batch)
                log_probs = torch.log(probs + 1e-8)
                loss = criterion(log_probs, act_batch)

                total_loss += loss.item() * obs_batch.size(0)
                preds = probs.argmax(dim=-1)
                total_correct += (preds == act_batch).sum().item()
                total_samples += obs_batch.size(0)

        avg_loss = total_loss / max(1, total_samples)
        acc = total_correct / max(1, total_samples)
        return avg_loss, acc

    for epoch in range(1, epochs + 1):
        policy.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()
            probs = policy(obs_batch)
            log_probs = torch.log(probs + 1e-8)
            loss = criterion(log_probs, act_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * obs_batch.size(0)
            preds = probs.argmax(dim=-1)
            running_correct += (preds == act_batch).sum().item()
            running_samples += obs_batch.size(0)

        train_loss = running_loss / max(1, running_samples)
        train_acc = running_correct / max(1, running_samples)

        if val_loader is not None:
            val_loss, val_acc = evaluate(val_loader)
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )
        else:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}"
            )

    # Save checkpoint in a format compatible with REINFORCEAgent.load_model
    models_dir = os.path.dirname(output_path) or "."
    os.makedirs(models_dir, exist_ok=True)

    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "source": "human_supervised_pretrain",
            "data_dir": data_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "state_dim": state_dim,
            "action_dim": action_dim,
        },
    }

    torch.save(checkpoint, output_path)
    print(f"Saved pre-trained model to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Supervised pre-training of the Slither.io policy network "
            "on human trajectories collected with slither_human_collect.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="human_trajectories",
        help="Directory containing human trajectory JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/human_pretrained.pt",
        help="Path to save the pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for supervised training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of supervised training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (0 disables validation).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optionally limit the number of JSON files loaded.",
    )

    args = parser.parse_args()

    train_supervised(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        val_split=args.val_split,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
