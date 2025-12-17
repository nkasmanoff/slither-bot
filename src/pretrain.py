"""Supervised pre-training of the Slither.io policy network.

Supports two modes:
1. Human trajectories: Load JSON files from slither_human_collect.py
2. Auto mode: Collect trajectories using rules-based policy

The resulting checkpoint can be loaded by either REINFORCE or A2C agents.
"""

import glob
import json
import math
import os
import time
from datetime import datetime
from random import shuffle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from selenium.webdriver.common.by import By
from torch.utils.data import DataLoader, Dataset, random_split

from .agents import PolicyNetwork
from .controller import SlitherController
from .utils import (
    NUM_ACTIONS,
    angle_to_discrete_action,
    extract_observation,
    setup_browser,
    wait_for_game_ready,
)


DANGER_DISTANCE = 300
SAFE_DISTANCE = 500


class TrajectoryDataset(Dataset):
    """Dataset over (observation, action) pairs from trajectory episodes.

    Actions are discrete indices (0 to NUM_ACTIONS-1).
    """

    def __init__(self, data_dir: str, max_files: int | None = None):
        self.observations: List[List[float]] = []
        self.actions: List[int] = []

        pattern = os.path.join(data_dir, "*.json")
        all_paths = sorted(glob.glob(pattern))
        if max_files is not None:
            all_paths = all_paths[:max_files]

        if not all_paths:
            raise ValueError(f"No JSON files found in '{data_dir}'")

        for path in all_paths:
            with open(path, "r") as f:
                data = json.load(f)

            for step in data.get("steps", []):
                obs = step.get("observation")
                action = step.get("action")

                if obs is None or action is None:
                    continue
                if not isinstance(obs, list) or not isinstance(action, (int, float)):
                    continue

                self.observations.append(obs)
                # Ensure action is an integer index
                self.actions.append(int(action))

        if not self.observations:
            raise ValueError(f"No valid (observation, action) pairs in '{data_dir}'")

        combined = list(zip(self.observations, self.actions))
        shuffle(combined)
        self.observations, self.actions = zip(*combined)
        self.observations = list(self.observations)
        self.actions = list(self.actions)
        self.state_dim = len(self.observations[0])

        print(f"Loaded {len(self.observations)} samples from {len(all_paths)} file(s)")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        act = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, act


def collect_auto_trajectories(
    output_dir: str = "trajectories",
    num_episodes: int = 5,
    max_steps: int = 1000,
    poll_interval: float = 0.1,
) -> List[str]:
    """Collect trajectories automatically using the rules-based policy.

    Args:
        output_dir: Directory for saving episode files.
        num_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.
        poll_interval: Seconds between samples.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    driver = setup_browser()

    print(f"\nCollecting {num_episodes} episodes with rules-based policy...")

    for episode_idx in range(num_episodes):
        play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
        for button in play_buttons:
            if "Play" in button.text:
                button.click()
                break

        print(f"Episode {episode_idx + 1}/{num_episodes} started")
        time.sleep(1)

        controller = SlitherController(
            driver, save_screenshots=False, record_video=False
        )
        wait_for_game_ready(controller)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        steps = []
        step_idx = 0
        fleeing = False
        prev_length = controller.get_snake_length()
        need_browser_restart = False
        last_length = prev_length
        no_change_count = 0
        STALL_THRESHOLD = 50

        while True:
            if controller.is_game_over():
                print(f"Episode {episode_idx + 1} game over!")
                break

            state = controller.get_detailed_state()
            if not state:
                controller.maintain_direction(0.5)
                step_idx += 1
                continue

            obs = extract_observation(state)
            snake_angle = state["snake"].get("angle") if state.get("snake") else None

            current_length = controller.get_snake_length()
            length_increase = current_length - prev_length
            # Match the reward shaping from environment.py
            reward = length_increase * 15.0 - 5.0

            other_snakes = state.get("other_snakes", [])
            nearest_enemy = other_snakes[0] if other_snakes else None
            foods = state.get("foods", [])

            # Determine target angle based on rules policy
            target_angle_deg = None
            if nearest_enemy and nearest_enemy["distance"] < DANGER_DISTANCE:
                fleeing = True
                # Flee: target is opposite direction from enemy
                angle_to_enemy_rad = nearest_enemy["angle_to"]
                target_angle_deg = (math.degrees(angle_to_enemy_rad) + 180) % 360
            elif (
                fleeing and nearest_enemy and nearest_enemy["distance"] < SAFE_DISTANCE
            ):
                angle_to_enemy_rad = nearest_enemy["angle_to"]
                target_angle_deg = (math.degrees(angle_to_enemy_rad) + 180) % 360
            else:
                fleeing = False
                if foods:
                    # Chase food: target is direction to nearest food
                    nearest_food = foods[0]
                    target_angle_deg = math.degrees(nearest_food["angle"]) % 360

            # Convert to discrete action index
            action = None
            if target_angle_deg is not None:
                action = angle_to_discrete_action(target_angle_deg)
                controller.move_to_angle(target_angle_deg)

            if current_length == last_length:
                no_change_count += 1
            else:
                no_change_count = 0
                last_length = current_length

            done = no_change_count >= STALL_THRESHOLD or step_idx >= max_steps

            if done and no_change_count >= STALL_THRESHOLD:
                print(f"Episode {episode_idx + 1} stalled. Ending...")
                need_browser_restart = True
                reward -= 50.0
                reward += current_length * 0.5

            step_data = {
                "t": step_idx,
                "observation": obs.tolist(),
                "action": action,
                "snake_angle_rad": (
                    float(snake_angle) if snake_angle is not None else None
                ),
                "snake_length": int(current_length),
                "length_increase": int(length_increase),
                "reward": float(reward),
                "done": bool(done),
            }
            steps.append(step_data)

            prev_length = current_length
            step_idx += 1

            if done:
                break

            time.sleep(poll_interval)

        duration = time.time() - start_time
        final_length = steps[-1]["snake_length"] if steps else 0

        episode_data = {
            "metadata": {
                "episode_index": episode_idx,
                "timestamp": timestamp,
                "policy": "rules_based",
                "poll_interval": poll_interval,
                "max_steps": max_steps,
                "num_steps": len(steps),
                "final_length": final_length,
                "duration_seconds": round(duration, 3),
            },
            "steps": steps,
        }

        filename = f"auto_episode_{episode_idx + 1}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)

        saved_files.append(filepath)
        print(
            f"Saved episode {episode_idx + 1} ({len(steps)} steps, length={final_length})"
        )

        if need_browser_restart:
            print("Restarting browser...")
            driver.quit()
            driver = setup_browser()
        else:
            time.sleep(2)

    driver.quit()
    print(f"\nCollected {len(saved_files)} episodes to {output_dir}")
    return saved_files


def train_supervised(
    data_dir: str,
    output_path: str = "models/pretrained.pt",
    batch_size: int = 64,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    val_split: float = 0.1,
    max_files: int | None = None,
    seed: int = 42,
) -> str:
    """Train PolicyNetwork by behavior cloning from trajectory data.

    Uses cross-entropy loss for discrete action classification.

    Args:
        data_dir: Directory containing trajectory JSON files.
        output_path: Path to save the trained model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        val_split: Fraction of data for validation.
        max_files: Optional limit on number of files to load.
        seed: Random seed.

    Returns:
        Path to saved model checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TrajectoryDataset(data_dir=data_dir, max_files=max_files)
    state_dim = dataset.state_dim

    if 0.0 < val_split < 1.0 and len(dataset) > 10:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"Split: {train_size} train, {val_size} val samples")
    else:
        train_ds, val_ds = dataset, None
        print(f"Using all {len(dataset)} samples for training")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = PolicyNetwork(state_dim, num_actions=NUM_ACTIONS).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    def evaluate(loader):
        policy.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                logits = policy(obs_batch)
                loss = criterion(logits, act_batch)
                total_loss += loss.item() * obs_batch.size(0)
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == act_batch).sum().item()
                total_samples += obs_batch.size(0)
        accuracy = total_correct / max(1, total_samples)
        return total_loss / max(1, total_samples), accuracy

    for epoch in range(1, epochs + 1):
        policy.train()
        running_loss, running_correct, running_samples = 0.0, 0, 0

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()
            logits = policy(obs_batch)
            loss = criterion(logits, act_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * obs_batch.size(0)
            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == act_batch).sum().item()
            running_samples += obs_batch.size(0)

        train_loss = running_loss / max(1, running_samples)
        train_acc = running_correct / max(1, running_samples)

        if val_loader:
            val_loss, val_acc = evaluate(val_loader)
            print(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )
        else:
            print(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f}, train_acc={train_acc:.3f}"
            )

    models_dir = os.path.dirname(output_path) or "."
    os.makedirs(models_dir, exist_ok=True)

    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "source": "supervised_pretrain",
            "data_dir": data_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "state_dim": state_dim,
            "num_actions": NUM_ACTIONS,
        },
    }

    torch.save(checkpoint, output_path)
    print(f"Saved pre-trained model to {output_path}")
    return output_path
