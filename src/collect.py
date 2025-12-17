"""Human trajectory collector for Slither.io.

Records human gameplay as (state, action, reward) tuples for use in
supervised pre-training or imitation learning.
"""

import json
import os
import time
from datetime import datetime

import numpy as np

from .controller import SlitherController
from .utils import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    angle_rad_to_discrete_action,
    extract_observation,
    setup_browser,
    start_game,
)


def collect_single_episode(
    output_dir, episode_index, poll_interval=0.25, max_steps=2000
):
    """Collect one human-played episode and save it to JSON.

    Records discrete actions (0 to NUM_ACTIONS-1) representing the
    direction the human is moving at each timestep.

    Args:
        output_dir: Directory where episode JSON will be saved.
        episode_index: Index for naming the episode file.
        poll_interval: Seconds between successive samples.
        max_steps: Safety cap on steps per episode.

    Returns:
        Path to the saved JSON file.
    """
    driver = setup_browser()
    start_game(driver)
    print("Bring the Chrome window to the front and play manually.")

    controller = SlitherController(driver, save_screenshots=False, record_video=False)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prev_length = controller.get_snake_length()
    steps = []
    step_idx = 0

    try:
        print(f"Recording episode {episode_index + 1}...")
        while True:
            state = controller.get_detailed_state()
            obs = extract_observation(state)

            snake_angle = None
            action = None
            if state and state.get("snake") is not None:
                snake_angle = state["snake"].get("angle")
                if snake_angle is not None:
                    # Convert angle to discrete action index
                    action = angle_rad_to_discrete_action(snake_angle)

            current_length = controller.get_snake_length()
            length_increase = current_length - prev_length

            # Match the reward shaping from environment.py
            reward = length_increase * 15.0 - 5.0

            done = controller.is_game_over()
            if done:
                reward -= 50.0
                reward += current_length * 0.5

            elapsed = time.time() - start_time

            step_data = {
                "t": step_idx,
                "elapsed_seconds": round(elapsed, 3),
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
                print("Game over detected; ending episode.")
                break

            if step_idx >= max_steps:
                print("Reached max_steps limit; ending episode.")
                break

            time.sleep(poll_interval)

    finally:
        driver.quit()

    duration = time.time() - start_time
    final_length = steps[-1]["snake_length"] if steps else 0

    episode_data = {
        "metadata": {
            "episode_index": episode_index,
            "timestamp": timestamp,
            "poll_interval": poll_interval,
            "max_steps": max_steps,
            "num_steps": len(steps),
            "final_length": final_length,
            "duration_seconds": round(duration, 3),
            "observation_dim": OBSERVATION_DIM,
            "num_actions": NUM_ACTIONS,
        },
        "steps": steps,
    }

    os.makedirs(output_dir, exist_ok=True)
    filename = f"human_episode_{episode_index + 1}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(episode_data, f, indent=2)

    print(f"Saved episode {episode_index + 1} to {filepath}")
    return filepath


def collect_human_trajectories(
    num_episodes=3, output_dir="trajectories", poll_interval=0.25, max_steps=2000
):
    """Collect multiple human-played episodes.

    Args:
        num_episodes: Number of episodes to collect.
        output_dir: Directory for saving episode files.
        poll_interval: Sampling interval in seconds.
        max_steps: Maximum steps per episode.

    Returns:
        List of saved file paths.
    """
    print(f"Collecting {num_episodes} human-played episode(s) to '{output_dir}'")

    saved_files = []
    for ep in range(num_episodes):
        filepath = collect_single_episode(
            output_dir=output_dir,
            episode_index=ep,
            poll_interval=poll_interval,
            max_steps=max_steps,
        )
        saved_files.append(filepath)

        if ep < num_episodes - 1:
            print("Preparing next episode...\n")

    print("All episodes collected.")
    return saved_files
