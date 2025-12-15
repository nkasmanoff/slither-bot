"""Human trajectory collector for Slither.io.

This script mirrors the state representation used in `slither_rl.SlitherEnv`,
but instead of controlling the snake with an RL policy it simply *observes*
while a human plays in the Selenium-controlled Chrome window.

Each episode (one life) is saved as a JSON file containing a sequence of
(state, action, reward) tuples, where:

- state: the 15-dimensional observation vector used by `SlitherEnv`:
  [current_angle, snake_length,
   nearest_food_distance, nearest_food_angle,
   nearest_prey_distance, nearest_prey_angle,
   nearest_enemy_distance, nearest_enemy_angle,
   num_nearby_foods, num_nearby_preys, num_nearby_enemies,
   food_quadrant_1_count, food_quadrant_2_count,
   food_quadrant_3_count, food_quadrant_4_count]
- action: a discrete action index in {0..7}, obtained by discretizing the
  snake's current angle into 8 bins of 45 degrees (same action space as
  `SlitherEnv` / `REINFORCEAgent`)
- reward: same reward function as in `SlitherEnv.step` so that these
  trajectories can be used directly for warm-starting RL if desired.

Usage (from repo root):

    python slither_human_collect.py --episodes 3 --output-dir human_trajectories

When Chrome opens, bring its window to the foreground and play normally using
your mouse. The script will record until you die (game over), then save an
episode JSON and start the next game.
"""

import os
import json
import time
from datetime import datetime

import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By

from slither import SlitherController


# Observation and action dimensions must match `SlitherEnv` in slither_rl.py
OBSERVATION_DIM = 15
ACTION_DIM = 8  # 8 discrete directions: 0, 45, 90, ..., 315 degrees


def extract_observation_from_state(state):
    """Replicate `SlitherEnv._get_obs` from slither_rl.py.

    Args:
        state (dict | None): Detailed game state from `SlitherController.get_detailed_state`.

        Returns:
        np.ndarray: 15-dimensional observation vector (dtype float32).
    """
    if not state or not state.get("snake"):
        # Return zero observation if game not ready
        return np.zeros(OBSERVATION_DIM, dtype=np.float32)

    snake = state["snake"]
    foods = state.get("foods", [])
    preys = state.get("preys", [])
    enemies = state.get("other_snakes", [])

    # Current angle (normalized to [-1, 1])
    current_angle = snake.get("angle", 0) / np.pi

    # Snake length (log-normalized for better scale)
    snake_length = np.log(max(snake.get("length", 1), 1)) / 10.0

    # Snake world position (used for quadrant-based food counts)
    snake_x = snake.get("x", 0.0)
    snake_y = snake.get("y", 0.0)

    # Nearest food
    if foods:
        nearest_food = foods[0]
        nearest_food_dist = np.tanh(nearest_food["distance"] / 500.0)  # Normalize
        nearest_food_angle = nearest_food["angle"] / np.pi
    else:
        nearest_food_dist = 1.0
        nearest_food_angle = 0.0

    # Nearest prey (high-value food from dead snakes)
    if preys:
        nearest_prey = preys[0]
        nearest_prey_dist = np.tanh(nearest_prey["distance"] / 500.0)  # Normalize
        nearest_prey_angle = nearest_prey["angle"] / np.pi
    else:
        nearest_prey_dist = 1.0
        nearest_prey_angle = 0.0

    # Nearest enemy
    if enemies:
        nearest_enemy = enemies[0]
        nearest_enemy_dist = np.tanh(nearest_enemy["distance"] / 500.0)
        nearest_enemy_angle = nearest_enemy["angle_to"] / np.pi
    else:
        nearest_enemy_dist = 1.0
        nearest_enemy_angle = 0.0

    # Counts (normalized)
    num_foods = min(len(foods), 50) / 50.0
    num_preys = min(len(preys), 20) / 20.0
    num_enemies = min(len(enemies), 15) / 15.0

    # Food distribution across 4 quadrants around the snake.
    # Quadrants are defined in the (dx, dy) plane centered at the snake:
    #   Q1: dx >= 0, dy >= 0
    #   Q2: dx <  0, dy >= 0
    #   Q3: dx <  0, dy <  0
    #   Q4: dx >= 0, dy <  0
    # Each count is normalized by the same 50-food cap used for num_foods.
    quadrant_counts = [0, 0, 0, 0]
    for f in foods:
        fx = f.get("x")
        fy = f.get("y")
        if fx is None or fy is None:
            continue
        dx = fx - snake_x
        dy = fy - snake_y
        if dx >= 0 and dy >= 0:
            quadrant_counts[0] += 1
        elif dx < 0 and dy >= 0:
            quadrant_counts[1] += 1
        elif dx < 0 and dy < 0:
            quadrant_counts[2] += 1
        elif dx >= 0 and dy < 0:
            quadrant_counts[3] += 1

    max_foods = 50.0
    food_quadrants = [min(count, max_foods) / max_foods for count in quadrant_counts]

    obs = np.array(
        [
            current_angle,
            snake_length,
            nearest_food_dist,
            nearest_food_angle,
            nearest_prey_dist,
            nearest_prey_angle,
            nearest_enemy_dist,
            nearest_enemy_angle,
            num_foods,
            num_preys,
            num_enemies,
            food_quadrants[0],
            food_quadrants[1],
            food_quadrants[2],
            food_quadrants[3],
        ],
        dtype=np.float32,
    )

    return obs


def angle_to_action_index(angle_rad):
    """Discretize a continuous snake angle (radians) into 8 bins.

    This mirrors the 8-way action space used by `SlitherEnv` / `REINFORCEAgent`.

    Args:
        angle_rad (float): Snake heading angle in radians (as reported by game).

    Returns:
        int: Discrete action index in {0..7}.
    """
    if angle_rad is None:
        return None

    # Normalize to [0, 2*pi)
    angle_rad = float(angle_rad) % (2.0 * np.pi)
    # Each bin is 45 degrees == pi/4 radians
    bin_size = (2.0 * np.pi) / ACTION_DIM
    action = int(np.round(angle_rad / bin_size)) % ACTION_DIM
    return int(action)


def setup_browser_and_start_game():
    """Launch Chrome, load slither.io, and start a new game.

    Returns:
        driver (webdriver.Chrome)
        controller (SlitherController)
    """
    driver = webdriver.Chrome()
    driver.get("http://slither.io")

    print("Waiting for game to load...")
    time.sleep(5)

    # Set game to low quality for better performance
    try:
        driver.find_element(By.ID, "grqi").click()
        print("Set game to low quality")
    except Exception:
        pass  # Quality button may not be available

    # Click the Play button to start the game
    play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_buttons:
        if "Play" in button.text:
            button.click()
            break

    print("Game started! Bring the Chrome window to the front and play manually.")
    time.sleep(2)

    controller = SlitherController(driver, save_screenshots=False, record_video=False)
    return driver, controller


def collect_single_episode(
    output_dir, episode_index, poll_interval=0.25, max_steps=2000
):
    """Collect one human-played episode and save it to JSON.

    Args:
        output_dir (str): Directory where episode JSON will be saved.
        episode_index (int): Index for naming the episode file.
        poll_interval (float): Seconds between successive samples.
        max_steps (int): Safety cap on steps per episode.

    Returns:
        str: Path to the saved JSON file.
    """
    driver, controller = setup_browser_and_start_game()

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prev_length = controller.get_snake_length()
    steps = []
    step_idx = 0

    try:
        print(f"Recording episode {episode_index + 1}...")
        while True:
            state = controller.get_detailed_state()
            obs = extract_observation_from_state(state)

            snake_angle = None
            action_idx = None
            if state and state.get("snake") is not None:
                snake_angle = state["snake"].get("angle")
                if snake_angle is not None:
                    action_idx = angle_to_action_index(snake_angle)

            current_length = controller.get_snake_length()
            length_increase = current_length - prev_length

            # Same reward structure as SlitherEnv.step
            reward = 0.0
            reward += length_increase * 10.0  # Large reward for eating food
            reward -= 2.5  # Small penalty for duration

            done = controller.is_game_over()
            if done:
                reward -= 50.0
                reward += current_length * 0.5  # Bonus for final length

            elapsed = time.time() - start_time

            step_data = {
                "t": step_idx,
                "elapsed_seconds": round(elapsed, 3),
                "observation": obs.tolist(),
                "action_index": action_idx,
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
            "action_dim": ACTION_DIM,
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
    num_episodes=3,
    output_dir="human_trajectories",
    poll_interval=0.25,
    max_steps=2000,
):
    """Collect multiple human-played episodes.

    This is a simple loop around `collect_single_episode`.
    """
    print(
        f"Collecting {num_episodes} human-played episode(s). "
        f"Data will be saved under '{output_dir}'."
    )

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Collect human-played Slither.io trajectories with the same "
            "state representation and reward function used in slither_rl."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of human-played episodes (lives) to record.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="human_trajectories",
        help="Directory where episode JSON files will be saved.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help=(
            "Sampling interval in seconds. Should roughly match the "
            "step delay used in slither_rl (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Safety cap on number of samples per episode.",
    )

    args = parser.parse_args()

    collect_human_trajectories(
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval,
        max_steps=args.max_steps,
    )
