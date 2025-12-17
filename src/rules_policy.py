"""Rules-based policy for Slither.io game.

Implements a simple heuristic policy:
- Seeks nearest food when safe
- Flees from enemy snakes when they're too close
"""

import json
import math
import os
import time
from datetime import datetime

from selenium.webdriver.common.by import By

from .controller import SlitherController
from .utils import setup_browser, wait_for_game_ready

DANGER_DISTANCE = 300
SAFE_DISTANCE = 500
STALL_THRESHOLD = 50


def run_rules_based_policy(num_games=5, record_video=False):
    """Run the rules-based policy for a specified number of games.

    Args:
        num_games: Number of games to play.
        record_video: Whether to record gameplay videos.

    Returns:
        Dictionary with game metrics.
    """
    driver = setup_browser()

    print(f"\nPlaying {num_games} games with rules-based policy...")

    game_steps, game_snake_lengths, game_rewards = [], [], []
    best_length, best_reward = 0, float("-inf")

    for game in range(num_games):
        play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
        for button in play_buttons:
            if "Play" in button.text:
                button.click()
                break
        print(f"Game {game + 1}/{num_games} started")
        time.sleep(1)

        controller = SlitherController(
            driver, save_screenshots=False, record_video=record_video
        )
        wait_for_game_ready(controller)

        fleeing = False
        steps = 0
        max_snake_length = 0
        total_reward = 0.0
        previous_length = controller.get_snake_length()
        need_browser_restart = False
        last_length = previous_length
        no_change_count = 0

        while True:
            if controller.is_game_over():
                print(f"Game {game + 1} over!")
                break

            state = controller.get_detailed_state()
            if not state:
                controller.maintain_direction(0.5)
                steps += 1
                continue

            current_length = controller.get_snake_length()
            max_snake_length = max(max_snake_length, current_length)
            steps += 1

            length_increase = current_length - previous_length
            step_reward = length_increase * 15.0 - 5
            total_reward += step_reward
            previous_length = current_length

            if current_length == last_length:
                no_change_count += 1
            else:
                no_change_count = 0
                last_length = current_length

            if no_change_count >= STALL_THRESHOLD:
                print(f"Game {game + 1} stalled. Ending...")
                need_browser_restart = True
                break

            other_snakes = state.get("other_snakes", [])
            nearest_enemy = other_snakes[0] if other_snakes else None
            foods = state.get("foods", [])

            if nearest_enemy and nearest_enemy["distance"] < DANGER_DISTANCE:
                fleeing = True
                angle_to_enemy_rad = nearest_enemy["angle_to"]
                angle_away_deg = math.degrees(angle_to_enemy_rad) + 180
                target_angle = -angle_away_deg % 360
                controller.move_to_angle(target_angle)
                time.sleep(0.05)

            elif (
                fleeing and nearest_enemy and nearest_enemy["distance"] < SAFE_DISTANCE
            ):
                angle_to_enemy_rad = nearest_enemy["angle_to"]
                angle_away_deg = math.degrees(angle_to_enemy_rad) + 180
                target_angle = -angle_away_deg % 360
                controller.move_to_angle(target_angle)
                time.sleep(0.05)

            else:
                fleeing = False
                if foods:
                    nearest_food = foods[0]
                    angle_to_food_rad = nearest_food["angle"]
                    angle_to_food_deg = math.degrees(angle_to_food_rad)
                    target_angle = -angle_to_food_deg % 360
                    controller.move_to_angle(target_angle)
                    time.sleep(0.1)
                else:
                    controller.maintain_direction(0.5)

            if steps > 1000:
                print(f"Game {game + 1} reached max steps")
                need_browser_restart = True
                break

        game_steps.append(steps)
        game_snake_lengths.append(max_snake_length)
        game_rewards.append(total_reward)
        best_length = max(best_length, max_snake_length)
        best_reward = max(best_reward, total_reward)

        print(
            f"Game {game + 1}/{num_games} | Steps: {steps} | Length: {max_snake_length} | Reward: {total_reward:.2f}"
        )

        if controller.record_video and controller.video_frames:
            video_path = controller._create_video()
            if video_path:
                print(f"Game {game + 1} video saved to: {video_path}")

        controller.save_game_log()

        if need_browser_restart:
            print("Restarting browser for next game...")
            driver.quit()
            driver = setup_browser()
        else:
            time.sleep(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_data = {
        "policy_type": "rules_based",
        "num_games": num_games,
        "game_steps": game_steps,
        "game_snake_lengths": game_snake_lengths,
        "game_rewards": game_rewards,
        "best_length": best_length,
        "best_reward": best_reward,
        "timestamp": timestamp,
    }

    metrics_dir = "inference_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(
        metrics_dir, f"inference_metrics_rules_{timestamp}.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nInference metrics saved to: {metrics_path}")
    print(f"Best Length: {best_length} | Best Reward: {best_reward:.2f}")

    driver.quit()
    return metrics_data
