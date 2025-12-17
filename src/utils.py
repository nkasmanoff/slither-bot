"""Common utilities for the Slither.io bot.

Contains shared helper functions for browser setup, game interaction,
and observation extraction used across multiple modules.
"""

import os
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Environment configuration
IS_RASPBERRY_PI = os.environ.get("SLITHER_RASPBERRY_PI", "false").lower() == "true"

# Observation dimensions (17 features with relative action space)
OBSERVATION_DIM = 17
ACTION_DIM = 1

# Maximum turn rate per step (in radians) - about 45 degrees
MAX_TURN_RATE = np.pi / 4


def setup_browser(headless: bool = False) -> webdriver.Chrome:
    """Set up and return a Chrome WebDriver instance.

    Args:
        headless: Whether to run Chrome in headless mode.

    Returns:
        Configured Chrome WebDriver instance.
    """
    if IS_RASPBERRY_PI:
        options = Options()
        options.binary_location = "/usr/bin/chromium-browser"
        options.add_argument("--kiosk")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
    else:
        options = Options()
        if headless:
            options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)

    driver.get("http://slither.io")
    print("Waiting for game to load...")
    time.sleep(5)

    # Set game to low quality for better performance
    try:
        driver.find_element(By.ID, "grqi").click()
        print("Set game to low quality")
    except Exception:
        pass

    return driver


def start_game(driver: webdriver.Chrome) -> None:
    """Click the Play button to start a new game.

    Args:
        driver: Chrome WebDriver instance.
    """
    play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_buttons:
        if "Play" in button.text:
            button.click()
            break
    print("Game started!")
    time.sleep(2)


def wait_for_game_ready(controller, max_wait: float = 10.0) -> int:
    """Wait for the game to be ready (snake length reset).

    Args:
        controller: SlitherController instance.
        max_wait: Maximum time to wait in seconds.

    Returns:
        Current snake length.
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        length = controller.get_snake_length()
        if 0 < length < 50:
            break
        time.sleep(0.5)
    return controller.get_snake_length()


def extract_observation(state: dict | None) -> np.ndarray:
    """Extract observation vector from game state.

    This uses a relative action space where all angles are relative to the
    snake's current heading. This makes learning easier since the agent
    just needs to learn "turn toward food, away from enemies".

    Args:
        state: Detailed game state from SlitherController.get_detailed_state().

    Returns:
        17-dimensional observation vector (dtype float32).

    Observation features:
        0: snake_length - Log-normalized snake length
        1: nearest_food_dist - Distance to nearest food (tanh-normalized)
        2: relative_food_angle - Angle to food relative to heading [-1, 1]
        3: nearest_prey_dist - Distance to nearest prey
        4: relative_prey_angle - Angle to prey relative to heading [-1, 1]
        5: nearest_enemy_dist - Distance to nearest enemy
        6: relative_enemy_angle - Angle to enemy relative to heading [-1, 1]
        7: num_foods - Normalized food count
        8: num_preys - Normalized prey count
        9: num_enemies - Normalized enemy count
        10-13: food_quadrants - Food density in 4 quadrants (relative to heading)
        14: food_efficiency - Weighted inverse-distance to nearby food
        15: enemy_threat - Weighted inverse-distance to nearby enemies
        16: nearest_enemy_head_dist - Distance to nearest enemy HEAD (collision danger)
    """
    if not state or not state.get("snake"):
        return np.zeros(OBSERVATION_DIM, dtype=np.float32)

    snake = state["snake"]
    foods = state.get("foods", [])
    preys = state.get("preys", [])
    enemies = state.get("other_snakes", [])

    # Current angle in normalized form (used for relative calculations)
    current_angle = snake.get("angle", 0) / np.pi

    # Snake length (log-normalized)
    snake_length = np.log(max(snake.get("length", 1), 1)) / 10.0

    # Snake world position
    snake_x = snake.get("x", 0.0)
    snake_y = snake.get("y", 0.0)

    # Helper function to compute relative angle
    def compute_relative_angle(absolute_angle_normalized):
        """Convert absolute angle to relative angle (both in [-1, 1] range)."""
        rel = absolute_angle_normalized - current_angle
        # Wrap to [-1, 1] (equivalent to [-pi, pi])
        while rel > 1.0:
            rel -= 2.0
        while rel < -1.0:
            rel += 2.0
        return rel

    # Nearest food
    if foods:
        nearest_food = foods[0]
        nearest_food_dist = np.tanh(nearest_food["distance"] / 500.0)
        nearest_food_angle = nearest_food["angle"] / np.pi
        relative_food_angle = compute_relative_angle(nearest_food_angle)
    else:
        nearest_food_dist = 1.0
        relative_food_angle = 0.0

    # Nearest prey (high-value food from dead snakes)
    if preys:
        nearest_prey = preys[0]
        nearest_prey_dist = np.tanh(nearest_prey["distance"] / 500.0)
        nearest_prey_angle = nearest_prey["angle"] / np.pi
        relative_prey_angle = compute_relative_angle(nearest_prey_angle)
    else:
        nearest_prey_dist = 1.0
        relative_prey_angle = 0.0

    # Nearest enemy (body or head)
    if enemies:
        nearest_enemy = enemies[0]
        nearest_enemy_dist = np.tanh(nearest_enemy["distance"] / 500.0)
        nearest_enemy_angle = nearest_enemy["angle_to"] / np.pi
        relative_enemy_angle = compute_relative_angle(nearest_enemy_angle)
        # Head distance is important for collision avoidance
        nearest_enemy_head_dist = np.tanh(
            nearest_enemy.get("head_distance", 1000) / 500.0
        )
    else:
        nearest_enemy_dist = 1.0
        relative_enemy_angle = 0.0
        nearest_enemy_head_dist = 1.0

    # Counts (normalized)
    num_foods = min(len(foods), 50) / 50.0
    num_preys = min(len(preys), 20) / 20.0
    num_enemies = min(len(enemies), 15) / 15.0

    # Food distribution across 4 quadrants (relative to snake heading)
    # Quadrants are: front-right, front-left, back-left, back-right
    snake_angle_rad = snake.get("angle", 0)
    quadrant_counts = [0, 0, 0, 0]
    for f in foods:
        fx = f.get("x")
        fy = f.get("y")
        if fx is None or fy is None:
            continue
        dx = fx - snake_x
        dy = fy - snake_y
        # Rotate to snake's reference frame
        cos_a = np.cos(-snake_angle_rad)
        sin_a = np.sin(-snake_angle_rad)
        rel_x = dx * cos_a - dy * sin_a  # Forward/backward
        rel_y = dx * sin_a + dy * cos_a  # Left/right
        if rel_x >= 0 and rel_y >= 0:
            quadrant_counts[0] += 1  # Front-right
        elif rel_x >= 0 and rel_y < 0:
            quadrant_counts[1] += 1  # Front-left
        elif rel_x < 0 and rel_y < 0:
            quadrant_counts[2] += 1  # Back-left
        elif rel_x < 0 and rel_y >= 0:
            quadrant_counts[3] += 1  # Back-right

    max_foods = 50.0
    food_quadrants = [min(count, max_foods) / max_foods for count in quadrant_counts]

    # Food efficiency
    food_efficiency = 0.0
    for f in foods[:10]:
        food_efficiency += 1.0 / (f["distance"] + 50.0)
    food_efficiency = np.tanh(food_efficiency)

    # Enemy threat
    enemy_threat = 0.0
    for e in enemies[:5]:
        dist_factor = 1.0 / (e["distance"] + 50.0)
        enemy_threat += dist_factor
    enemy_threat = np.tanh(enemy_threat)

    return np.array(
        [
            snake_length,
            nearest_food_dist,
            relative_food_angle,
            nearest_prey_dist,
            relative_prey_angle,
            nearest_enemy_dist,
            relative_enemy_angle,
            num_foods,
            num_preys,
            num_enemies,
            food_quadrants[0],
            food_quadrants[1],
            food_quadrants[2],
            food_quadrants[3],
            food_efficiency,
            enemy_threat,
            nearest_enemy_head_dist,
        ],
        dtype=np.float32,
    )


def angle_to_continuous_action(angle_rad: float) -> float:
    """Convert snake angle (radians) to continuous action in [-1, 1].

    Args:
        angle_rad: Snake heading angle in radians.

    Returns:
        Continuous action in [-1, 1].
    """
    if angle_rad is None:
        return 0.0

    angle_rad = float(angle_rad) % (2.0 * np.pi)
    return float(angle_rad / np.pi - 1.0)


def continuous_action_to_turn(action: float) -> float:
    """Convert continuous action [-1, 1] to turn delta in radians.

    The action represents a relative turn:
    - action = -1: turn right (clockwise) by MAX_TURN_RATE
    - action = 0: go straight
    - action = +1: turn left (counter-clockwise) by MAX_TURN_RATE

    Args:
        action: Continuous value in [-1, 1].

    Returns:
        Turn delta in radians [-MAX_TURN_RATE, +MAX_TURN_RATE].
    """
    return float(action) * MAX_TURN_RATE


def apply_turn_to_angle(current_angle_rad: float, turn_delta: float) -> float:
    """Apply a turn delta to get new absolute angle in degrees.

    Args:
        current_angle_rad: Current heading in radians.
        turn_delta: Turn amount in radians (positive = CCW, negative = CW).

    Returns:
        New angle in degrees [0, 360].
    """
    new_angle_rad = current_angle_rad + turn_delta
    new_angle_deg = np.degrees(new_angle_rad) % 360
    return float(new_angle_deg)


def continuous_action_to_angle(action: float) -> float:
    """Convert continuous action [-1, 1] to angle in degrees [0, 360].

    DEPRECATED: Use continuous_action_to_turn() for relative actions.
    Kept for backward compatibility with trajectory data.

    Args:
        action: Continuous value in [-1, 1].

    Returns:
        Angle in degrees [0, 360].
    """
    return (action + 1.0) * 180.0
