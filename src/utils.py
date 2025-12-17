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

# Observation dimensions (21 features for discrete action space)
OBSERVATION_DIM = 21

# Discrete action space: 12 directions at 30° intervals
NUM_ACTIONS = 12
ANGLE_PER_ACTION = 360.0 / NUM_ACTIONS  # 30 degrees


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

    Designed for discrete action space (12 absolute directions).
    Uses ABSOLUTE angles so the agent can directly match angles to actions.

    Args:
        state: Detailed game state from SlitherController.get_detailed_state().

    Returns:
        21-dimensional observation vector (dtype float32).

    Observation features:
        0: current_angle - Snake's heading as discrete action index / NUM_ACTIONS
        1: snake_length - Log-normalized snake length
        2: nearest_food_dist - Distance to nearest food (tanh-normalized)
        3: food_action - Which discrete action points toward food (normalized)
        4: nearest_prey_dist - Distance to nearest prey
        5: prey_action - Which discrete action points toward prey (normalized)
        6: nearest_enemy_dist - Distance to nearest enemy body
        7: enemy_action - Which discrete action points toward enemy (normalized)
        8: nearest_enemy_head_dist - Distance to enemy head
        9: num_foods - Normalized food count
        10: num_preys - Normalized prey count
        11: num_enemies - Normalized enemy count
        12: food_efficiency - Weighted inverse-distance to nearby food
        13: enemy_threat - Overall threat level from nearby enemies
        14-17: danger_quadrant - Danger level in front/right/back/left
        18-20: closest_enemy_in_front/right/left - Binary danger indicators
    """
    if not state or not state.get("snake"):
        return np.zeros(OBSERVATION_DIM, dtype=np.float32)

    snake = state["snake"]
    foods = state.get("foods", [])
    preys = state.get("preys", [])
    enemies = state.get("other_snakes", [])

    # Current angle as discrete action index (normalized to [0, 1])
    snake_angle_rad = snake.get("angle", 0)
    snake_angle_deg = np.degrees(snake_angle_rad) % 360.0
    current_action = int(round(snake_angle_deg / ANGLE_PER_ACTION)) % NUM_ACTIONS
    current_angle_norm = current_action / NUM_ACTIONS

    # Snake length (log-normalized)
    snake_length = np.log(max(snake.get("length", 1), 1)) / 10.0

    def angle_to_action_normalized(angle_rad):
        """Convert angle in radians to normalized action index [0, 1]."""
        angle_deg = np.degrees(angle_rad) % 360.0
        action = int(round(angle_deg / ANGLE_PER_ACTION)) % NUM_ACTIONS
        return action / NUM_ACTIONS

    # Nearest food
    if foods:
        nearest_food = foods[0]
        nearest_food_dist = np.tanh(nearest_food["distance"] / 500.0)
        food_action = angle_to_action_normalized(nearest_food["angle"])
    else:
        nearest_food_dist = 1.0
        food_action = 0.0

    # Nearest prey (high-value food from dead snakes)
    if preys:
        nearest_prey = preys[0]
        nearest_prey_dist = np.tanh(nearest_prey["distance"] / 500.0)
        prey_action = angle_to_action_normalized(nearest_prey["angle"])
    else:
        nearest_prey_dist = 1.0
        prey_action = 0.0

    # Nearest enemy (body)
    if enemies:
        nearest_enemy = enemies[0]
        nearest_enemy_dist = np.tanh(nearest_enemy["distance"] / 500.0)
        enemy_action = angle_to_action_normalized(nearest_enemy["angle_to"])
        nearest_enemy_head_dist = np.tanh(
            nearest_enemy.get("head_distance", 1000) / 500.0
        )
    else:
        nearest_enemy_dist = 1.0
        enemy_action = 0.0
        nearest_enemy_head_dist = 1.0

    # Counts (normalized)
    num_foods = min(len(foods), 50) / 50.0
    num_preys = min(len(preys), 20) / 20.0
    num_enemies = min(len(enemies), 15) / 15.0

    # Food efficiency
    food_efficiency = 0.0
    for f in foods[:10]:
        food_efficiency += 1.0 / (f["distance"] + 50.0)
    food_efficiency = np.tanh(food_efficiency)

    # Enemy threat (overall)
    enemy_threat = 0.0
    for e in enemies[:5]:
        dist_factor = 1.0 / (e["distance"] + 50.0)
        enemy_threat += dist_factor
    enemy_threat = np.tanh(enemy_threat)

    # Directional danger: compute danger level in 4 quadrants relative to heading
    # Front (±45°), Right (45°-135°), Back (135°-225°), Left (225°-315°)
    danger_quadrants = [0.0, 0.0, 0.0, 0.0]  # front, right, back, left
    DANGER_RADIUS = 300  # Consider enemies within this distance

    for e in enemies:
        dist = e.get("distance", 1000)
        if dist > DANGER_RADIUS:
            continue

        # Compute relative angle to enemy
        enemy_angle_rad = e.get("angle_to", 0)
        relative_angle = (enemy_angle_rad - snake_angle_rad) % (2 * np.pi)
        relative_angle_deg = np.degrees(relative_angle)

        # Danger contribution (closer = more dangerous)
        danger = 1.0 - (dist / DANGER_RADIUS)

        # Assign to quadrant
        if relative_angle_deg < 45 or relative_angle_deg >= 315:
            danger_quadrants[0] += danger  # Front
        elif 45 <= relative_angle_deg < 135:
            danger_quadrants[1] += danger  # Right
        elif 135 <= relative_angle_deg < 225:
            danger_quadrants[2] += danger  # Back
        else:
            danger_quadrants[3] += danger  # Left

    # Normalize danger quadrants
    danger_quadrants = [min(d, 1.0) for d in danger_quadrants]

    # Binary danger indicators for immediate collision avoidance
    IMMEDIATE_DANGER_DIST = 150
    danger_front = 0.0
    danger_right = 0.0
    danger_left = 0.0

    for e in enemies:
        dist = e.get("distance", 1000)
        if dist > IMMEDIATE_DANGER_DIST:
            continue

        enemy_angle_rad = e.get("angle_to", 0)
        relative_angle = (enemy_angle_rad - snake_angle_rad) % (2 * np.pi)
        relative_angle_deg = np.degrees(relative_angle)

        if relative_angle_deg < 60 or relative_angle_deg >= 300:
            danger_front = 1.0
        elif 60 <= relative_angle_deg < 150:
            danger_right = 1.0
        elif 210 <= relative_angle_deg < 300:
            danger_left = 1.0

    return np.array(
        [
            current_angle_norm,
            snake_length,
            nearest_food_dist,
            food_action,
            nearest_prey_dist,
            prey_action,
            nearest_enemy_dist,
            enemy_action,
            nearest_enemy_head_dist,
            num_foods,
            num_preys,
            num_enemies,
            food_efficiency,
            enemy_threat,
            danger_quadrants[0],  # front
            danger_quadrants[1],  # right
            danger_quadrants[2],  # back
            danger_quadrants[3],  # left
            danger_front,
            danger_right,
            danger_left,
        ],
        dtype=np.float32,
    )


def discrete_action_to_angle(action: int) -> float:
    """Convert discrete action index to angle in degrees.

    Args:
        action: Discrete action index (0 to NUM_ACTIONS-1).

    Returns:
        Angle in degrees [0, 360).
    """
    return float(action * ANGLE_PER_ACTION)


def angle_to_discrete_action(angle_deg: float) -> int:
    """Convert angle in degrees to nearest discrete action index.

    Args:
        angle_deg: Angle in degrees.

    Returns:
        Discrete action index (0 to NUM_ACTIONS-1).
    """
    # Normalize to [0, 360)
    angle_deg = float(angle_deg) % 360.0
    # Round to nearest action
    action = int(round(angle_deg / ANGLE_PER_ACTION)) % NUM_ACTIONS
    return action


def angle_rad_to_discrete_action(angle_rad: float) -> int:
    """Convert angle in radians to nearest discrete action index.

    Args:
        angle_rad: Angle in radians.

    Returns:
        Discrete action index (0 to NUM_ACTIONS-1).
    """
    if angle_rad is None:
        return 0
    angle_deg = np.degrees(angle_rad) % 360.0
    return angle_to_discrete_action(angle_deg)
