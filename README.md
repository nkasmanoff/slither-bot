# Slither Bot

An autonomous bot that plays [Slither.io](http://slither.io) using Selenium WebDriver, with both rule-based and reinforcement learning approaches.

## Features

-   **Rule-based policy** (`slither.py`): Hand-crafted food-seeking and danger avoidance behavior
-   **RL policy** (`slither_rl.py`): Policy gradient (REINFORCE) agent that learns through online training
-   **Game state logging**: Captures screenshots and detailed JSON logs for each frame
-   **State extraction**: Extracts game world data including food positions, other snakes, and player stats via JavaScript injection

## How It Works

The bot injects JavaScript to read game state variables (`window.snake`, `window.foods`, `window.slithers`) and controls the snake by setting mouse position and angle variables directly.

### Rule-Based Policy (`slither.py`)

**Policy Logic:**

1. If an enemy snake is within 300 units → **FLEE** (move opposite direction)
2. If still fleeing and enemy within 500 units → **KEEP FLEEING**
3. Otherwise → **SEEK FOOD** (move towards nearest food pellet)

### Reinforcement Learning Policy (`slither_rl.py`)

**REINFORCE Algorithm:**

-   **State**: 8-dimensional observation (angle, length, nearest food/enemy distance & angle, counts)
-   **Actions**: 8 discrete directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
-   **Rewards**:
    -   +10 per unit of length increase (food collection)
    -   +0.1 per survival step
    -   -10 for dying + 0.5 × final length
-   **Training**: Online policy gradient using REINFORCE with baseline normalization
-   **Network**: Simple 3-layer MLP with ReLU activations

## Installation

```bash
pip install -r requirements.txt
```

You'll also need [ChromeDriver](https://chromedriver.chromium.org/) installed and in your PATH.

## Usage

### Rule-Based Bot

```bash
python slither.py
```

The bot will:

1. Open Chrome and navigate to slither.io
2. Click Play to start the game
3. Run the food-seeking + danger avoidance policy
4. Save screenshots to `games/<timestamp>/images/`
5. Save game state log to `games/<timestamp>/game_log.json`

### RL Training

```bash
python slither_rl.py
```

This will:

1. Initialize the REINFORCE agent with a neural network policy
2. Train for 50 episodes (configurable)
3. Each episode: agent plays until death, then updates policy based on rewards
4. Print training progress (episode reward, steps, max length, loss)

The agent learns online - it improves its policy after each game by maximizing expected return.

## Output Structure

```
games/
└── 2025-12-13_17-42-55/
    ├── game_log.json       # Full game state for each frame
    └── images/
        ├── frame_000000_angle_45.0_speed_normal.png
        ├── frame_000001_angle_90.0_speed_normal.png
        └── ...
```

## Game Log Format

Each frame in `game_log.json` contains:

-   `state`: Current angle, boosting status, snake length/rank
-   `action`: What action was taken
-   `game_world`: Detailed world state (snake position, nearby foods, other snakes)
-   `previous_actions`: Last 5 actions for context

## Visualization

Use `visualize.ipynb` to overlay extracted game state on screenshots for validation.

## Future Work

-   ✅ ~~Train an RL model using collected game states~~ (Implemented: REINFORCE policy gradient)
-   Add more sophisticated RL algorithms (PPO, A2C, DQN)
-   Improve danger detection (consider snake body segments, not just heads)
-   Add boosting strategy for chasing prey or escaping danger
-   Implement model saving/loading for trained policies
-   Add curriculum learning (start with easier scenarios)
