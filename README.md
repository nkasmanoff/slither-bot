# Slither Bot

An autonomous bot that plays [Slither.io](http://slither.io) using Selenium WebDriver.

## Features

-   **Food-seeking behavior**: Navigates towards the nearest food pellet
-   **Danger avoidance**: Detects nearby enemy snakes and flees when they get too close
-   **Game state logging**: Captures screenshots and detailed JSON logs for each frame
-   **State extraction**: Extracts game world data including food positions, other snakes, and player stats via JavaScript injection

## How It Works

The bot injects JavaScript to read game state variables (`window.snake`, `window.foods`, `window.slithers`) and controls the snake by setting mouse position and angle variables directly.

**Policy Logic:**

1. If an enemy snake is within 300 units → **FLEE** (move opposite direction)
2. If still fleeing and enemy within 500 units → **KEEP FLEEING**
3. Otherwise → **SEEK FOOD** (move towards nearest food pellet)

## Installation

```bash
pip install selenium
```

You'll also need [ChromeDriver](https://chromedriver.chromium.org/) installed and in your PATH.

## Usage

```bash
python slither.py
```

The bot will:

1. Open Chrome and navigate to slither.io
2. Click Play to start the game
3. Run the food-seeking + danger avoidance policy
4. Save screenshots to `games/<timestamp>/images/`
5. Save game state log to `games/<timestamp>/game_log.json`

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

-   Train an RL model using collected game states and other snake data
-   Improve danger detection (consider snake body segments, not just heads)
-   Add boosting strategy for chasing prey or escaping danger
