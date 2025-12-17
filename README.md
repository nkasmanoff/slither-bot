# Slither Bot

An autonomous bot that plays [Slither.io](http://slither.io) using Selenium WebDriver, with both rule-based and reinforcement learning approaches.

## Features

-   **Rules-based policy**: Hand-crafted food-seeking and danger avoidance behavior
-   **RL agents**: REINFORCE and A2C (Actor-Critic) agents that learn through online training
-   **Imitation learning**: Collect human or auto-generated trajectories for pre-training
-   **Video recording**: Record gameplay videos with model state annotations
-   **State extraction**: Extracts game world data via JavaScript injection

## Project Structure

```
slither-bot/
├── src/                    # Core library modules
│   ├── __init__.py         # Package exports
│   ├── agents.py           # RL agents (REINFORCE, A2C)
│   ├── controller.py       # Selenium game controller
│   ├── environment.py      # Gym environment wrapper
│   ├── rules_policy.py     # Heuristic policy
│   ├── collect.py          # Human trajectory collection
│   ├── pretrain.py         # Supervised pre-training
│   ├── compare.py          # Agent comparison utilities
│   ├── profile.py          # Performance profiling
│   └── utils.py            # Shared helper functions
├── train.py                # Train RL agents
├── play.py                 # Play with trained models
├── collect.py              # Collect trajectories
├── pretrain.py             # Pre-train on trajectories
├── compare.py              # Compare agent performance
├── models/                 # Saved model checkpoints
├── trajectories/           # Collected trajectory data
├── training_logs/          # Training metrics
├── inference_logs/         # Inference metrics
└── games/                  # Recorded gameplay videos
```

## Installation

```bash
pip install -r requirements.txt
```

You'll also need [ChromeDriver](https://chromedriver.chromium.org/) installed and in your PATH.

## Quick Start

### 1. Train an RL Agent

```bash
# Train with A2C (recommended)
python train.py --algorithm a2c --episodes 50

# Train with REINFORCE
python train.py --algorithm reinforce --episodes 50
```

### 2. Play with a Trained Model

```bash
# Play with best A2C model
python play.py --agent a2c --model models/best_model_a2c.pt --games 5

# Play with rules-based policy
python play.py --agent rules --games 5
```

## Usage Examples

### Training

```bash
# Basic A2C training
python train.py --algorithm a2c --episodes 100

# A2C with custom N-step updates
python train.py --algorithm a2c --episodes 100 --n-steps 32

# Train from pre-trained checkpoint
python train.py --algorithm a2c --pretrained models/pretrained.pt

# Record training videos
python train.py --algorithm a2c --episodes 10 --record-video

# Faster/slower decision making
python train.py --action-delay 0.1   # Faster (10 Hz)
python train.py --action-delay 0.25  # Slower, more stable (4 Hz)
```

### Playing / Inference

```bash
# Play with A2C model
python play.py --agent a2c --model models/best_model_a2c.pt

# Play with REINFORCE model
python play.py --agent reinforce --model models/best_model.pt

# Play with rules-based policy
python play.py --agent rules

# Play without model (random baseline)
python play.py --agent a2c

# Record gameplay
python play.py --agent a2c --model models/best_model_a2c.pt --record-video
```

### Collecting Trajectories

```bash
# Collect human-played trajectories (you play manually)
python collect.py --mode human --episodes 5

# Collect auto trajectories using rules-based policy
python collect.py --mode auto --episodes 10

# Custom output directory
python collect.py --mode auto --episodes 5 --output-dir my_trajectories

# Adjust sampling rate
python collect.py --mode auto --poll-interval 0.1
```

### Pre-training

```bash
# Pre-train on existing trajectories
python pretrain.py --data-dir trajectories --epochs 50

# Collect auto trajectories then pre-train
python pretrain.py --auto --auto-episodes 10 --epochs 50

# Custom learning rate and batch size
python pretrain.py --data-dir trajectories --epochs 100 --lr 0.0001 --batch-size 128
```

### Comparing Agents

```bash
# Run comparison between all agent types
python compare.py --games 3

# More games for better statistics
python compare.py --games 10
```

## How It Works

### Game Controller

The bot injects JavaScript to read game state variables (`window.snake`, `window.foods`, `window.slithers`) and controls the snake by setting mouse position and angle variables directly.

### Rules-Based Policy

Simple heuristic with two modes:

1. **Flee**: If enemy snake is within 300 units, move in opposite direction
2. **Seek**: Otherwise, move towards nearest food pellet

### Reinforcement Learning

**Observation Space** (18 dimensions):

-   Current angle, snake length
-   Nearest food/prey/enemy distance & angle
-   Count of nearby foods, preys, enemies
-   Food distribution across 4 quadrants
-   Relative food angle, food efficiency, enemy threat

**Action Space**: Continuous value in [-1, 1] mapped to angle [0, 360]°

**Reward Function**:

-   +15 per unit of length increase
-   -5 per step (encourages efficiency)
-   -50 for dying + 0.5 × final length

**Algorithms**:

-   **REINFORCE**: Updates at end of episode, higher variance
-   **A2C**: N-step updates during episode, more stable for long games

## Output Files

### Training Logs (`training_logs/`)

```json
{
  "algorithm": "A2C",
  "episode_rewards": [...],
  "episode_max_lengths": [...],
  "best_length": 150,
  "timestamp": "20251216_123456"
}
```

### Inference Logs (`inference_logs/`)

```json
{
    "agent_type": "a2c",
    "game_snake_lengths": [45, 67, 89],
    "best_length": 89
}
```

### Trajectories (`trajectories/`)

```json
{
  "metadata": {"num_steps": 500, "final_length": 75},
  "steps": [
    {"observation": [...], "action": 0.5, "reward": 10.0}
  ]
}
```

## Tips

-   **A2C is recommended** over REINFORCE for long episodes
-   Use `--action-delay 0.1` for faster gameplay, `0.2` for more stable training
-   Pre-training on rules-based trajectories can speed up RL learning
-   Record videos (`--record-video`) to debug agent behavior
-   Check `inference_logs/` for comparison statistics

## Requirements

-   Python 3.10+
-   Chrome browser
-   ChromeDriver (matching your Chrome version)
-   See `requirements.txt` for Python packages
