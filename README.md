# Slither Bot

An autonomous bot that plays [Slither.io](http://slither.io) using Selenium WebDriver, with both rule-based and reinforcement learning approaches.

## Features

-   **Rules-based policy**: Hand-crafted food-seeking and danger avoidance behavior
-   **RL agents**: REINFORCE, A2C (Actor-Critic), and PPO (Proximal Policy Optimization) agents that learn through online training
-   **Imitation learning**: Collect human or auto-generated trajectories for pre-training
-   **Video recording**: Record gameplay videos with model state annotations
-   **State extraction**: Extracts game world data via JavaScript injection

## Project Structure

```
slither-bot/
├── src/                       # Core library modules
│   ├── __init__.py            # Package exports
│   ├── agents.py              # RL agents (REINFORCE, A2C, PPO)
│   ├── controller.py          # Selenium game controller
│   ├── environment.py         # Gym environment wrapper
│   ├── rules_policy.py        # Heuristic policy
│   ├── collect.py             # Human trajectory collection
│   ├── pretrain.py            # Supervised pre-training
│   ├── compare.py             # Agent comparison utilities
│   ├── profile.py             # Performance profiling
│   └── utils.py               # Shared helper functions
├── train.py                   # Train RL agents
├── play.py                    # Play with trained models
├── collect.py                 # Collect trajectories
├── pretrain.py                # Pre-train on trajectories
├── compare.py                 # Compare agent performance
├── visualize.ipynb            # Notebook for data visualization
├── Plot Results.ipynb         # Notebook for plotting results
├── Compare Results.ipynb      # Notebook for comparing agents
├── assets/                    # Images and static assets
├── models/                    # Saved model checkpoints
├── trajectories/              # Collected trajectory data
├── training_logs/             # Training metrics
├── inference_logs/            # Inference metrics
└── games/                     # Recorded gameplay videos
```

## Installation

```bash
pip install -r requirements.txt
```

You'll also need [ChromeDriver](https://chromedriver.chromium.org/) installed and in your PATH.

## Quick Start

### 1. Train an RL Agent

```bash
# Train with A2C (recommended for fast training)
python train.py --algorithm a2c --episodes 50

# Train with PPO (recommended for stability)
python train.py --algorithm ppo --episodes 50

# Train with REINFORCE
python train.py --algorithm reinforce --episodes 50
```

### 2. Play with a Trained Model

```bash
# Play with best A2C model
python play.py --agent a2c --model models/best_model_a2c.pt --games 5

# Play with best PPO model (if available)
python play.py --agent a2c --model models/best_model_ppo.pt --games 5

# Play with rules-based policy
python play.py --agent rules --games 5
```

## Usage Examples

### Training

```bash
# Basic A2C training
python train.py --algorithm a2c --episodes 100

# Basic PPO training (recommended for best results)
python train.py --algorithm ppo --episodes 100

# A2C with custom N-step updates (default: 64)
python train.py --algorithm a2c --episodes 250 --n-steps 64

# PPO with custom parameters (default n-steps: 256, batch-size: 64, n-epochs: 10)
python train.py --algorithm ppo --episodes 100 --n-steps 256 --batch-size 64 --n-epochs 10

# Train from pre-trained checkpoint
python train.py --algorithm a2c --pretrained models/pretrained.pt

# Adjust exploration with entropy coefficient (default: 0.05)
python train.py --algorithm a2c --entropy-coef 0.1  # Increase if policy gets stuck

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

# Play without model (untrained/random weights baseline)
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

The bot injects JavaScript to read game state variables (`window.snake`, `window.foods`, `window.slithers`, `window.preys`) and controls the snake by setting mouse position and angle variables directly.

### Rules-Based Policy

Simple heuristic with two modes:

1. **Flee**: If enemy snake is within 300 units, move in opposite direction
2. **Seek**: Otherwise, move towards nearest food pellet

### Reinforcement Learning

**Observation Space** (23 dimensions):

-   Current angle (discrete action index), snake length (log-normalized)
-   Nearest food/prey/enemy distance & action direction
-   Nearest enemy head distance
-   Count of nearby foods, preys, enemies (normalized)
-   Food efficiency (weighted by inverse distance)
-   Enemy threat level (weighted by proximity)
-   Danger levels in 4 quadrants (front/right/back/left)
-   Binary danger indicators (enemy in front/right/left)
-   Last action encoding (sin/cos for cyclical continuity)

**Action Space**: Discrete space with 12 actions representing directions at 30° intervals (0°, 30°, 60°, ..., 330°)

**Reward Function**:

-   +1.0 per unit of length increase (eating food)
-   +0.01 per step (small survival bonus)
-   -0.05 to -0.5 for proximity to enemies (scaled by distance)
-   -5.0 for dying + 0.01 × final length

**Algorithms**:

-   **REINFORCE**: Vanilla policy gradient with updates at end of episode, higher variance
-   **A2C**: Actor-Critic with N-step updates during episode, more stable for long games
-   **PPO**: Proximal Policy Optimization with clipped objective and mini-batch updates, good balance of stability and sample efficiency

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

-   **PPO is recommended** for best stability and sample efficiency, followed by A2C for faster training
-   **REINFORCE** has higher variance and is less stable for long episodes
-   Use `--action-delay 0.1` for faster gameplay, `0.15-0.2` for more stable training
-   Increase `--entropy-coef` (e.g., 0.1 or 0.15) if the policy collapses or gets stuck in local optima
-   Pre-training on rules-based trajectories can speed up RL learning
-   Record videos (`--record-video`) to debug agent behavior
-   Check `inference_logs/` for comparison statistics

## Requirements

-   Python 3.10+
-   Chrome browser
-   ChromeDriver (matching your Chrome version)
-   See `requirements.txt` for Python packages
