"""
Reinforcement learning wrapper around slither.py with simple REINFORCE policy gradient.

This implements online learning where:
- Reward is based on food collection (length increase) and survival time
- Uses a simple neural network policy
- Trains using REINFORCE algorithm after each episode
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import os
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from slither import SlitherController

# Set to True if running on Raspberry Pi
IS_RASPBERRY_PI = False


STAGNATION_WINDOW_SIZE = 100
STAGNATION_THRESHOLD_SIZE = 10

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


class PolicyNetwork(nn.Module):
    """Simple neural network policy for action selection."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class SlitherEnv(gym.Env):
    """Gym environment wrapper for Slither.io game."""

    def __init__(self, driver, save_screenshots=False, record_video=False):
        super(SlitherEnv, self).__init__()
        self.controller = SlitherController(
            driver=driver, save_screenshots=save_screenshots, record_video=record_video
        )

        # Action space: 8 discrete directions (0, 45, 90, 135, 180, 225, 270, 315 degrees)
        self.action_space = gym.spaces.Discrete(8)

        # Observation space: [
        #   current_angle (normalized),
        #   snake_length (normalized),
        #   nearest_food_distance (normalized),
        #   nearest_food_angle (normalized),
        #   nearest_prey_distance (normalized),
        #   nearest_prey_angle (normalized),
        #   nearest_enemy_distance (normalized),
        #   nearest_enemy_angle (normalized),
        #   num_nearby_foods,
        #   num_nearby_preys,
        #   num_nearby_enemies,
        #   food_quadrant_1_count (normalized),
        #   food_quadrant_2_count (normalized),
        #   food_quadrant_3_count (normalized),
        #   food_quadrant_4_count (normalized),
        # ]
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )

        self.previous_length = 0
        self.steps = 0
        self.driver = driver

    def _get_obs(self):
        """Extract observation from game state."""
        state = self.controller.get_detailed_state()

        if not state or not state.get("snake"):
            # Return zero observation if game not ready
            return np.zeros(15, dtype=np.float32)

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
        food_quadrants = [
            min(count, max_foods) / max_foods for count in quadrant_counts
        ]

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

    def _action_to_angle(self, action):
        """Convert discrete action to angle in degrees."""
        return action * 45  # 0, 45, 90, 135, 180, 225, 270, 315

    def step(self, action, observation=None, probabilities=None):
        """Execute action and return next state, reward, done, info.

        Args:
            action: Action to execute
            observation: Optional observation for video annotation
            probabilities: Optional action probabilities for video annotation
        """
        # Convert action to angle and move
        angle = self._action_to_angle(action)
        self.controller.move_to_angle(angle)

        # Set annotation for the frame that was just captured
        if observation is not None or probabilities is not None:
            self.controller.set_frame_annotation(
                observation=observation, probabilities=probabilities, action=action
            )

        # Small delay to let game update, capture frames during delay
        # Capture frames more frequently to avoid missing gameplay
        delay = 0.25
        num_captures = 3  # Capture 3 frames during the delay
        capture_interval = delay / num_captures

        for i in range(num_captures):
            time.sleep(capture_interval)
            # Capture intermediate frames (without annotation, they'll use previous annotation)
            if (
                i < num_captures - 1
            ):  # Don't capture on last iteration, we'll get obs next
                self.controller.capture_frame_only()

        # Get new observation
        obs = self._get_obs()

        # Calculate reward based on length increase and survival
        current_length = self.controller.get_snake_length()
        length_increase = current_length - self.previous_length

        # Reward components:
        # 1. Food collection (length increase) - main reward
        # 2. Small survival bonus per step
        # 3. Penalty for dying
        reward = 0.0
        reward += length_increase * 10.0  # Large reward for eating food
        reward -= 2.5  # Small penalty for duration

        self.previous_length = current_length
        self.steps += 1

        # Check if game is over
        done = self.controller.is_game_over()

        if done:
            # Penalty for dying, but offset by length achieved
            reward -= 50.0
            reward += current_length * 0.5  # Bonus for final length

        info = {
            "length": current_length,
            "steps": self.steps,
            "length_increase": length_increase,
        }

        return obs, reward, done, info

    def reset(self):
        """Reset environment for new episode."""
        # Note: Video creation is now handled in the training loop after each episode
        # This prevents double-saving and ensures videos are saved even if reset() is called multiple times

        # If game is over, click play button
        if self.controller.is_game_over():
            try:
                play_buttons = self.driver.find_elements(By.CLASS_NAME, "btnt")
                for button in play_buttons:
                    if "Play" in button.text and button.is_displayed():
                        button.click()
                        break
                time.sleep(3)  # Wait for game to start
            except Exception as e:
                print(f"Error resetting: {e}")

        self.previous_length = self.controller.get_snake_length()
        self.steps = 0

        return self._get_obs()


class REINFORCEAgent:
    """Simple REINFORCE policy gradient agent."""

    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

        # Storage for episode
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, return_probs=False):
        """Select action using current policy.

        Args:
            state: Current state observation
            return_probs: If True, also return action probabilities

        Returns:
            action (int): Selected action
            probs (torch.Tensor, optional): Action probabilities if return_probs=True
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()

        # Save log probability for training
        self.saved_log_probs.append(m.log_prob(action))

        if return_probs:
            return action.item(), probs.squeeze().detach().cpu().numpy()
        return action.item()

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns for stable training
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.rewards) == 0:
            return 0.0

        # Compute returns
        returns = self.compute_returns(self.rewards)

        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Backprop and update
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode storage
        loss_value = policy_loss.item()
        self.saved_log_probs = []
        self.rewards = []

        return loss_value

    def store_reward(self, reward):
        """Store reward from environment step."""
        self.rewards.append(reward)

    def save_model(self, filepath):
        """Save model weights to file."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights from file."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {filepath}")


def setup_browser_and_game(record_video=False):
    """Set up browser and start a new game.

    Returns:
        driver: WebDriver instance
        env: SlitherEnv instance
    """
    # Setup driver
    if IS_RASPBERRY_PI:
        # Raspberry Pi configuration with kiosk mode
        options = Options()
        options.binary_location = "/usr/bin/chromium-browser"
        options.add_argument("--kiosk")  # Fullscreen with no navigation bar
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Standard Chrome driver
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

    # Start game
    play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_buttons:
        if "Play" in button.text:
            button.click()
            break

    print("Game started!")
    time.sleep(2)

    # Create environment
    env = SlitherEnv(driver, save_screenshots=False, record_video=record_video)

    return driver, env


def train_agent(num_episodes=100, record_video=False, pretrained_model_path=None):
    """Train the REINFORCE agent on Slither.io.

    Args:
        num_episodes: Number of training episodes.
        record_video: Whether to record gameplay videos.
        pretrained_model_path: Optional path to a model checkpoint created by
            `slither_human_pretrain.py` (or another script) using the same
            `PolicyNetwork` architecture. If provided, the policy and optimizer
            weights are loaded before RL training begins.
    """
    # Setup initial browser and game
    driver, env = setup_browser_and_game(record_video=record_video)
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=0.001,
        gamma=0.99,
    )

    # Optionally warm-start from a pre-trained policy
    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        print(
            f"Loading pre-trained model from {pretrained_model_path} before RL training."
        )
        agent.load_model(pretrained_model_path)
    elif pretrained_model_path is not None:
        print(
            f"Warning: pretrained_model_path '{pretrained_model_path}' not found; training from scratch."
        )

    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Track best performance
    best_length = 0
    best_reward = float("-inf")

    # Track metrics for logging
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_max_lengths = []

    print(f"Starting training for {num_episodes} episodes...")
    print(
        f"State dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.n}"
    )

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        max_length = 0
        recent_lengths = []  # Track lengths over last 5 steps

        done = False
        while not done:
            # Select action from policy (also get probabilities for annotation)
            action, probs = agent.select_action(state, return_probs=True)

            # Take step in environment (pass observation and probabilities for video annotation)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )

            # Store reward
            agent.store_reward(reward)

            episode_reward += reward
            episode_length += 1
            current_length = info["length"]
            max_length = max(max_length, current_length)

            # Track recent lengths for stagnation check
            recent_lengths.append(current_length)
            if len(recent_lengths) > STAGNATION_WINDOW_SIZE:
                recent_lengths.pop(0)  # Keep only last STAGNATION_WINDOW_SIZE steps

            state = next_state

            # Check if length hasn't increased by more than STAGNATION_THRESHOLD_SIZE in last STAGNATION_WINDOW_SIZE steps
            if len(recent_lengths) >= STAGNATION_WINDOW_SIZE:
                length_increase = recent_lengths[-1] - recent_lengths[0]
                if length_increase <= STAGNATION_THRESHOLD_SIZE:
                    done = True

            # Limit episode length to prevent infinite loops
            if episode_length > 1000:
                done = True

        # Save video for this episode before resetting
        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Episode {episode + 1} video saved to: {video_path}")

        # Update policy after episode ends
        loss = agent.update_policy()

        # Log metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_losses.append(loss)
        episode_max_lengths.append(max_length)

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Steps: {episode_length} | "
            f"Reward: {episode_reward:.2f} | "
            f"Max Length: {max_length} | "
            f"Loss: {loss:.4f}"
        )

        # Save model if it's the best so far (based on max_length)
        if max_length > best_length:
            best_length = max_length
            best_reward = episode_reward
            model_path = os.path.join(models_dir, "best_model.pt")
            agent.save_model(model_path)
            print(f"🏆 New best model! Length: {best_length}")

        # Quit browser and restart for next episode (unless it's the last episode)
        if episode < num_episodes - 1:
            print("Quitting browser and restarting for next episode...")
            driver.quit()
            driver, env = setup_browser_and_game(record_video=record_video)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(models_dir, f"final_model_{timestamp}.pt")
    agent.save_model(final_model_path)

    # Save training metrics to JSON file
    metrics_data = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_max_lengths": episode_max_lengths,
        "num_episodes": num_episodes,
        "best_length": best_length,
        "best_reward": best_reward,
        "timestamp": timestamp,
    }

    metrics_dir = "training_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"training_metrics_{timestamp}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best Length Achieved: {best_length}")
    print(f"Best Model Reward: {best_reward:.2f}")
    print(f"Best model saved at: {os.path.join(models_dir, 'best_model.pt')}")
    print(f"Final model saved at: {final_model_path}")
    print(f"Training metrics saved at: {metrics_path}")
    print("=" * 50)

    driver.quit()

    return agent, metrics_path


def plot_training_metrics(metrics_path=None, save_path=None):
    """Plot training metrics from a saved JSON file.

    Args:
        metrics_path: Path to the metrics JSON file. If None, uses the most recent one.
        save_path: Path to save the plot. If None, displays the plot.
    """
    import matplotlib.pyplot as plt

    # Load metrics
    if metrics_path is None:
        # Find the most recent metrics file
        metrics_dir = "training_logs"
        if not os.path.exists(metrics_dir):
            print(f"Error: {metrics_dir} directory not found. Run training first.")
            return

        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith(".json")]
        if not metrics_files:
            print(f"Error: No metrics files found in {metrics_dir}")
            return

        metrics_files.sort(reverse=True)
        metrics_path = os.path.join(metrics_dir, metrics_files[0])
        print(f"Using most recent metrics file: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    episode_rewards = metrics["episode_rewards"]
    episode_lengths = metrics["episode_lengths"]
    episode_losses = metrics["episode_losses"]
    episode_max_lengths = metrics["episode_max_lengths"]
    num_episodes = metrics["num_episodes"]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    episodes = range(1, num_episodes + 1)

    # Plot rewards
    axes[0].plot(episodes, episode_rewards, alpha=0.6, linewidth=1)
    axes[0].plot(episodes, episode_rewards, "o", markersize=3, alpha=0.4)
    # Add moving average
    if len(episode_rewards) > 10:
        window = min(10, len(episode_rewards) // 5)
        moving_avg = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        axes[0].plot(
            range(window, num_episodes + 1),
            moving_avg,
            "r-",
            linewidth=2,
            label=f"{window}-episode moving average",
        )
        axes[0].legend()
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Episode Reward Over Time")
    axes[0].grid(True, alpha=0.3)

    # Plot episode lengths
    axes[1].plot(episodes, episode_lengths, alpha=0.6, linewidth=1, color="green")
    axes[1].plot(episodes, episode_lengths, "o", markersize=3, alpha=0.4, color="green")
    # Add moving average
    if len(episode_lengths) > 10:
        window = min(10, len(episode_lengths) // 5)
        moving_avg = np.convolve(
            episode_lengths, np.ones(window) / window, mode="valid"
        )
        axes[1].plot(
            range(window, num_episodes + 1),
            moving_avg,
            "r-",
            linewidth=2,
            label=f"{window}-episode moving average",
        )
        axes[1].legend()
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length (steps)")
    axes[1].set_title("Episode Length Over Time")
    axes[1].grid(True, alpha=0.3)

    # Plot losses
    axes[2].plot(episodes, episode_losses, alpha=0.6, linewidth=1, color="red")
    axes[2].plot(episodes, episode_losses, "o", markersize=3, alpha=0.4, color="red")
    # Add moving average
    if len(episode_losses) > 10:
        window = min(10, len(episode_losses) // 5)
        moving_avg = np.convolve(episode_losses, np.ones(window) / window, mode="valid")
        axes[2].plot(
            range(window, num_episodes + 1),
            moving_avg,
            "b-",
            linewidth=2,
            label=f"{window}-episode moving average",
        )
        axes[2].legend()
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Policy Loss Over Time")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Training Summary Statistics")
    print("=" * 50)
    print(f"Total Episodes: {num_episodes}")
    print("\nReward Statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std: {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print("\nEpisode Length Statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.2f}")
    print(f"  Std: {np.std(episode_lengths):.2f}")
    print(f"  Min: {np.min(episode_lengths):.0f}")
    print(f"  Max: {np.max(episode_lengths):.0f}")
    print("\nLoss Statistics:")
    print(f"  Mean: {np.mean(episode_losses):.4f}")
    print(f"  Std: {np.std(episode_losses):.4f}")
    print(f"  Min: {np.min(episode_losses):.4f}")
    print(f"  Max: {np.max(episode_losses):.4f}")
    print("\nMax Length Statistics:")
    print(f"  Mean: {np.mean(episode_max_lengths):.2f}")
    print(f"  Std: {np.std(episode_max_lengths):.2f}")
    print(f"  Min: {np.min(episode_max_lengths):.0f}")
    print(f"  Max: {np.max(episode_max_lengths):.0f}")
    print("=" * 50)


def load_and_play(model_path="models/best_model.pt", num_games=5, record_video=False):
    """Load a trained model and play games with it."""
    # Setup driver
    if IS_RASPBERRY_PI:
        # Raspberry Pi configuration with kiosk mode
        options = Options()
        options.binary_location = "/usr/bin/chromium-browser"
        options.add_argument("--kiosk")  # Fullscreen with no navigation bar
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Standard Chrome driver
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

    # Start first game
    play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_buttons:
        if "Play" in button.text:
            button.click()
            break

    print("Game started!")
    time.sleep(2)

    # Create environment and agent
    env = SlitherEnv(driver, save_screenshots=False, record_video=record_video)
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    # Load the trained model
    agent.load_model(model_path)

    print(f"\nPlaying {num_games} games with loaded model...")

    for game in range(num_games):
        state = env.reset()
        game_length = 0
        max_length = 0
        done = False

        while not done:
            # Select action using loaded policy (also get probabilities for annotation)
            action, probs = agent.select_action(state, return_probs=True)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )

            game_length += 1
            max_length = max(max_length, info["length"])
            state = next_state

            if game_length > 1000:
                done = True

        print(
            f"Game {game + 1}/{num_games} | Steps: {game_length} | Max Length: {max_length}"
        )

        # Save video for this game
        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Game {game + 1} video saved to: {video_path}")

    driver.quit()
    print("Demo complete!")


if __name__ == "__main__":
    # Optional: warm-start from a pre-trained supervised model.
    # If the environment variable SLITHER_PRETRAINED_MODEL is set to a valid
    # checkpoint path, it will be loaded before RL training begins.

    pretrained_model_path = os.path.join(
        os.path.dirname(__file__), "models", "human_pretrained.pt"
    )
    # Train the agent
    trained_agent, metrics_path = train_agent(
        num_episodes=50,
        record_video=False,
        pretrained_model_path=None,
    )

    # Plot training metrics
    print("\nGenerating training plots...")
    plot_training_metrics(
        metrics_path=metrics_path, save_path=metrics_path.replace(".json", ".png")
    )  # Set save_path to save instead of display

    # To load and play with a trained model instead, uncomment:
    # load_and_play(model_path="models/best_model.pt", num_games=1, record_video=True)
