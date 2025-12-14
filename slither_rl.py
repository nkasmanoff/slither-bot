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
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from slither import SlitherController


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

    def __init__(self, driver, save_screenshots=False):
        super(SlitherEnv, self).__init__()
        self.controller = SlitherController(
            driver=driver, save_screenshots=save_screenshots
        )

        # Action space: 8 discrete directions (0, 45, 90, 135, 180, 225, 270, 315 degrees)
        self.action_space = gym.spaces.Discrete(8)

        # Observation space: [
        #   current_angle (normalized),
        #   snake_length (normalized),
        #   nearest_food_distance (normalized),
        #   nearest_food_angle (normalized),
        #   nearest_enemy_distance (normalized),
        #   nearest_enemy_angle (normalized),
        #   num_nearby_foods,
        #   num_nearby_enemies
        # ]
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.previous_length = 0
        self.steps = 0
        self.driver = driver

    def _get_obs(self):
        """Extract observation from game state."""
        state = self.controller.get_detailed_state()

        if not state or not state.get("snake"):
            # Return zero observation if game not ready
            return np.zeros(8, dtype=np.float32)

        snake = state["snake"]
        foods = state.get("foods", [])
        enemies = state.get("other_snakes", [])

        # Current angle (normalized to [-1, 1])
        current_angle = snake.get("angle", 0) / np.pi

        # Snake length (log-normalized for better scale)
        snake_length = np.log(max(snake.get("length", 1), 1)) / 10.0

        # Nearest food
        if foods:
            nearest_food = foods[0]
            nearest_food_dist = np.tanh(nearest_food["distance"] / 500.0)  # Normalize
            nearest_food_angle = nearest_food["angle"] / np.pi
        else:
            nearest_food_dist = 1.0
            nearest_food_angle = 0.0

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
        num_enemies = min(len(enemies), 15) / 15.0

        obs = np.array(
            [
                current_angle,
                snake_length,
                nearest_food_dist,
                nearest_food_angle,
                nearest_enemy_dist,
                nearest_enemy_angle,
                num_foods,
                num_enemies,
            ],
            dtype=np.float32,
        )

        return obs

    def _action_to_angle(self, action):
        """Convert discrete action to angle in degrees."""
        return action * 45  # 0, 45, 90, 135, 180, 225, 270, 315

    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        # Convert action to angle and move
        angle = self._action_to_angle(action)
        self.controller.move_to_angle(angle)

        # Small delay to let game update
        time.sleep(0.25)

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

    def select_action(self, state):
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()

        # Save log probability for training
        self.saved_log_probs.append(m.log_prob(action))

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


def train_agent(num_episodes=100):
    """Train the REINFORCE agent on Slither.io."""
    # Setup driver
    driver = webdriver.Chrome()
    driver.get("http://slither.io")

    print("Waiting for game to load...")
    time.sleep(5)

    # Start first game
    play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_buttons:
        if "Play" in button.text:
            button.click()
            break

    print("Game started!")
    time.sleep(2)

    # Create environment and agent
    env = SlitherEnv(driver, save_screenshots=False)
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=0.001,
        gamma=0.99,
    )

    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Track best performance
    best_length = 0
    best_reward = float("-inf")

    print(f"Starting training for {num_episodes} episodes...")
    print(
        f"State dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.n}"
    )

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        max_length = 0

        done = False
        while not done:
            # Select action from policy
            action = agent.select_action(state)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Store reward
            agent.store_reward(reward)

            episode_reward += reward
            episode_length += 1
            max_length = max(max_length, info["length"])

            state = next_state

            # Limit episode length to prevent infinite loops
            if episode_length > 1000:
                done = True

        # Update policy after episode ends
        loss = agent.update_policy()

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

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(models_dir, f"final_model_{timestamp}.pt")
    agent.save_model(final_model_path)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best Length Achieved: {best_length}")
    print(f"Best Model Reward: {best_reward:.2f}")
    print(f"Best model saved at: {os.path.join(models_dir, 'best_model.pt')}")
    print(f"Final model saved at: {final_model_path}")
    print("=" * 50)

    driver.quit()

    return agent


def load_and_play(model_path="models/best_model.pt", num_games=5):
    """Load a trained model and play games with it."""
    # Setup driver
    driver = webdriver.Chrome()
    driver.get("http://slither.io")

    print("Waiting for game to load...")
    time.sleep(5)

    # Start first game
    play_buttons = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_buttons:
        if "Play" in button.text:
            button.click()
            break

    print("Game started!")
    time.sleep(2)

    # Create environment and agent
    env = SlitherEnv(driver, save_screenshots=False)
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
            # Select action using loaded policy
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            game_length += 1
            max_length = max(max_length, info["length"])
            state = next_state

            if game_length > 1000:
                done = True

        print(
            f"Game {game + 1}/{num_games} | Steps: {game_length} | Max Length: {max_length}"
        )

    driver.quit()
    print("Demo complete!")


if __name__ == "__main__":
    # Train the agent
    trained_agent = train_agent(num_episodes=50)

    # To load and play with a trained model instead, uncomment:
    # load_and_play(model_path="models/best_model.pt", num_games=5)
