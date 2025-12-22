"""Gym environment wrapper for Slither.io with RL training support.

Provides SlitherEnv class and training functions for REINFORCE and A2C algorithms.
"""

import json
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from selenium.webdriver.common.by import By

from .agents import A2CAgent, PPOAgent, REINFORCEAgent
from .controller import SlitherController
from .utils import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    discrete_action_to_angle,
    extract_observation,
    setup_browser,
    start_game,
)

STAGNATION_WINDOW_SIZE = 500
STAGNATION_THRESHOLD_SIZE = 10


class SlitherEnv(gym.Env):
    """Gym environment wrapper for Slither.io game."""

    def __init__(
        self, driver, save_screenshots=False, record_video=False, action_delay=0.15
    ):
        """Initialize the Slither.io environment.

        Args:
            driver: Selenium WebDriver instance.
            save_screenshots: Whether to save screenshots to disk.
            record_video: Whether to record gameplay video.
            action_delay: Delay between actions in seconds.
        """
        super().__init__()
        self.controller = SlitherController(driver, save_screenshots, record_video)
        self.action_delay = action_delay
        # Discrete action space: 12 directions at 30° intervals
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.previous_length = 0
        self.steps = 0
        self.driver = driver
        self.last_action = None  # Track last action for observation

    def _get_obs(self, state=None):
        """Extract observation from game state."""
        if state is None:
            state = self.controller.get_detailed_state()
        return extract_observation(state, last_action=self.last_action)

    def step(self, action, observation=None, probabilities=None):
        """Execute action and return next state, reward, done, info.

        The action is a discrete index (0 to NUM_ACTIONS-1) representing
        absolute directions at 30° intervals:
        - action 0 = 0°, action 1 = 30°, ..., action 11 = 330°
        """
        # Convert discrete action to angle in degrees
        angle_deg = discrete_action_to_angle(action)
        self.controller.move_to_angle(angle_deg)

        # Store action for next observation (temporal context)
        self.last_action = action

        if observation is not None or probabilities is not None:
            self.controller.set_frame_annotation(
                observation=observation, probabilities=probabilities, action=action
            )

        delay = self.action_delay
        num_captures = 1
        capture_interval = delay / num_captures

        for i in range(num_captures):
            time.sleep(capture_interval)
            if i < num_captures - 1:
                self.controller.capture_frame_only()

        full_state = self.controller.get_full_state()
        obs = self._get_obs(state=full_state)

        current_length = full_state.get("snake_length", 0) if full_state else 0
        done = full_state.get("is_game_over", False) if full_state else False
        length_increase = current_length - self.previous_length

        # === Reward Shaping ===
        # NOTE: Rewards scaled down to keep values stable for RL training
        reward = 0.0

        # Reward for eating food (growing)
        reward += length_increase * 1.0

        # Small survival bonus (encourages staying alive)
        reward += 0.01

        # Danger proximity penalty - scaled down for stability
        enemies = full_state.get("other_snakes", []) if full_state else []
        if enemies:
            nearest_enemy_dist = enemies[0].get("distance", 1000)
            nearest_enemy_head_dist = enemies[0].get("head_distance", 1000)
            # Use the minimum of body and head distance for danger
            min_danger_dist = min(nearest_enemy_dist, nearest_enemy_head_dist)

            if min_danger_dist < 60:  # Critical danger zone
                reward -= 0.5
            elif min_danger_dist < 120:  # High danger
                reward -= 0.2
            elif min_danger_dist < 200:  # Moderate danger
                reward -= 0.05

        self.previous_length = current_length
        self.steps += 1

        if done:
            # Death penalty - significant but not overwhelming
            reward -= 5.0
            # Small bonus for length achieved (encourages longer survival)
            reward += current_length * 0.01

        info = {
            "length": current_length,
            "steps": self.steps,
            "length_increase": length_increase,
        }
        return obs, reward, done, info

    def reset(self):
        """Reset environment for new episode."""
        if self.controller.is_game_over():
            try:
                play_buttons = self.driver.find_elements(By.CLASS_NAME, "btnt")
                for button in play_buttons:
                    if "Play" in button.text and button.is_displayed():
                        button.click()
                        break
                time.sleep(3)
            except Exception as e:
                print(f"Error resetting: {e}")

        max_wait = 10
        start_time = time.time()
        while time.time() - start_time < max_wait:
            length = self.controller.get_snake_length()
            if 0 < length < 50:
                break
            time.sleep(0.5)

        self.previous_length = self.controller.get_snake_length()
        self.steps = 0
        self.last_action = None  # Reset action history for new episode
        return self._get_obs()


def setup_browser_and_game(record_video=False, action_delay=0.15):
    """Set up browser and start a new game.

    Returns:
        driver: WebDriver instance.
        env: SlitherEnv instance.
    """
    driver = setup_browser()
    start_game(driver)
    env = SlitherEnv(
        driver,
        save_screenshots=False,
        record_video=record_video,
        action_delay=action_delay,
    )
    print(
        f"Decision frequency: ~{1/action_delay:.1f} Hz (action_delay={action_delay}s)"
    )
    return driver, env


def train_agent(
    num_episodes=100, record_video=False, pretrained_model_path=None, action_delay=0.15
):
    """Train the REINFORCE agent on Slither.io."""
    driver, env = setup_browser_and_game(
        record_video=record_video, action_delay=action_delay
    )
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        learning_rate=0.001,
        gamma=0.99,
    )

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained model from {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
    elif pretrained_model_path:
        print(f"Warning: pretrained_model_path '{pretrained_model_path}' not found")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    best_length = 0
    best_reward = float("-inf")
    episode_rewards, episode_lengths, episode_losses, episode_max_lengths = (
        [],
        [],
        [],
        [],
    )

    print(f"Starting training for {num_episodes} episodes...")
    print(
        f"State dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.shape[0]}"
    )

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward, episode_length, max_length = 0, 0, 0
        recent_lengths = []
        done = False

        while not done:
            action, probs = agent.select_action(state, return_probs=True)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )
            agent.store_reward(reward)

            episode_reward += reward
            episode_length += 1
            current_length = info["length"]
            max_length = max(max_length, current_length)

            recent_lengths.append(current_length)
            if len(recent_lengths) > STAGNATION_WINDOW_SIZE:
                recent_lengths.pop(0)

            state = next_state

            if len(recent_lengths) >= STAGNATION_WINDOW_SIZE:
                if recent_lengths[-1] - recent_lengths[0] <= STAGNATION_THRESHOLD_SIZE:
                    done = True

            if episode_length > 1000:
                done = True

        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Episode {episode + 1} video saved to: {video_path}")

        loss = agent.update_policy()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_losses.append(loss)
        episode_max_lengths.append(max_length)

        print(
            f"Episode {episode + 1}/{num_episodes} | Steps: {episode_length} | Reward: {episode_reward:.2f} | Max Length: {max_length} | Loss: {loss:.4f}"
        )

        if max_length > best_length:
            best_length = max_length
            best_reward = episode_reward
            agent.save_model(os.path.join(models_dir, "best_model.pt"))
            print(f"New best model! Length: {best_length}")

        if episode < num_episodes - 1:
            print("Restarting browser for next episode...")
            driver.quit()
            driver, env = setup_browser_and_game(
                record_video=record_video, action_delay=action_delay
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(models_dir, f"final_model_{timestamp}.pt")
    agent.save_model(final_model_path)

    _save_training_metrics(
        episode_rewards,
        episode_lengths,
        episode_losses,
        episode_max_lengths,
        num_episodes,
        best_length,
        best_reward,
        timestamp,
        "reinforce",
    )

    driver.quit()
    return agent


def train_agent_a2c(
    num_episodes=100,
    record_video=False,
    pretrained_model_path=None,
    n_steps=64,
    learning_rate=0.0003,
    action_delay=0.15,
    entropy_coef=0.05,
):
    """Train the A2C agent on Slither.io."""
    driver, env = setup_browser_and_game(
        record_video=record_video, action_delay=action_delay
    )
    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=n_steps,
        value_loss_coef=0.5,
        entropy_coef=entropy_coef,  # Higher entropy prevents policy collapse
    )

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained model from {pretrained_model_path}")
        try:
            agent.load_model(pretrained_model_path)
        except KeyError:
            print("Checkpoint appears to be from PolicyNetwork, converting...")
            agent.load_from_policy_network(pretrained_model_path)
    elif pretrained_model_path:
        print(f"Warning: pretrained_model_path '{pretrained_model_path}' not found")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    best_length = 0
    best_reward = float("-inf")
    episode_rewards, episode_lengths, episode_losses, episode_max_lengths = (
        [],
        [],
        [],
        [],
    )

    print(f"Starting A2C training for {num_episodes} episodes...")
    print(f"N-step updates every {n_steps} steps, entropy_coef={entropy_coef}")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward, episode_length, max_length = 0, 0, 0
        recent_lengths = []
        episode_update_losses = []
        episode_entropies = []
        done = False

        while not done:
            action, probs = agent.select_action(state, return_probs=True)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )
            agent.store_reward(reward, done=done)

            episode_reward += reward
            episode_length += 1
            current_length = info["length"]
            max_length = max(max_length, current_length)

            recent_lengths.append(current_length)
            if len(recent_lengths) > STAGNATION_WINDOW_SIZE:
                recent_lengths.pop(0)

            if agent.should_update():
                loss_info = agent.update_policy(
                    next_state=next_state if not done else None
                )
                episode_update_losses.append(loss_info["total_loss"])
                episode_entropies.append(loss_info.get("entropy", 0.0))

            state = next_state

            if len(recent_lengths) >= STAGNATION_WINDOW_SIZE:
                if recent_lengths[-1] - recent_lengths[0] <= STAGNATION_THRESHOLD_SIZE:
                    done = True

            if episode_length > 1000:
                done = True

        if len(agent.rewards) > 0:
            loss_info = agent.update_policy(next_state=None)
            episode_update_losses.append(loss_info["total_loss"])
            episode_entropies.append(loss_info.get("entropy", 0.0))

        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Episode {episode + 1} video saved to: {video_path}")

        avg_loss = np.mean(episode_update_losses) if episode_update_losses else 0.0
        avg_entropy = np.mean(episode_entropies) if episode_entropies else 0.0

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_losses.append(avg_loss)
        episode_max_lengths.append(max_length)

        # Log entropy - if it drops below ~1.0 the policy is collapsing
        print(
            f"Episode {episode + 1}/{num_episodes} | Steps: {episode_length} | Reward: {episode_reward:.2f} | "
            f"Max Length: {max_length} | Loss: {avg_loss:.2f} | Entropy: {avg_entropy:.3f}"
        )

        if max_length > best_length:
            best_length = max_length
            best_reward = episode_reward
            agent.save_model(os.path.join(models_dir, "best_model_a2c.pt"))
            print(f"New best model! Length: {best_length}")

        if episode < num_episodes - 1:
            print("Restarting browser for next episode...")
            driver.quit()
            driver, env = setup_browser_and_game(
                record_video=record_video, action_delay=action_delay
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(models_dir, f"final_model_a2c_{timestamp}.pt")
    agent.save_model(final_model_path)

    _save_training_metrics(
        episode_rewards,
        episode_lengths,
        episode_losses,
        episode_max_lengths,
        num_episodes,
        best_length,
        best_reward,
        timestamp,
        "a2c",
        n_steps=n_steps,
        total_updates=agent.total_updates,
    )

    driver.quit()
    return agent


def train_agent_ppo(
    num_episodes=100,
    record_video=False,
    pretrained_model_path=None,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=0.0003,
    action_delay=0.15,
):
    """Train the PPO agent on Slither.io."""
    driver, env = setup_browser_and_game(
        record_video=record_video, action_delay=action_delay
    )
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained model from {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
    elif pretrained_model_path:
        print(f"Warning: pretrained_model_path '{pretrained_model_path}' not found")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    best_length = 0
    best_reward = float("-inf")
    episode_rewards, episode_lengths, episode_losses, episode_max_lengths = (
        [],
        [],
        [],
        [],
    )

    print(f"Starting PPO training for {num_episodes} episodes...")
    print(f"Updates every {n_steps} steps with {n_epochs} epochs")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward, episode_length, max_length = 0, 0, 0
        recent_lengths = []
        episode_update_losses = []
        done = False

        while not done:
            action, probs = agent.select_action(state, return_probs=True)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )
            agent.store_reward(reward, done=done)

            episode_reward += reward
            episode_length += 1
            current_length = info["length"]
            max_length = max(max_length, current_length)

            recent_lengths.append(current_length)
            if len(recent_lengths) > STAGNATION_WINDOW_SIZE:
                recent_lengths.pop(0)

            if agent.should_update():
                loss_info = agent.update_policy(
                    next_state=next_state if not done else None
                )
                episode_update_losses.append(loss_info["total_loss"])

            state = next_state

            if len(recent_lengths) >= STAGNATION_WINDOW_SIZE:
                if recent_lengths[-1] - recent_lengths[0] <= STAGNATION_THRESHOLD_SIZE:
                    done = True

            if episode_length > 1000:
                done = True

        if len(agent.rewards) > 0:
            loss_info = agent.update_policy(next_state=None)
            episode_update_losses.append(loss_info["total_loss"])

        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Episode {episode + 1} video saved to: {video_path}")

        avg_loss = np.mean(episode_update_losses) if episode_update_losses else 0.0

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_losses.append(avg_loss)
        episode_max_lengths.append(max_length)

        print(
            f"Episode {episode + 1}/{num_episodes} | Steps: {episode_length} | Updates: {len(episode_update_losses)} | Reward: {episode_reward:.2f} | Max Length: {max_length} | Avg Loss: {avg_loss:.4f}"
        )

        if max_length > best_length:
            best_length = max_length
            best_reward = episode_reward
            agent.save_model(os.path.join(models_dir, "best_model_ppo.pt"))
            print(f"New best model! Length: {best_length}")

        if episode < num_episodes - 1:
            print("Restarting browser for next episode...")
            driver.quit()
            driver, env = setup_browser_and_game(
                record_video=record_video, action_delay=action_delay
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(models_dir, f"final_model_ppo_{timestamp}.pt")
    agent.save_model(final_model_path)

    _save_training_metrics(
        episode_rewards,
        episode_lengths,
        episode_losses,
        episode_max_lengths,
        num_episodes,
        best_length,
        best_reward,
        timestamp,
        "ppo",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        total_updates=agent.total_updates,
    )

    driver.quit()
    return agent


def _save_training_metrics(
    episode_rewards,
    episode_lengths,
    episode_losses,
    episode_max_lengths,
    num_episodes,
    best_length,
    best_reward,
    timestamp,
    algorithm,
    **kwargs,
):
    """Save training metrics to JSON file."""
    metrics_data = {
        "algorithm": algorithm.upper(),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_losses": episode_losses,
        "episode_max_lengths": episode_max_lengths,
        "num_episodes": num_episodes,
        "best_length": best_length,
        "best_reward": best_reward,
        "timestamp": timestamp,
        **kwargs,
    }

    metrics_dir = "training_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(
        metrics_dir, f"training_metrics_{algorithm}_{timestamp}.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Length: {best_length}, Best Reward: {best_reward:.2f}")
    print(f"Metrics saved to: {metrics_path}")


def load_and_play(
    model_path=None,
    num_games=5,
    record_video=False,
    agent_type="reinforce",
    action_delay=0.1,
):
    """Load a trained model and play games with it."""
    driver = setup_browser()
    start_game(driver)

    env = SlitherEnv(
        driver,
        save_screenshots=False,
        record_video=record_video,
        action_delay=action_delay,
    )
    print(f"Decision frequency: ~{1/action_delay:.1f} Hz")

    if agent_type.lower() == "a2c":
        agent = A2CAgent(
            state_dim=env.observation_space.shape[0],
        )
    elif agent_type.lower() == "ppo":
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
        )
    else:
        agent = REINFORCEAgent(
            state_dim=env.observation_space.shape[0],
        )

    if model_path:
        agent.load_model(model_path)
    else:
        print("Using random weights (no model loaded)")

    print(f"\nPlaying {num_games} games with {agent_type.upper()} agent...")

    game_steps, game_snake_lengths, game_rewards = [], [], []
    best_length, best_reward = 0, float("-inf")

    for game in range(num_games):
        state = env.reset()
        steps, max_snake_length, total_reward = 0, 0, 0.0
        done = False

        while not done:
            action, probs = agent.select_action(state, return_probs=True)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )

            steps += 1
            total_reward += reward
            max_snake_length = max(max_snake_length, info["length"])
            state = next_state

            if steps > 1000:
                done = True

        game_steps.append(steps)
        game_snake_lengths.append(max_snake_length)
        game_rewards.append(total_reward)
        best_length = max(best_length, max_snake_length)
        best_reward = max(best_reward, total_reward)

        print(
            f"Game {game + 1}/{num_games} | Steps: {steps} | Length: {max_snake_length} | Reward: {total_reward:.2f}"
        )

        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Game {game + 1} video saved to: {video_path}")

        if hasattr(agent, "clear_buffer"):
            agent.clear_buffer()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_data = {
        "model_path": model_path,
        "agent_type": agent_type,
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
    metrics_path = os.path.join(metrics_dir, f"inference_metrics_{timestamp}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nInference metrics saved to: {metrics_path}")
    print(f"Best Length: {best_length} | Best Reward: {best_reward:.2f}")

    driver.quit()
    return metrics_data
