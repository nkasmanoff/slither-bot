"""RL Agents for Slither.io.

Contains:
- PolicyNetwork: Neural network for REINFORCE agent (discrete actions)
- ActorCriticNetwork: Neural network for A2C and PPO agents (discrete actions)
- REINFORCEAgent: Vanilla policy gradient (updates at end of episode)
- A2CAgent: Actor-Critic with N-step updates (updates during episode)
- PPOAgent: Proximal Policy Optimization with clipped objective (mini-batch updates)

Action Space: Discrete, 12 actions representing directions at 30° intervals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .utils import NUM_ACTIONS

device = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):
    """Neural network policy for discrete action selection.

    Outputs logits over NUM_ACTIONS discrete actions.
    Architecture: 3 hidden layers × 192 units.
    """

    def __init__(self, state_dim, num_actions=NUM_ACTIONS, hidden_dim=192):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """Forward pass returning action logits."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.action_head(x)
        return logits

    def get_action(self, x, deterministic=False):
        """Sample an action from the policy.

        Args:
            x: State tensor.
            deterministic: If True, return argmax action without sampling.

        Returns:
            action: Sampled action index (0 to NUM_ACTIONS-1).
            log_prob: Log probability of the action.
            probs: Action probabilities.
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, probs


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared backbone for discrete actions.

    Architecture: 3 hidden layers × 192 units (shallower but wider than before).
    This reduces sequential latency while maintaining capacity.
    """

    def __init__(self, state_dim, num_actions=NUM_ACTIONS, hidden_dim=192):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass returning action logits and value."""
        shared = self.shared(x)
        logits = self.action_head(shared)
        value = self.value_head(shared)
        return logits, value

    def get_action(self, x, deterministic=False):
        """Sample an action from the policy."""
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value, probs

    def get_value(self, x):
        """Get only state value."""
        shared = self.shared(x)
        return self.value_head(shared)

    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and entropy for given state-action pairs."""
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


class REINFORCEAgent:
    """REINFORCE policy gradient agent with discrete actions.

    Updates policy only at the end of each episode using full returns.
    """

    def __init__(self, state_dim, learning_rate=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, return_probs=False, deterministic=False):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, probs = self.policy.get_action(state_tensor, deterministic)
        self.saved_log_probs.append(log_prob)
        action_idx = action.item()

        if return_probs:
            return action_idx, {
                "probs": probs.squeeze().detach().cpu().numpy().tolist(),
                "action": action_idx,
            }
        return action_idx

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.rewards) == 0:
            return 0.0

        returns = self.compute_returns(self.rewards)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

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
        checkpoint = torch.load(filepath, map_location=device)

        # Check for dimension mismatch
        saved_state = checkpoint["policy_state_dict"]
        saved_input_dim = saved_state["fc1.weight"].shape[1]
        current_input_dim = self.policy.fc1.weight.shape[1]

        if saved_input_dim != current_input_dim:
            raise RuntimeError(
                f"Model dimension mismatch: saved model has input_dim={saved_input_dim}, "
                f"but current model expects input_dim={current_input_dim}. "
                f"The observation space has changed since this model was trained. "
                f"Please re-collect trajectories and retrain the model."
            )

        self.policy.load_state_dict(saved_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.to(device)
        print(f"Model loaded from {filepath}")


class A2CAgent:
    """Actor-Critic agent with N-step updates for discrete actions.

    Updates policy every N steps using TD-style advantages, enabling
    learning during long episodes without waiting for episode end.
    """

    def __init__(
        self,
        state_dim,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=64,
        value_loss_coef=0.5,
        entropy_coef=0.05,  # Increased from 0.01 to prevent entropy collapse
        max_grad_norm=0.5,
    ):
        self.network = ActorCriticNetwork(state_dim, hidden_dim=192).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.total_updates = 0

    def select_action(self, state, return_probs=False, deterministic=False):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, value, probs = self.network.get_action(
                state_tensor, deterministic
            )

        action_idx = action.item()
        self.states.append(state)
        self.actions.append(action_idx)
        self.values.append(value.squeeze().item())
        self.log_probs.append(log_prob.item())

        if return_probs:
            return action_idx, {
                "probs": probs.squeeze().detach().cpu().numpy().tolist(),
                "action": action_idx,
            }
        return action_idx

    def store_reward(self, reward, done=False):
        """Store reward and done flag from environment step."""
        self.rewards.append(reward)
        self.dones.append(done)

    def should_update(self):
        """Check if we have enough steps for an update."""
        return len(self.rewards) >= self.n_steps

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + self.gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update_policy(self, next_state=None):
        """Update policy using collected rollout."""
        if len(self.rewards) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = (
                    torch.FloatTensor(next_state).unsqueeze(0).to(device)
                )
                next_value = self.network.get_value(next_state_tensor).squeeze().item()
        else:
            next_value = 0.0

        advantages = self.compute_gae(next_value)
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        states_tensor = torch.FloatTensor(np.array(self.states)).to(device)
        actions_tensor = torch.LongTensor(self.actions).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)

        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        log_probs, entropy, values = self.network.evaluate_actions(
            states_tensor, actions_tensor
        )
        # Fix shape mismatch: ensure both tensors have same shape
        values = values.squeeze(-1)  # Remove last dim but keep batch dim
        if values.dim() == 0:
            values = values.unsqueeze(0)  # Handle single-element case
        entropy = entropy.mean()

        policy_loss = -(log_probs * advantages_tensor).mean()
        value_loss = F.mse_loss(values, returns_tensor)
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.total_updates += 1

        self.clear_buffer()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }

    def clear_buffer(self):
        """Clear the rollout buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def save_model(self, filepath):
        """Save model weights to file."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_updates": self.total_updates,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights from file."""
        checkpoint = torch.load(filepath, map_location=device)

        # Check for dimension mismatch
        saved_state = checkpoint["network_state_dict"]
        saved_input_dim = saved_state["shared.0.weight"].shape[1]
        current_input_dim = self.network.shared[0].weight.shape[1]

        if saved_input_dim != current_input_dim:
            raise RuntimeError(
                f"Model dimension mismatch: saved model has input_dim={saved_input_dim}, "
                f"but current model expects input_dim={current_input_dim}. "
                f"The observation space has changed since this model was trained. "
                f"Please re-collect trajectories and retrain the model."
            )

        self.network.load_state_dict(saved_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
        self.network.to(device)
        print(f"Model loaded from {filepath}")

    def load_from_policy_network(self, filepath):
        """Load weights from a PolicyNetwork checkpoint (from REINFORCE or pretraining)."""
        checkpoint = torch.load(filepath, map_location=device)
        policy_state = checkpoint["policy_state_dict"]

        # Check for dimension mismatch
        saved_input_dim = policy_state["fc1.weight"].shape[1]
        current_input_dim = self.network.shared[0].weight.shape[1]

        if saved_input_dim != current_input_dim:
            raise RuntimeError(
                f"Model dimension mismatch: saved model has input_dim={saved_input_dim}, "
                f"but current model expects input_dim={current_input_dim}. "
                f"The observation space has changed since this model was trained. "
                f"Please re-collect trajectories and retrain the model."
            )

        new_state = {
            "shared.0.weight": policy_state["fc1.weight"],
            "shared.0.bias": policy_state["fc1.bias"],
            "shared.2.weight": policy_state["fc2.weight"],
            "shared.2.bias": policy_state["fc2.bias"],
            "shared.4.weight": policy_state["fc3.weight"],
            "shared.4.bias": policy_state["fc3.bias"],
            "action_head.weight": policy_state["action_head.weight"],
            "action_head.bias": policy_state["action_head.bias"],
        }

        self.network.load_state_dict(new_state, strict=False)
        self.network.to(device)
        print(f"Loaded policy weights from {filepath} (value head initialized fresh)")


class PPOAgent:
    """Proximal Policy Optimization agent with discrete actions.

    Updates policy using clipped surrogate objective over multiple epochs.
    More stable than A2C and REINFORCE for complex environments.
    """

    def __init__(
        self,
        state_dim,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.network = ActorCriticNetwork(state_dim, hidden_dim=192).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.total_updates = 0

    def select_action(self, state, return_probs=False, deterministic=False):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, value, probs = self.network.get_action(
                state_tensor, deterministic
            )

        action_idx = action.item()
        self.states.append(state)
        self.actions.append(action_idx)
        self.values.append(value.squeeze().item())
        self.log_probs.append(log_prob.item())

        if return_probs:
            return action_idx, {
                "probs": probs.squeeze().detach().cpu().numpy().tolist(),
                "action": action_idx,
            }
        return action_idx

    def store_reward(self, reward, done=False):
        """Store reward and done flag from environment step."""
        self.rewards.append(reward)
        self.dones.append(done)

    def should_update(self):
        """Check if we have enough steps for an update."""
        return len(self.rewards) >= self.n_steps

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + self.gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update_policy(self, next_state=None):
        """Update policy using collected rollout buffer."""
        if len(self.rewards) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = (
                    torch.FloatTensor(next_state).unsqueeze(0).to(device)
                )
                next_value = self.network.get_value(next_state_tensor).squeeze().item()
        else:
            next_value = 0.0

        advantages = self.compute_gae(next_value)
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_entropy = 0
        epoch_total_loss = 0

        # PPO Multi-epoch update
        dataset_size = len(self.rewards)
        indices = np.arange(dataset_size)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                new_log_probs, entropy, values = self.network.evaluate_actions(
                    batch_states, batch_actions
                )
                values = values.squeeze()

                # Ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy_loss.item()
                epoch_total_loss += total_loss.item()

        num_updates = self.n_epochs * (dataset_size // self.batch_size + 1)
        self.total_updates += 1
        self.clear_buffer()

        return {
            "policy_loss": epoch_policy_loss / num_updates,
            "value_loss": epoch_value_loss / num_updates,
            "entropy": epoch_entropy / num_updates,
            "total_loss": epoch_total_loss / num_updates,
        }

    def clear_buffer(self):
        """Clear the rollout buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def save_model(self, filepath):
        """Save model weights to file."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_updates": self.total_updates,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights from file."""
        checkpoint = torch.load(filepath, map_location=device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
        print(f"Model loaded from {filepath}")
