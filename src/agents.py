"""RL Agents for Slither.io.

Contains:
- PolicyNetwork: Neural network for REINFORCE agent
- ActorCriticNetwork: Neural network for A2C agent
- REINFORCEAgent: Vanilla policy gradient (updates at end of episode)
- A2CAgent: Actor-Critic with N-step updates (updates during episode)

Action Space: Continuous, single value in [-1, 1] mapped to rotation angle [0, 360].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = "cuda" if torch.cuda.is_available() else "cpu"

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class PolicyNetwork(nn.Module):
    """Neural network policy for continuous action selection.

    Outputs a Gaussian distribution over actions in [-1, 1].
    """

    def __init__(self, state_dim, action_dim=1, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Forward pass returning mean and std for the action distribution."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = torch.tanh(self.mean_head(x))
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, x, deterministic=False):
        """Sample an action from the policy.

        Args:
            x: State tensor.
            deterministic: If True, return mean action without sampling.

        Returns:
            action: Sampled action in [-1, 1].
            log_prob: Log probability of the action.
            mean: Mean of the distribution.
            std: Standard deviation.
        """
        mean, std = self.forward(x)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean, std


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared backbone for continuous actions."""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass returning action distribution params and value."""
        shared = self.shared(x)
        mean = torch.tanh(self.mean_head(shared))
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        value = self.value_head(shared)
        return mean, std, value

    def get_action(self, x, deterministic=False):
        """Sample an action from the policy."""
        mean, std, value = self.forward(x)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, mean, std

    def get_value(self, x):
        """Get only state value."""
        shared = self.shared(x)
        return self.value_head(shared)

    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and entropy for given state-action pairs."""
        mean, std, values = self.forward(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, entropy, values


class REINFORCEAgent:
    """REINFORCE policy gradient agent with continuous actions.

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
        action, log_prob, mean, std = self.policy.get_action(
            state_tensor, deterministic
        )
        self.saved_log_probs.append(log_prob)
        action_value = action.squeeze().item()

        if return_probs:
            return action_value, {
                "mean": mean.squeeze().item(),
                "std": std.squeeze().item(),
                "action": action_value,
            }
        return action_value

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
    """Actor-Critic agent with N-step updates for continuous actions.

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
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.network = ActorCriticNetwork(state_dim).to(device)
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
            action, log_prob, value, mean, std = self.network.get_action(
                state_tensor, deterministic
            )

        action_value = action.squeeze().item()
        self.states.append(state)
        self.actions.append(action_value)
        self.values.append(value.squeeze().item())
        self.log_probs.append(log_prob.item())

        if return_probs:
            return action_value, {
                "mean": mean.squeeze().item(),
                "std": std.squeeze().item(),
                "action": action_value,
            }
        return action_value

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
        actions_tensor = torch.FloatTensor(self.actions).unsqueeze(-1).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)

        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        log_probs, entropy, values = self.network.evaluate_actions(
            states_tensor, actions_tensor
        )
        values = values.squeeze()

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
            "mean_head.weight": policy_state["mean_head.weight"],
            "mean_head.bias": policy_state["mean_head.bias"],
            "log_std": policy_state["log_std"],
        }

        self.network.load_state_dict(new_state, strict=False)
        self.network.to(device)
        print(f"Loaded policy weights from {filepath} (value head initialized fresh)")
