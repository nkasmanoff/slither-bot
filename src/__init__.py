"""Slither.io Bot - RL agents for playing Slither.io.

This package provides tools for:
- Controlling the Slither.io game via Selenium
- Training RL agents (REINFORCE, A2C)
- Collecting human/auto trajectories for imitation learning
- Comparing different agent strategies
"""

from .agents import A2CAgent, ActorCriticNetwork, PolicyNetwork, REINFORCEAgent
from .collect import collect_human_trajectories, collect_single_episode
from .compare import run_comparison
from .controller import SlitherController
from .environment import (
    SlitherEnv,
    load_and_play,
    setup_browser_and_game,
    train_agent,
    train_agent_a2c,
)
from .pretrain import collect_auto_trajectories, train_supervised
from .profile import Profiler, profile_training_loop, quick_profile_selenium
from .rules_policy import run_rules_based_policy
from .utils import (
    MAX_TURN_RATE,
    OBSERVATION_DIM,
    angle_to_continuous_action,
    apply_turn_to_angle,
    continuous_action_to_angle,
    continuous_action_to_turn,
    extract_observation,
    setup_browser,
    start_game,
    wait_for_game_ready,
)

__all__ = [
    # Agents
    "PolicyNetwork",
    "ActorCriticNetwork",
    "REINFORCEAgent",
    "A2CAgent",
    # Controller
    "SlitherController",
    # Environment
    "SlitherEnv",
    "setup_browser_and_game",
    "train_agent",
    "train_agent_a2c",
    "load_and_play",
    # Rules policy
    "run_rules_based_policy",
    # Collection
    "collect_human_trajectories",
    "collect_single_episode",
    "collect_auto_trajectories",
    # Pretraining
    "train_supervised",
    # Comparison
    "run_comparison",
    # Profiling
    "Profiler",
    "quick_profile_selenium",
    "profile_training_loop",
    # Utilities
    "setup_browser",
    "start_game",
    "wait_for_game_ready",
    "extract_observation",
    "angle_to_continuous_action",
    "continuous_action_to_angle",
    "continuous_action_to_turn",
    "apply_turn_to_angle",
    "OBSERVATION_DIM",
    "MAX_TURN_RATE",
]
