"""Profiling utilities for the training loop.

Provides tools to measure Selenium latency and identify performance bottlenecks.
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from .controller import SlitherController


class Profiler:
    """Simple profiler to track function timing."""

    def __init__(self):
        self.times = defaultdict(list)
        self.call_counts = defaultdict(int)
        self._start_times = {}

    @contextmanager
    def track(self, name):
        """Context manager to track time for a named section."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name].append(elapsed)
            self.call_counts[name] += 1

    def start(self, name):
        """Start timing a named section."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name):
        """Stop timing a named section."""
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name].append(elapsed)
            self.call_counts[name] += 1
            del self._start_times[name]

    def report(self):
        """Print a summary report of all tracked timings."""
        print("\n" + "=" * 70)
        print("PROFILING REPORT")
        print("=" * 70)

        total_time = sum(sum(times) for times in self.times.values())
        sorted_items = sorted(self.times.items(), key=lambda x: sum(x[1]), reverse=True)

        print(f"\n{'Function':<35} {'Total':>10} {'Calls':>8} {'Avg':>10} {'%':>7}")
        print("-" * 70)

        for name, times in sorted_items:
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            print(
                f"{name:<35} {total:>9.3f}s {count:>8} {avg*1000:>9.2f}ms {pct:>6.1f}%"
            )

        print("-" * 70)
        print(f"{'TOTAL':<35} {total_time:>9.3f}s")
        print("=" * 70)

        return self.times


def quick_profile_selenium(driver, num_calls=10):
    """Profile Selenium call latency.

    Args:
        driver: Selenium WebDriver instance.
        num_calls: Number of times to call each function.

    Returns:
        Dictionary of timing results.
    """
    controller = SlitherController(driver, save_screenshots=False, record_video=False)
    times = {}

    # Profile batched state call
    durations = []
    for _ in range(num_calls):
        start = time.perf_counter()
        controller.get_full_state()
        durations.append(time.perf_counter() - start)
    times["get_full_state [BATCHED]"] = durations

    # Profile individual calls for comparison
    durations = []
    for _ in range(num_calls):
        start = time.perf_counter()
        controller.get_detailed_state()
        durations.append(time.perf_counter() - start)
    times["get_detailed_state"] = durations

    durations = []
    for _ in range(num_calls):
        start = time.perf_counter()
        controller.get_snake_length()
        durations.append(time.perf_counter() - start)
    times["get_snake_length"] = durations

    durations = []
    for _ in range(num_calls):
        start = time.perf_counter()
        controller.is_game_over()
        durations.append(time.perf_counter() - start)
    times["is_game_over"] = durations

    durations = []
    for i in range(num_calls):
        start = time.perf_counter()
        controller.move_to_angle(i * 45)
        durations.append(time.perf_counter() - start)
    times["move_to_angle"] = durations

    durations = []
    for _ in range(num_calls):
        start = time.perf_counter()
        driver.get_screenshot_as_png()
        durations.append(time.perf_counter() - start)
    times["get_screenshot_as_png"] = durations

    # Report
    print("\n" + "=" * 70)
    print("SELENIUM LATENCY PROFILE")
    print("=" * 70)
    print(f"\n{'Function':<35} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for name, durations in times.items():
        avg = np.mean(durations) * 1000
        min_t = np.min(durations) * 1000
        max_t = np.max(durations) * 1000
        print(f"{name:<35} {avg:>9.1f}ms {min_t:>9.1f}ms {max_t:>9.1f}ms")

    print("-" * 70)

    # Calculate savings
    old_total = (
        np.mean(times["get_detailed_state"])
        + np.mean(times["get_snake_length"])
        + np.mean(times["is_game_over"])
    ) * 1000
    new_total = np.mean(times["get_full_state [BATCHED]"]) * 1000

    print(f"\nOLD (3 separate calls): {old_total:.1f}ms")
    print(f"NEW (1 batched call):   {new_total:.1f}ms")
    print(
        f"SAVINGS:                {old_total - new_total:.1f}ms ({(1 - new_total/old_total)*100:.0f}%)"
    )
    print("=" * 70)

    return times


def profile_training_loop(num_steps=50):
    """Run a profiled training session.

    Args:
        num_steps: Number of steps to profile.
    """
    from .agents import A2CAgent
    from .environment import setup_browser_and_game

    profiler = Profiler()

    print(f"Starting profiled training for {num_steps} steps...")

    with profiler.track("setup_browser_and_game"):
        driver, env = setup_browser_and_game(record_video=False)

    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        n_steps=16,
    )

    with profiler.track("env.reset"):
        state = env.reset()

    steps = 0
    done = False

    print(f"Running {num_steps} steps with profiling...\n")

    while steps < num_steps:
        with profiler.track("agent.select_action"):
            action, probs = agent.select_action(state, return_probs=True)

        with profiler.track("env.step"):
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )

        with profiler.track("agent.store_reward"):
            agent.store_reward(reward, done=done)

        if agent.should_update():
            with profiler.track("agent.update_policy"):
                agent.update_policy(next_state=next_state if not done else None)

        state = next_state
        steps += 1

        if steps % 10 == 0:
            print(f"Step {steps}/{num_steps}")

        if done:
            with profiler.track("env.reset"):
                state = env.reset()
            done = False

    profiler.report()
    driver.quit()

    print("\nProfiling complete!")
