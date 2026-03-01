import json
from pathlib import Path
import numpy as np


class StepwiseReward:
    """Rule-based reward: success bonus, step penalty, timeout penalty."""

    def __init__(self, success_reward=1.0, step_penalty=-0.01, timeout_penalty=-1.0):
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.timeout_penalty = timeout_penalty

    def compute(self, state: dict, env_reward: float, done: bool, truncated: bool = False) -> float:
        if state.get("success"):
            return self.success_reward
        if truncated or (done and not state.get("success")):
            return self.timeout_penalty
        return self.step_penalty


class TrajectoryScorer:
    """Score full trajectories for filtering in the self-improvement loop."""

    def score(self, trajectory: list[dict]) -> float:
        total_reward = sum(t["reward"] for t in trajectory)
        success = any(t["state"].get("success", False) for t in trajectory)
        steps = len(trajectory)
        if success:
            return 1.0 + (1.0 / max(steps, 1))
        return total_reward / max(steps, 1)

    def filter_successful(self, trajectories: list[list[dict]], threshold=0.0) -> list[list[dict]]:
        return [t for t in trajectories if self.score(t) > threshold]
