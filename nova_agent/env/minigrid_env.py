import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper


ACTION_NAMES = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

ACTION_FROM_NAME = {v: k for k, v in ACTION_NAMES.items()}


class NOVAEnv:
    def __init__(self, env_id="MiniGrid-Empty-8x8-v0", max_steps=50, seed=42):
        self.env_id = env_id
        self.max_steps = max_steps
        self.seed = seed
        self._env = FullyObsWrapper(
            gym.make(env_id, max_steps=max_steps, render_mode=None)
        )
        self.step_count = 0

    def reset(self):
        obs, info = self._env.reset(seed=self.seed)
        self.step_count = 0
        return self._parse_obs(obs), info

    def step(self, action):
        if isinstance(action, str):
            action = ACTION_FROM_NAME.get(action, 2)
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.step_count += 1
        done = terminated or truncated
        structured = self._parse_obs(obs)
        structured["done"] = done
        structured["success"] = terminated and reward > 0
        return structured, reward, done, info

    def _parse_obs(self, obs):
        env = self._env.unwrapped
        return {
            "agent_pos": tuple(env.agent_pos),
            "agent_dir": env.agent_dir,
            "goal_pos": self._find_goal(env),
            "visible_objects": self._get_visible_objects(env),
            "step": self.step_count,
            "max_steps": self.max_steps,
        }

    def _find_goal(self, env):
        for y in range(env.grid.height):
            for x in range(env.grid.width):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return (x, y)
        return None

    def _get_visible_objects(self, env):
        objects = []
        for y in range(env.grid.height):
            for x in range(env.grid.width):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type not in ("empty", "goal"):
                    objects.append({
                        "type": cell.type,
                        "pos": (x, y),
                        "color": getattr(cell, "color", None),
                    })
        return objects

    @property
    def action_space(self):
        return list(ACTION_NAMES.values())

    def close(self):
        self._env.close()
