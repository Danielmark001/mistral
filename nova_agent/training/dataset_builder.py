import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.minigrid_env import NOVAEnv
from env.state_serializer import serialize_state
from models.reward_model import StepwiseReward, TrajectoryScorer


def collect_trajectories(
    planner,
    n_episodes: int = 100,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    max_steps: int = 50,
    seed: int = 42,
) -> list:
    rewarder = StepwiseReward()
    trajectories = []

    for ep in range(n_episodes):
        env = NOVAEnv(env_id=env_id, max_steps=max_steps, seed=seed + ep)
        state, _ = env.reset()
        traj = []

        while True:
            text = serialize_state(state)
            action = planner.predict(text)
            next_state, env_reward, done, info = env.step(action)
            truncated = next_state.get("step", 0) >= max_steps and not next_state.get("success", False)

            reward = rewarder.compute(
                state=next_state,
                env_reward=env_reward,
                done=done,
                truncated=truncated,
            )
            traj.append({
                "state": next_state,
                "serialized": text,
                "action": action,
                "reward": reward,
            })
            state = next_state
            if done:
                break

        env.close()
        trajectories.append(traj)
        if (ep + 1) % 10 == 0:
            print(f"  collected {ep + 1}/{n_episodes} episodes")

    return trajectories


def build_dataset(
    trajectories: list,
    output_path: str,
    filter_successful: bool = True,
) -> int:
    if filter_successful:
        trajectories = TrajectoryScorer().filter_successful(trajectories, threshold=0.0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for traj in trajectories:
            for step in traj:
                f.write(json.dumps({
                    "input": step["serialized"],
                    "output": step["action"],
                    "reward": step["reward"],
                }) + "\n")
                count += 1

    print(f"Saved {count} samples to {output_path}")
    return count


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
