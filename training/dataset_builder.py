import json
from pathlib import Path
from typing import Optional

from env.minigrid_env import NOVAEnv
from env.state_serializer import serialize_state
from models.reward_model import StepwiseReward, TrajectoryScorer


def collect_trajectories(
    planner,
    n_episodes: int = 100,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    max_steps: int = 50,
    seed: int = 42,
) -> list[list[dict]]:
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

            reward = rewarder.compute(
                state=next_state,
                env_reward=env_reward,
                done=done,
                truncated=next_state.get("step", 0) >= max_steps,
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
    trajectories: list[list[dict]],
    output_path: str,
    filter_successful: bool = True,
) -> int:
    scorer = TrajectoryScorer()

    if filter_successful:
        trajectories = scorer.filter_successful(trajectories, threshold=0.0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w") as f:
        for traj in trajectories:
            for step in traj:
                record = {
                    "input": step["serialized"],
                    "output": step["action"],
                    "reward": step["reward"],
                }
                f.write(json.dumps(record) + "\n")
                count += 1

    print(f"Saved {count} samples to {output_path}")
    return count


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
