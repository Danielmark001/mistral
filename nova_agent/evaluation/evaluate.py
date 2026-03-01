import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import weave
import wandb
from env.minigrid_env import NOVAEnv
from env.state_serializer import serialize_state


@weave.op()
def _agent_step(planner, serialized_state: str) -> str:
    return planner.predict(serialized_state)


def evaluate_planner(
    planner,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    n_episodes: int = 20,
    seed: int = 9999,
    generation: int = None,
    wandb_log: bool = False,
    wandb_project: str = "nova-planner",
) -> dict:
    if n_episodes <= 0:
        return {"success_rate": 0.0, "avg_steps": 0.0, "min_steps": 0, "max_steps": 0, "action_distribution": {}}

    successes = 0
    total_steps = []
    action_counts = {}

    for ep in range(n_episodes):
        env = NOVAEnv(env_id=env_id, seed=seed + ep)
        state, _ = env.reset()
        steps = 0

        while True:
            text = serialize_state(state)
            action = _agent_step(planner, text)
            action_counts[action] = action_counts.get(action, 0) + 1
            state, _, done, _ = env.step(action)
            steps += 1
            if done:
                if state.get("success"):
                    successes += 1
                break

        total_steps.append(steps)
        env.close()

    metrics = {
        "success_rate": successes / n_episodes,
        "avg_steps": sum(total_steps) / len(total_steps),
        "min_steps": min(total_steps),
        "max_steps": max(total_steps),
        "action_distribution": action_counts,
    }

    run_was_none = wandb.run is None
    if wandb_log and run_was_none:
        run_name = f"eval-gen-{generation}" if generation is not None else "eval"
        wandb.init(project=wandb_project, name=run_name, tags=["eval"])

    if wandb.run is not None:
        payload = {
            "eval/success_rate": metrics["success_rate"],
            "eval/avg_steps": metrics["avg_steps"],
            "eval/min_steps": metrics["min_steps"],
            "eval/max_steps": metrics["max_steps"],
        }
        if generation is not None:
            payload["generation"] = generation

        action_table = wandb.Table(
            columns=["action", "count"],
            data=[[a, c] for a, c in action_counts.items()],
        )
        payload["eval/action_distribution"] = wandb.plot.bar(
            action_table, "action", "count", title="Action Distribution"
        )
        wandb.log(payload)

        if wandb_log and run_was_none:
            wandb.finish()

    return metrics
