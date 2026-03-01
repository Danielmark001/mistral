import wandb
from pathlib import Path

from models.planner import NOVAPlanner
from training.dataset_builder import collect_trajectories, build_dataset
from training.train_planner import train_planner
from evaluation.evaluate import evaluate_planner
from models.reward_model import TrajectoryScorer


def self_improve_loop(
    model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    env_id: str = "MiniGrid-Empty-8x8-v0",
    n_generations: int = 3,
    episodes_per_gen: int = 100,
    eval_episodes: int = 20,
    num_epochs: int = 3,
    output_base: str = "outputs/nova",
    wandb_project: str = "nova-planner",
):
    print(f"Starting NOVA self-improvement: {n_generations} generations")
    planner = NOVAPlanner(model_name=model_name)
    scorer = TrajectoryScorer()

    for gen in range(n_generations):
        gen_num = gen + 1
        print(f"\n--- Generation {gen_num} ---")
        gen_dir = Path(output_base) / f"gen_{gen_num}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        data_path = gen_dir / "train_data.jsonl"
        model_path = gen_dir / "model"

        # one W&B run for the entire generation
        run = wandb.init(
            project=wandb_project,
            name=f"nova-gen-{gen_num}",
            tags=[f"gen-{gen_num}"],
            config={
                "generation": gen_num,
                "model": model_name,
                "env": env_id,
                "episodes": episodes_per_gen,
                "eval_episodes": eval_episodes,
                "num_epochs": num_epochs,
            },
        )

        # --- collect ---
        print(f"Collecting {episodes_per_gen} episodes...")
        trajectories = collect_trajectories(
            planner=planner,
            n_episodes=episodes_per_gen,
            env_id=env_id,
            seed=gen * 1000,
        )

        successful = scorer.filter_successful(trajectories, threshold=0.0)
        collect_success_rate = len(successful) / max(len(trajectories), 1)

        wandb.log({
            "collect/total_episodes": len(trajectories),
            "collect/successful_episodes": len(successful),
            "collect/success_rate": collect_success_rate,
            "generation": gen_num,
        })

        n_samples = build_dataset(
            trajectories=trajectories,
            output_path=str(data_path),
            filter_successful=True,
        )
        wandb.log({"collect/dataset_samples": n_samples, "generation": gen_num})
        print(f"Dataset built: {n_samples} samples ({collect_success_rate:.1%} success rate)")

        if n_samples == 0:
            print("No successful trajectories — skipping training this generation.")
            wandb.finish()
            continue

        # --- train (logs to current run via report_to=wandb) ---
        print("Training planner...")
        train_planner(
            planner=planner,
            train_path=str(data_path),
            output_dir=str(model_path),
            num_epochs=num_epochs,
            generation=gen_num,
            wandb_project=wandb_project,
            manage_wandb=False,
        )

        # --- evaluate (logs to current run) ---
        print("Evaluating...")
        metrics = evaluate_planner(
            planner=planner,
            env_id=env_id,
            n_episodes=eval_episodes,
            seed=gen * 1000 + 500,
            generation=gen_num,
        )

        print(f"Gen {gen_num} | success_rate={metrics['success_rate']:.2f} | avg_steps={metrics['avg_steps']:.1f}")
        wandb.finish()

    print("\nSelf-improvement complete.")
