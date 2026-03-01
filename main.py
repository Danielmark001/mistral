import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import NOVAConfig
from utils.logging import get_logger

logger = get_logger("nova")


def run_collect(cfg: NOVAConfig):
    from models.planner import NOVAPlanner
    from training.dataset_builder import collect_trajectories, build_dataset

    logger.info("Loading planner for data collection...")
    planner = NOVAPlanner(model_name=cfg.model_name)

    logger.info(f"Collecting {cfg.episodes_per_gen} episodes...")
    trajectories = collect_trajectories(
        planner=planner,
        n_episodes=cfg.episodes_per_gen,
        env_id=cfg.env_id,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
    )

    out_path = Path(cfg.output_dir) / "initial_data.jsonl"
    n = build_dataset(trajectories, str(out_path), filter_successful=False)
    logger.info(f"Collected {n} samples -> {out_path}")


def run_train(cfg: NOVAConfig):
    from models.planner import NOVAPlanner
    from training.train_planner import train_planner

    data_path = Path(cfg.output_dir) / "initial_data.jsonl"
    model_path = Path(cfg.output_dir) / "model"

    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Run --mode collect first.")
        sys.exit(1)

    logger.info("Loading planner for training...")
    planner = NOVAPlanner(model_name=cfg.model_name)

    train_planner(
        planner=planner,
        train_path=str(data_path),
        output_dir=str(model_path),
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        generation=0,
        wandb_project=cfg.wandb_project,
    )


def run_self_improve(cfg: NOVAConfig):
    from training.self_improve import self_improve_loop

    self_improve_loop(
        model_name=cfg.model_name,
        env_id=cfg.env_id,
        n_generations=cfg.n_generations,
        episodes_per_gen=cfg.episodes_per_gen,
        eval_episodes=cfg.eval_episodes,
        num_epochs=cfg.num_epochs,
        output_base=cfg.output_dir,
        wandb_project=cfg.wandb_project,
    )


def run_evaluate(cfg: NOVAConfig):
    from models.planner import NOVAPlanner
    from evaluation.evaluate import evaluate_planner

    model_path = Path(cfg.output_dir) / "model"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Train first.")
        sys.exit(1)

    logger.info("Loading trained planner for evaluation...")
    planner = NOVAPlanner.load(str(model_path), base_model_name=cfg.model_name)

    metrics = evaluate_planner(
        planner=planner,
        env_id=cfg.env_id,
        n_episodes=cfg.eval_episodes,
        seed=cfg.seed + 9999,
        wandb_log=True,
        wandb_project=cfg.wandb_project,
    )

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="NOVA Agent")
    parser.add_argument("--mode", choices=["collect", "train", "self_improve", "evaluate"], required=True)
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--env", type=str, default=None, help="Override env ID")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = NOVAConfig.load(args.config)
    else:
        cfg = NOVAConfig()

    if args.model:
        cfg.model_name = args.model
    if args.env:
        cfg.env_id = args.env
    if args.generations:
        cfg.n_generations = args.generations
    if args.episodes:
        cfg.episodes_per_gen = args.episodes
    if args.output:
        cfg.output_dir = args.output
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project

    modes = {
        "collect": run_collect,
        "train": run_train,
        "self_improve": run_self_improve,
        "evaluate": run_evaluate,
    }
    modes[args.mode](cfg)


if __name__ == "__main__":
    main()
