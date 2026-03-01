import json
from pathlib import Path
from dataclasses import dataclass, asdict, field


@dataclass
class NOVAConfig:
    model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    env_id: str = "MiniGrid-Empty-8x8-v0"
    max_steps: int = 50
    n_generations: int = 3
    episodes_per_gen: int = 100
    eval_episodes: int = 20
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    output_dir: str = "outputs/nova"
    wandb_project: str = "nova-planner"
    seed: int = 42

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
