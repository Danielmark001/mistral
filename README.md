# NOVA — Navigable Open-World Virtual Agent

A self-improving agent that fine-tunes Mistral Small 3.1 on structured navigation tasks using LoRA, with full Weights & Biases tracking across generations.

Built for the **Mistral AI Worldwide Hackathon 2026** — Fine-tuning by Weights & Biases track.

## How it works

1. The environment (MiniGrid) produces a structured state — agent position, goal, obstacles
2. The state is serialized into plain text and fed into Mistral Small 3.1
3. The model predicts one action from a discrete vocabulary
4. Successful trajectories are filtered and used to fine-tune the model via LoRA
5. This loop repeats across N generations, with every metric tracked in W&B

## Setup

```bash
pip install -r requirements.txt
wandb login
```

Requires 4x A100 GPUs (or equivalent) for full float16 inference + LoRA training.

## Usage

```bash
# collect initial trajectories
python main.py --mode collect

# train on collected data
python main.py --mode train

# self-improvement loop (collect → train → eval, N generations)
python main.py --mode self_improve --generations 3

# evaluate a trained model
python main.py --mode evaluate
```

Override defaults inline:

```bash
python main.py --mode self_improve \
  --generations 5 \
  --episodes 200 \
  --env MiniGrid-FourRooms-v0 \
  --wandb-project my-nova-run
```

## Project structure

```
nova_agent/
├── env/
│   ├── minigrid_env.py       # MiniGrid wrapper with structured state
│   └── state_serializer.py   # converts state dict to text prompt
├── models/
│   ├── planner.py            # Mistral Small 3.1 + LoRA, action prediction
│   ├── reward_model.py       # step penalty, success bonus, trajectory scoring
│   └── lora_config.py        # LoRA config (r=16, q_proj + v_proj)
├── training/
│   ├── dataset_builder.py    # episode collection, JSONL export
│   ├── train_planner.py      # HuggingFace Trainer + W&B
│   └── self_improve.py       # N-generation self-improvement loop
├── evaluation/
│   └── evaluate.py           # success rate, step stats, action distribution
├── utils/
│   ├── config.py             # NOVAConfig dataclass
│   └── logging.py            # stdout logger
└── main.py                   # CLI entry point
```

## W&B tracking

One run per generation logs:

| Metric | Description |
|---|---|
| `collect/success_rate` | fraction of successful episodes during collection |
| `collect/dataset_samples` | samples written to JSONL after filtering |
| `train/loss` | cross-entropy on action token, every 10 steps |
| `eval/success_rate` | post-training success rate |
| `eval/avg_steps` | average steps to goal |
| `eval/action_distribution` | bar chart of action counts |

## Model

`mistralai/Mistral-Small-3.1-24B-Instruct-2503` — Apache 2.0

## License

Apache 2.0
