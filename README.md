# NOVA — Navigable Open-World Virtual Agent

**Mistral AI Worldwide Hackathon 2026 — Fine-Tuning by Weights & Biases Track**

Fine-tuning Mistral Small 3.1 (24B) with LoRA across 3 self-improvement generations on structured navigation tasks. Success rate improves from **4% (zero-shot baseline) to 84%** after generation 3. Average episode length drops from **49.1 to 14.6 steps**.

---

## W&B Report

Full training report with loss curves, eval metrics, action distributions, ablations, and failure analysis:

**https://wandb.ai/leadgen12344-nanyang-technological-university-singapore/nova-planner/reports/NOVA:-Self-Improving-Navigation-Agent-via-LoRA-Fine-Tuning--VmlldzoxNjA2ODk1MQ==**

W&B Project (all runs + artifacts):

**https://wandb.ai/leadgen12344-nanyang-technological-university-singapore/nova-planner**

---

## Results

| | Baseline | Gen 1 | Gen 2 | Gen 3 |
|---|---|---|---|---|
| Success rate | 4% | 31% | 58% | **84%** |
| Avg steps to goal | 49.1 | 37.8 | 27.2 | **14.6** |
| Token accuracy | — | 61% | 79% | **93%** |
| Dataset samples | 0 | 312 | 694 | 1,189 |
| Collection success | 0% | 18% | 43% | 71% |

---

## W&B Track Checklist

This submission addresses every judging criterion for the Fine-Tuning by Weights & Biases track:

### Technical Quality
- Fine-tuning is non-trivial: zero-shot Mistral Small 3.1 achieves only 4% — prompt engineering is insufficient
- Full pipeline: data collection, trajectory filtering, LoRA fine-tuning, evaluation, self-improvement loop
- Validated ablations: LoRA rank (r=16 vs r=8), target modules, trajectory filtering, chat template formatting
- Failure mode analysis documented in the W&B report (section 6)

### Experiment Tracking — W&B Models
Every run logs the following to W&B:

**Training metrics (per step):**
- `train/loss` — cross-entropy on action token, smoothed
- `train/perplexity` — exp(loss)
- `train/token_accuracy` — fraction of action tokens predicted correctly
- `train/grad_norm` — gradient norm with clipping events visible
- `train/learning_rate` — linear warmup + cosine decay schedule

**Collection metrics (per generation):**
- `collect/success_rate` — fraction of episodes that succeeded
- `collect/successful_episodes` — count of kept trajectories
- `collect/dataset_samples` — JSONL samples written after filtering

**Evaluation metrics (per generation):**
- `eval/success_rate` + `eval/success_std`
- `eval/avg_steps` + `eval/median_steps` + `eval/std_steps`
- `eval/action_distribution` — bar chart per generation
- `improvement/success_rate_delta` — delta vs baseline
- `improvement/avg_steps_delta` — steps improvement vs baseline

### Artifacts — W&B Models
LoRA adapter weights saved as a W&B Artifact after every generation:

```python
import wandb
api = wandb.Api()
artifact = api.artifact(
    "leadgen12344-nanyang-technological-university-singapore/nova-planner/nova-lora-gen-3:latest",
    type="model"
)
artifact.download("./adapter_weights")
```

Artifacts are typed `model`, tagged per generation, and include full config metadata.

### Tracing & Evaluation — W&B Weave
All inference calls during evaluation are traced with W&B Weave:

```python
import weave

weave.init("nova-planner")

@weave.op()
def _agent_step(planner, serialized_state: str) -> str:
    return planner.predict(serialized_state)
```

Every `(input_state, predicted_action)` pair is captured with timestamps and latency.

### W&B Report
9-section report covering:
1. Methodology — environment, state representation, LoRA config, reward design
2. Training Dynamics — loss, perplexity, token accuracy, gradient norm, LR schedule
3. Data Collection — trajectory collection, filtering, dataset flywheel
4. Evaluation Results — success rate, episode length, action distribution
5. Generation Comparison Summary — scalar panels for all key metrics
6. Failure Analysis — two identified failure modes in gen-3 (16% remaining failures)
7. Ablations & Observations — 5 validated design decisions
8. Reproducibility — artifact pull snippet, seed policy
9. Key Takeaways

---

## Architecture

```
Environment (MiniGrid)
    ↓
State Serializer  →  plain text: position, goal, obstacles, step count
    ↓
Mistral Small 3.1 (24B Instruct) + LoRA  →  discrete action prediction
    ↓
Action Executor  →  step environment
    ↓
Reward Model  →  +1 success / -0.01 step / -1 timeout
    ↓
Trajectory Filter  →  keep score > 0
    ↓
JSONL Dataset  →  fine-tune  →  repeat
```

**Desktop / Vision mode:** the agent also takes real screenshots via `mss`, interprets them using Mistral Small 3.1's vision capability, and executes keyboard/mouse actions via `pyautogui`.

---

## Model Config

```yaml
base_model  : mistralai/Mistral-Small-3.1-24B-Instruct-2503
lora_r      : 16
lora_alpha  : 32
lora_drop   : 0.1
targets     : q_proj, v_proj
trainable   : ~26M / 24B params (0.108%)
precision   : bfloat16
hardware    : 4x A100 80GB (device_map=auto)
lr          : 2e-4 (linear warmup 5 steps, cosine decay)
batch_size  : 4
epochs      : 3 per generation
```

---

## Setup

```bash
pip install -r requirements.txt
wandb login
huggingface-cli login   # required to download Mistral Small 3.1
```

## Usage

```bash
# run the full self-improvement loop (collect → train → eval, 3 generations)
python main.py --mode self_improve --generations 3 --episodes 100

# collect data only
python main.py --mode collect

# train on existing data
python main.py --mode train

# evaluate a trained model
python main.py --mode evaluate

# run on real desktop using vision + keyboard/mouse
python main.py --mode desktop --task "Open a terminal and run ls"
```

Override any config inline:

```bash
python main.py --mode self_improve \
  --generations 5 \
  --episodes 200 \
  --env MiniGrid-FourRooms-v0 \
  --wandb-project nova-planner
```

---

## Project Structure

```
nova_agent/
├── env/
│   ├── minigrid_env.py       # MiniGrid wrapper, structured state parsing
│   ├── state_serializer.py   # state dict -> text prompt
│   ├── desktop_env.py        # real screen capture via mss
│   ├── desktop_agent.py      # vision loop with screenshot logging to W&B
│   └── io_controller.py      # keyboard/mouse control via pyautogui
├── models/
│   ├── planner.py            # Mistral Small 3.1 + LoRA, text and vision inference
│   ├── reward_model.py       # StepwiseReward, TrajectoryScorer
│   └── lora_config.py        # LoRA config
├── training/
│   ├── dataset_builder.py    # episode collection, JSONL export
│   ├── train_planner.py      # HF Trainer, W&B logging, artifact upload
│   └── self_improve.py       # N-generation loop with baseline eval
├── evaluation/
│   └── evaluate.py           # success rate, steps, action dist, Weave tracing
├── utils/
│   ├── config.py             # NOVAConfig dataclass
│   └── logging.py            # stdout logger
├── main.py                   # CLI entry point
└── create_report.py          # generates W&B report programmatically
```

---

## Reproduce the W&B Report

```bash
pip install wandb wandb-workspaces
python create_report.py
```

---

## License

Apache 2.0 — same as Mistral Small 3.1.
