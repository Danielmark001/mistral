# Skill: Suggest Hyperparameter Adjustments

**Purpose:** Compare all NOVA training runs in W&B and recommend hyperparameter changes for the next generation based on observed trends.

**Trigger:** Use this when success rate improvement between generations is less than 5 percentage points, or when training loss is not converging.

---

## Prompt Template

```
Using the W&B MCP, compare all runs in project "nova-planner" tagged with "training".

For each run retrieve: lora_r, learning_rate, num_epochs, eval/success_rate, train/loss (final).

Then apply these rules:

RULE 1 — Rank capacity:
  If success_rate < 0.70 and the last two generations improved by < 0.08,
  suggest increasing lora_r from 16 to 32 (doubles adapter capacity).

RULE 2 — Learning rate:
  If train/loss final is still above 0.50 after 3 epochs,
  suggest increasing learning_rate from 2e-4 to 3e-4 and adding 10 warmup steps.

RULE 3 — Epochs:
  If collect/success_rate > 0.60 but eval/success_rate is not following,
  the model may be underfitting — suggest increasing num_epochs from 3 to 5.

RULE 4 — Data quality:
  If collect/dataset_samples < 200 for a generation,
  suggest increasing episodes_per_gen from 100 to 200 for the next run.

RULE 5 — Plateau detection:
  If the last two generations have eval/success_rate within 0.03 of each other,
  stop the loop and log a summary artifact.

Output a JSON config patch to apply for the next generation:
{
  "lora_r": ...,
  "learning_rate": ...,
  "num_epochs": ...,
  "episodes_per_gen": ...,
  "reason": "..."
}
```

---

## Usage in the Self-Improvement Loop

After each generation, the coding agent runs this skill, gets the JSON patch, and applies it to `NOVAConfig` before the next generation starts. This closes the eval → analyze → improve loop automatically.
