# Skill: Analyze NOVA Generation Metrics

**Purpose:** Query W&B for the latest NOVA generation run and surface key metrics for the self-improvement decision.

**Trigger:** Use this after each generation completes to decide whether to continue training, adjust hyperparameters, or stop.

---

## Prompt Template

```
Using the W&B MCP, retrieve metrics from the most recent run in project "nova-planner".

Report the following:
1. eval/success_rate — did it improve vs the previous generation?
2. eval/avg_steps — is the agent getting more efficient?
3. collect/success_rate — is the data flywheel working (more successful collections)?
4. train/loss (final value) — is training converging?
5. improvement/success_rate_delta — total delta vs the zero-shot baseline

Then answer:
- Should we continue to the next generation, or has performance plateaued?
- Is the success_rate delta < 0.05 between the last two generations? If yes, suggest increasing lora_r from 16 to 32.
- Is avg_steps still above 20? If yes, suggest adding a step-efficiency reward term.
- Flag if collect/success_rate dropped vs the previous generation (sign of overfitting).
```

---

## Expected Output Format

```
Generation N summary:
  eval/success_rate    : 0.XX  (delta: +0.XX vs baseline)
  eval/avg_steps       : XX.X  (delta: -XX.X vs baseline)
  collect/success_rate : 0.XX
  train/loss (final)   : 0.XX

Decision: [continue / stop / adjust_hyperparams]
Reason: ...
Suggested change (if any): ...
```
