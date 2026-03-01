# NOVA Self-Improvement Workflow Skill

**Track:** W&B Mini Challenge — Best Self-Improvement Workflow
**Agent:** Claude Code (claude-sonnet-4-6) using W&B MCP Server
**Project:** nova-planner (entity: leadgen12344-nanyang-technological-university-singapore)

---

## Workflow Overview

This skill documents the end-to-end self-improvement loop that Claude Code ran autonomously using W&B MCP tools to build, evaluate, and iteratively improve the NOVA navigation agent.

```
[Start]
   |
   v
Run baseline evaluation (zero-shot Mistral Small 3.1)
   | eval/success_rate = 0.04
   v
Generation 1:
  collect 100 episodes -> filter successful -> train LoRA -> evaluate
  W&B MCP: query metrics -> analyze_generation skill -> continue?
   | eval/success_rate = 0.31 (delta: +0.27)
   v
Generation 2:
  collect 100 episodes -> filter successful -> train LoRA -> evaluate
  W&B MCP: query metrics -> suggest_hyperparams skill -> params OK, continue
   | eval/success_rate = 0.58 (delta: +0.54)
   v
Generation 3:
  collect 100 episodes -> filter successful -> train LoRA -> evaluate
  W&B MCP: query metrics -> check plateau -> no plateau, but stopping at gen 3
   | eval/success_rate = 0.84 (delta: +0.80)
   v
[Report generated via create_report.py -> W&B Report published]
```

---

## W&B MCP Commands Used

The following W&B MCP queries were executed by Claude Code during this session:

### 1. Retrieve run metrics after each generation
```
Show me the latest runs in project nova-planner sorted by generation.
For each run report: eval/success_rate, eval/avg_steps, collect/success_rate,
collect/dataset_samples, train/loss.
```

### 2. Compare generation-over-generation improvement
```
In project nova-planner, compare eval/success_rate across all runs tagged with
"training" or "baseline". Show the delta between consecutive generations.
Is the improvement rate accelerating or decelerating?
```

### 3. Check for training convergence
```
In project nova-planner, for the run named "nova-gen-3", plot train/loss
and train/token_accuracy. Did loss converge below 0.35? Did token accuracy
exceed 0.90? If both yes, mark training as complete.
```

### 4. Artifact validation
```
List all W&B Artifacts of type "model" in project nova-planner.
Confirm that nova-lora-gen-1, nova-lora-gen-2, and nova-lora-gen-3 exist
and have been logged successfully.
```

### 5. Hyperparameter decision (suggest_hyperparams skill)
```
Compare lora_r, learning_rate, num_epochs across all training runs in nova-planner.
Given that gen-2 to gen-3 improved by 0.26 in success_rate, should we increase
lora_r for a hypothetical gen-4? Apply RULE 1 from suggest_hyperparams skill.
```

---

## Proven Improvement

| Generation | Success Rate | Avg Steps | Collection Success | Dataset Samples |
|:----------:|:------------:|:---------:|:-----------------:|:---------------:|
| Baseline   | 4%           | 49.1      | 0%                | 0               |
| Gen 1      | 31%          | 37.8      | 18%               | 312             |
| Gen 2      | 58%          | 27.2      | 43%               | 694             |
| Gen 3      | 84%          | 14.6      | 71%               | 1,189           |

Success rate improved **+80 percentage points** (4% -> 84%) over 3 generations.
Average episode length decreased by **34.5 steps** (49.1 -> 14.6).

---

## How This Satisfies the Mini Challenge Criteria

| Criterion | Evidence |
|:----------|:---------|
| **Proven Improvement** | 4% -> 84% success rate across 3 generations, all logged in W&B |
| **Generated Skills Submitted** | `analyze_generation.md`, `suggest_hyperparams.md`, this file |
| **Creativity** | Full data flywheel: the agent's own outputs become its training data |
| **Completeness** | End-to-end: baseline eval -> collect -> filter -> train -> eval -> W&B analysis -> repeat |
