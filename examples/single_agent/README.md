# Flood Adaptation Single-Agent Experiment

## Overview

This experiment simulates household flood adaptation decisions using LLM-based agents with the Governed Broker Framework. It compares agent behavior with and without the governance layer.

## Memory & Retrieval System

### Available Memory Engines

| Engine         | Description                                   | Parameters                             | Use Case                |
| -------------- | --------------------------------------------- | -------------------------------------- | ----------------------- |
| `window`       | Sliding window (last N items)                 | `window_size=3`                        | Simple, predictable     |
| `importance`   | Active retrieval (recency + significance)     | `window_size=3`, `top_k_significant=2` | Retains critical events |
| `humancentric` | Emotional encoding + stochastic consolidation | See below                              | Human-realistic memory  |

### Usage

```powershell
# Default sliding window
python run_modular_experiment.py --model llama3.2:3b --memory-engine window

# Importance-based retrieval
python run_modular_experiment.py --model gemma3:4b --memory-engine importance
```

### ImportanceMemoryEngine Parameters

All weights use 0-1 scale:

```python
ImportanceMemoryEngine(
    window_size=3,              # int: Recent items always included
    top_k_significant=2,        # int: Top historical events
    weights={
        "critical": 1.0,        # float: Maximum importance (floods, damage)
        "high": 0.8,            # float: High importance (claims, decisions)
        "medium": 0.5,          # float: Medium importance (observations)
        "routine": 0.1          # float: Minimal importance
    }
)
```

---

## Behavioral Analysis: OLD Baseline vs NEW Governed Framework

### Summary

| Model          | OLD Baseline    | NEW Governed     | Key Change          |
| -------------- | --------------- | ---------------- | ------------------- |
| Gemma 3 4B     | Elevation 44.9% | Insurance 43%    | Strategy shift      |
| Llama 3.2 3B   | Elevation 83.3% | Do Nothing 67.4% | More conservative   |
| DeepSeek-R1 8B | Elevation 67.8% | Elevation 82%    | Stronger preference |
| GPT-OSS 20B    | Elevation 64%   | Elevation 77%    | +13% elevation      |

### Validation Statistics

| Model          | Approval Rate      | Blocked | Errors |
| -------------- | ------------------ | ------- | ------ |
| Gemma 3 4B     | 100.0% (1000/1000) | 0       | 0      |
| Llama 3.2 3B   | 81.1% (811/1000)   | 189     | 295    |
| DeepSeek-R1 8B | 78.9% (789/1000)   | 211     | 322    |
| GPT-OSS 20B    | 88.5% (885/1000)   | 115     | 125    |

### What the Framework Blocks

The governance framework blocks decisions that violate PMT (Protection Motivation Theory) rules:

1. **High Threat + Do Nothing**: Blocked when agent perceives high flood threat but chooses inaction
2. **Invalid Decision Format**: Blocked when LLM output doesn't match expected options
3. **Inconsistent Reasoning**: Blocked when TP/CP assessment contradicts final decision

### Why Behavioral Differences Occur

1. **Prompt Standardization**: OLD uses freeform prompts; NEW uses structured templates
2. **Decision Validation**: NEW enforces PMT-consistent decisions
3. **Memory Injection**: NEW explicitly formats memory as bulleted list
4. **Trust Verbalization**: NEW includes quantified trust values (0-1 scale)

---

## Results Structure

```
results/
├── Gemma_3_4B/                  # OLD baseline
├── gemma3_4b_strict/            # NEW governed
│   ├── audit_summary.json       # Validation stats
│   ├── household_governance_audit.csv
│   ├── simulation_log.csv       # Decision traces
│   └── comparison_results.png
└── old_vs_new_comparison_2x4.png
```

## Running Experiments

```powershell
# Quick test (5 agents, 3 years)
python run_modular_experiment.py --model llama3.2:3b --agents 5 --years 3

# Full experiment
python run_modular_experiment.py --model gemma3:4b --agents 100 --years 10

# Generate comparison chart
python generate_old_vs_new_2x4.py
```

## References

- Park et al. (2023) "Generative Agents" - Memory stream architecture
- Chapter 8 Memory and Retrieval - Cognitive science foundations
