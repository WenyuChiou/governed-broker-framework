# Skill-Governed Flood Adaptation Experiment (v2-2 Clean)

This experiment demonstrates the **Generic Governed Broker Framework** applied to flood adaptation. All domain logic is isolated in configuration files (YAML/CSV), with the experiment code using only generic framework modules.

## Architecture: Experiment ↔ Framework Separation

| Layer | Experiment-Specific | Framework (Generic) |
|-------|---------------------|---------------------|
| **Agents** | `agent_initial_profiles.csv` | `agents.base_agent.BaseAgent` |
| **State** | CSV columns (dynamic) | `simulation.state_manager.SharedState` |
| **Skills** | `skill_registry.yaml` | `broker.skill_registry.SkillRegistry` |
| **Prompts** | `agent_types.yaml` | `broker.context_builder.create_context_builder` |
| **Validation** | `coherence_rules` in YAML | `validators.AgentValidator` |
| **Execution** | `FloodSimulation` (inherits) | `simulation.BaseSimulationEngine` |

## Validation (PMT Coherence)

The experiment uses **Protection Motivation Theory (PMT)** coherence rules to validate LLM decisions. These rules are grounded in behavioral science literature.

### Theoretical Foundation

PMT (Rogers, 1983) posits that protection motivation arises from two cognitive processes:

1. **Threat Appraisal (TP):** Perceived severity + vulnerability of the threat
2. **Coping Appraisal (CP):** Perceived efficacy + self-efficacy - response costs

> **Key Literature:**
> - Rogers, R.W. (1983). Cognitive and psychological processes in fear appeals and attitude change: A revised theory of protection motivation.
> - Floyd, D.L., Prentice-Dunn, S., & Rogers, R.W. (2000). A meta-analysis of research on protection motivation theory. *Journal of Applied Social Psychology*.
> - Bubeck, P., Botzen, W.J.W., & Aerts, J.C.J.H. (2012). A review of risk perceptions and other factors that influence flood mitigation behavior. *Risk Analysis*.

### Validation Rules

Defined in `agent_types.yaml`:

| Rule | Condition | Blocked Actions | PMT Justification |
|------|-----------|-----------------|-------------------|
| **urgency_check** | TP = High | `do_nothing` | High threat perception should trigger protective action (Rogers, 1983) |
| **coping_alignment** | CP = Low | `elevate_house` | Low coping capacity makes costly actions irrational (Floyd et al., 2000) |

```yaml
coherence_rules:
  urgency_check:
    construct: TP
    when_above: ["H"]
    blocked_skills: ["do_nothing"]
    message: "High Threat but chose inaction"
  
  coping_alignment:
    construct: CP
    when_above: ["L"]
    blocked_skills: ["elevate_house"]
    message: "Low Coping but chose expensive action"
```

**Design Decision:** `relocate` is NOT blocked even with low TP/CP to allow modeling of panic-driven or irrational human behavior, consistent with behavioral economics findings (Kahneman & Tversky, 1979).

### Validation Flow

1. LLM outputs `Threat Appraisal: High because...`
2. Parser extracts `TP = 'H'`
3. Validator checks: if `TP in ['H']` AND `decision in ['do_nothing']` → **Invalid**
4. Broker triggers retry with error message

## Data Input/Output

### Input Files
| File | Format | Purpose |
|------|--------|---------|
| `agent_initial_profiles.csv` | CSV | Agent profiles (any columns auto-available in prompts) |
| `flood_years.csv` | CSV | Years with flood events |
| `agent_types.yaml` | YAML | Prompt templates, validation rules, actions |
| `skill_registry.yaml` | YAML | Skill definitions and execution mappings |

### Output Files
| File | Location | Purpose |
|------|----------|---------|
| `experiment.log` | `results/<model>/` | High-level simulation log |
| `audit_trace.jsonl` | `results/<model>/` | Detailed decision trace |
| `audit_summary.json` | `results/<model>/` | Aggregated statistics |
| `comparison_results.png` | `results/<model>/` | Visualization |

## Running the Experiment

```bash
# Full experiment (100 agents, 10 years)
python run_experiment.py --model llama3.2:3b --num-agents 100 --num-years 10

# Quick validation test
python run_experiment.py --model mock --num-agents 50 --num-years 1

# With custom output directory
python run_experiment.py --model llama3.2:3b --output-dir results/my_run
```

## Adding New Agent Attributes

1. Add column to `agent_initial_profiles.csv` (e.g., `income`)
2. Use in prompt template: `Your income is {income}`
3. No code changes required ✅

## Framework Module Independence

The experiment code (`run_experiment.py`) imports ONLY from:
- `agents.base_agent` (generic agent)
- `simulation.state_manager` (generic state)
- `broker.*` (generic broker components)
- `validators.*` (generic validation)

**No framework files are modified for this experiment.**
