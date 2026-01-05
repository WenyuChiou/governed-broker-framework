# Agent Pipeline Specification

## 1. Context → Reasoning → Action Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT DECISION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        1. CONTEXT (Input)                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   A. State (Read-only)                                              │   │
│  │      ├── Static: agent_id, mg, tenure, region_id                    │   │
│  │      └── Dynamic: elevated, has_insurance, cumulative_damage        │   │
│  │                                                                     │   │
│  │   B. Memory (State) → Retrieval (Tool)                              │   │
│  │      ├── memory.retrieve(top_k=5)  ← Tool invocation                │   │
│  │      └── Returns: List[str] of past experiences                     │   │
│  │                                                                     │   │
│  │   C. Environment (Read-only)                                        │   │
│  │      ├── year, flood_occurred                                       │   │
│  │      ├── subsidy_rate, premium_rate                                 │   │
│  │      └── neighbor_actions (optional, limited visibility)            │   │
│  │                                                                     │   │
│  │   OUTPUT: Prompt (text for LLM)                                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       2. REASONING (LLM)                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   INPUT: Prompt                                                     │   │
│  │                                                                     │   │
│  │   PROCESSING: LLM generates structured response                     │   │
│  │      ├── Evaluate: TP, CP, SP, SC, PA (level + explanation)         │   │
│  │      ├── Decide: decision_number                                    │   │
│  │      └── Justify: justification                                     │   │
│  │                                                                     │   │
│  │   OUTPUT: Raw text response                                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        3. ACTION (Output)                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   A. Parser                                                         │   │
│  │      ├── Input: Raw LLM response                                    │   │
│  │      └── Output: HouseholdOutput dataclass                          │   │
│  │                                                                     │   │
│  │   B. Validator                                                      │   │
│  │      ├── Input: HouseholdOutput + State                             │   │
│  │      ├── Check: R1-R9 rules                                         │   │
│  │      └── Output: ValidationResult (errors, warnings)                │   │
│  │                                                                     │   │
│  │   C. Executor                                                       │   │
│  │      ├── If valid: Apply state changes                              │   │
│  │      └── If invalid: Log error, no state change                     │   │
│  │                                                                     │   │
│  │   D. Audit                                                          │   │
│  │      └── Log: Complete trace (regardless of validation)             │   │
│  │                                                                     │   │
│  │   E. Memory Update                                                  │   │
│  │      └── memory.add_experience(decision, year)                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Audit Log Specification

### 2.1 Household Audit Trace (`household_audit.jsonl`)

每一行是一個 JSON 物件，記錄一次決策：

```json
{
  "timestamp": "2026-01-05T03:45:00.123456",
  "year": 3,
  
  "agent": {
    "agent_id": "H001",
    "mg": true,
    "tenure": "Owner",
    "region_id": "NJ"
  },
  
  "state_before": {
    "elevated": false,
    "has_insurance": true,
    "cumulative_damage": 50000.0,
    "relocated": false
  },
  
  "context": {
    "subsidy_rate": 0.5,
    "premium_rate": 0.05,
    "flood_occurred": true,
    "memory_retrieved": [
      "Year 2: Flood caused $30,000 damage",
      "Year 2: Purchased insurance"
    ]
  },
  
  "reasoning": {
    "constructs": {
      "TP": {"level": "HIGH", "explanation": "Experienced significant damage"},
      "CP": {"level": "MODERATE", "explanation": "Limited income but subsidy helps"},
      "SP": {"level": "HIGH", "explanation": "Government offers 50% subsidy"},
      "SC": {"level": "MODERATE", "explanation": "Believe I can act with support"},
      "PA": {"level": "PARTIAL", "explanation": "Have insurance but not elevated"}
    },
    "justification": "Given high threat and available subsidy, elevation is best choice."
  },
  
  "action": {
    "decision_number": 2,
    "decision_skill": "elevate_house"
  },
  
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": ["R5: LOW coping but chose elevate_house"],
    "rules_checked": ["R1", "R2", "R3", "R5", "R6", "R7", "R8"]
  },
  
  "state_after": {
    "elevated": true,
    "has_insurance": true,
    "cumulative_damage": 50000.0,
    "relocated": false
  }
}
```

### 2.2 Institutional Audit Trace (`institutional_audit.jsonl`)

```json
{
  "timestamp": "2026-01-05T03:46:00.123456",
  "year": 3,
  "agent_type": "insurance",
  "agent_id": "InsuranceCo",
  
  "state_before": {
    "loss_ratio": 0.85,
    "total_policies": 15,
    "risk_pool": 500000,
    "premium_rate": 0.05
  },
  
  "reasoning": {
    "analysis": "Loss ratio is concerning at 85%",
    "justification": "Raise premium to maintain solvency"
  },
  
  "action": {
    "decision": "RAISE",
    "adjustment_pct": 0.08
  },
  
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": []
  },
  
  "state_after": {
    "premium_rate": 0.054
  }
}
```

### 2.3 Audit Summary (`audit_summary.json`)

```json
{
  "experiment_name": "exp3_multi_agent",
  "simulation_config": {
    "years": 10,
    "n_households": 25,
    "seed": 42,
    "llm_model": "llama3.2:3b",
    "llm_temperature": 0.7
  },
  
  "totals": {
    "household_decisions": 250,
    "institutional_decisions": 20
  },
  
  "decision_distribution": {
    "buy_insurance": {"count": 82, "pct": "32.8%"},
    "elevate_house": {"count": 35, "pct": "14.0%"},
    "relocate": {"count": 53, "pct": "21.2%"},
    "do_nothing": {"count": 80, "pct": "32.0%"}
  },
  
  "construct_distribution": {
    "TP": {"LOW": 40, "MODERATE": 180, "HIGH": 30},
    "CP": {"LOW": 140, "MODERATE": 90, "HIGH": 20},
    "SP": {"LOW": 15, "MODERATE": 170, "HIGH": 65},
    "SC": {"LOW": 60, "MODERATE": 175, "HIGH": 15},
    "PA": {"NONE": 130, "PARTIAL": 80, "FULL": 40}
  },
  
  "validation_summary": {
    "total_errors": 5,
    "total_warnings": 75,
    "error_rate": "2.0%",
    "warning_rate": "30.0%",
    "by_rule": {
      "R1": {"errors": 0, "warnings": 12},
      "R4": {"errors": 3, "warnings": 0},
      "R5": {"errors": 0, "warnings": 45},
      "R8": {"errors": 2, "warnings": 18}
    }
  },
  
  "outcome_metrics": {
    "final_elevated": 23,
    "final_insured": 12,
    "final_relocated": 8,
    "total_damage": 1250000.0,
    "total_claims_paid": 320000.0
  },
  
  "finalized_at": "2026-01-05T03:50:00.000000"
}
```

---

## 3. Validator Log Specification

### 3.1 ValidationResult Structure

```python
@dataclass
class ValidationResult:
    valid: bool                        # True if no errors
    errors: List[str]                  # List of error messages (blocking)
    warnings: List[str]                # List of warning messages (non-blocking)
    rule_checks: Dict[str, bool]       # Which rules were checked and passed
```

### 3.2 Rule Output Format

每個 rule 返回固定格式的訊息：

```
[RULE_ID]: [Description of violation/concern]
```

例如：
```
R1: HIGH threat + HIGH coping but chose do_nothing - possible irrational behavior
R4: Renters cannot elevate property they don't own
R5: LOW coping perception but chose elevate_house - may face affordability issues
R8: PA=NONE doesn't match actual state (elevated=True, insured=False) -> expected PARTIAL
```

### 3.3 Validation Output in Audit

```json
"validation": {
  "valid": false,
  "errors": [
    "R4: Renters cannot elevate property they don't own",
    "R9: Renter cannot elevate_house - valid actions: buy_insurance, relocate, do_nothing"
  ],
  "warnings": [
    "R5: LOW coping perception but chose elevate_house - may face affordability issues"
  ],
  "rules_checked": ["R1", "R2", "R4", "R5", "R6", "R7", "R8", "R9"]
}
```

---

## 4. Implementation Mapping

| Component | File | Function/Class |
|-----------|------|----------------|
| Context Builder | `prompts.py` | `build_household_prompt()` |
| Memory Retrieval | `broker/memory.py` | `CognitiveMemory.retrieve()` |
| LLM Call | `run_experiment.py` | `call_llm()` |
| Parser | `parsers.py` | `parse_household_response()` |
| Validator | `validators.py` | `RuleRegistry.validate()` |
| Executor | `run_experiment.py` | `apply_household_decision()` |
| Audit Writer | `audit_writer.py` | `AuditWriter.write_household_trace()` |
| Memory Update | `broker/memory.py` | `CognitiveMemory.update_after_decision()` |

---

## 5. Data Flow Summary

```
Year Loop:
│
├── For each Household Agent:
│   │
│   ├── 1. CONTEXT
│   │   ├── Read: agent.state.*
│   │   ├── Tool: memory.retrieve(top_k=5)
│   │   ├── Read: environment.*
│   │   └── Build: prompt = build_household_prompt(...)
│   │
│   ├── 2. REASONING
│   │   ├── Call: response = call_llm(prompt)
│   │   └── (LLM generates TP/CP/SP/SC/PA + decision)
│   │
│   └── 3. ACTION
│       ├── Parse: output = parse_household_response(response)
│       ├── Validate: result = validator.validate(output, state)
│       ├── Execute: if result.valid → apply_decision(agent, output)
│       ├── Audit: audit.write_household_trace(output, state, context)
│       └── Memory: agent.memory.update_after_decision(...)
│
├── For Insurance Agent:
│   └── (Similar flow with InsuranceOutput)
│
├── For Government Agent:
│   └── (Similar flow with GovernmentOutput)
│
├── Environment Update:
│   ├── Process: flood_event()
│   ├── Process: calculate_damage()
│   └── Process: insurance_claims()
│
└── End Year
```
