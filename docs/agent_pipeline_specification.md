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

---

## 6. Behavior → State Changes

### 6.1 Household State Changes

| Decision | State Changes | Type |
|----------|---------------|------|
| `buy_insurance` | `has_insurance = True` | Annual (resets each year) |
| `elevate_house` | `elevated = True` | Permanent |
| `relocate` | `relocated = True` | Permanent |
| `do_nothing` | (no change) | - |

```python
def apply_household_decision(agent: HouseholdAgent, output: HouseholdOutput):
    """Apply validated decision to agent state."""
    
    if output.decision_skill == "buy_insurance":
        agent.state.has_insurance = True
        # Cost: property_value * premium_rate
        
    elif output.decision_skill == "elevate_house":
        agent.state.elevated = True
        # Cost: ELEVATION_COST * (1 - subsidy_rate)
        
    elif output.decision_skill == "relocate":
        agent.state.relocated = True
        # Remove from active population
        
    # do_nothing: no state change
```

### 6.2 Insurance State Changes

| Decision | State Changes |
|----------|---------------|
| `RAISE` | `premium_rate += adjustment_pct` |
| `LOWER` | `premium_rate -= adjustment_pct` |
| `MAINTAIN` | (no change) |

```python
def apply_insurance_decision(ins: InsuranceAgent, output: InsuranceOutput):
    if output.decision == "RAISE":
        ins.state.premium_rate *= (1 + output.adjustment_pct)
    elif output.decision == "LOWER":
        ins.state.premium_rate *= (1 - output.adjustment_pct)
```

### 6.3 Government State Changes

| Decision | State Changes |
|----------|---------------|
| `INCREASE` | `subsidy_rate += adjustment_pct` |
| `DECREASE` | `subsidy_rate -= adjustment_pct` |
| `MAINTAIN` | (no change) |

```python
def apply_government_decision(gov: GovernmentAgent, output: GovernmentOutput):
    if output.decision == "INCREASE":
        gov.state.subsidy_rate = min(1.0, gov.state.subsidy_rate + output.adjustment_pct)
    elif output.decision == "DECREASE":
        gov.state.subsidy_rate = max(0.0, gov.state.subsidy_rate - output.adjustment_pct)
    
    gov.state.priority_group = output.priority  # "MG" or "ALL"
```

### 6.4 Annual Resets

```python
def annual_reset(households: List[HouseholdAgent]):
    """Reset non-permanent states at year start."""
    for agent in households:
        if not agent.state.relocated:
            # Insurance is annual - resets unless renewed
            agent.state.has_insurance = False
```

---

## 7. Environment Modules

### 7.1 Module Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT MODULES                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌───────────────────────────────────────────────────┐    │
│   │  Module 1: Flood Event Generator                 │    │
│   │  - Probabilistic flood occurrence               │    │
│   │  - Flood severity (depth/extent)                │    │
│   └───────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│   ┌───────────────────────────────────────────────────┐    │
│   │  Module 2: Damage Calculator                     │    │
│   │  - Physical damage based on flood + state       │    │
│   │  - Elevated = reduced damage                    │    │
│   └───────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│   ┌───────────────────────────────────────────────────┐    │
│   │  Module 3: Insurance Claims (Future)             │    │
│   │  - Claim processing                             │    │
│   │  - Payout calculation                           │    │
│   └───────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│   ┌───────────────────────────────────────────────────┐    │
│   │  Module 4: Buyout Program (Future)               │    │
│   │  - Eligibility check                            │    │
│   │  - Approval/denial                              │    │
│   └───────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Module 1: Flood Event Generator

```python
class FloodEventGenerator:
    """Generate flood events probabilistically."""
    
    def __init__(self, config: FloodConfig):
        self.annual_probability = config.annual_probability  # e.g., 0.3
        self.severity_distribution = config.severity_dist    # Normal(μ, σ)
    
    def generate(self, year: int, seed: int = None) -> FloodEvent:
        """Generate flood event for the year."""
        if random.random() < self.annual_probability:
            severity = self.severity_distribution.sample()
            return FloodEvent(
                year=year,
                occurred=True,
                severity=severity,  # 0.0 - 1.0
                affected_regions=["NJ", "NY"]
            )
        return FloodEvent(year=year, occurred=False)
```

### 7.3 Module 2: Damage Calculator

```python
class DamageCalculator:
    """Calculate damage based on flood event and agent state."""
    
    def calculate(self, agent: HouseholdAgent, flood: FloodEvent) -> float:
        """
        Calculate damage for a household.
        
        Factors:
        - Flood severity
        - Elevated status (reduces damage)
        - Property value
        - MG status (may affect exposure)
        """
        if not flood.occurred or agent.state.relocated:
            return 0.0
        
        base_damage = agent.state.property_value * flood.severity
        
        # Elevation reduces damage by 80%
        if agent.state.elevated:
            base_damage *= 0.2
        
        # MG may have more vulnerable housing
        if agent.state.mg:
            base_damage *= 1.2  # 20% more vulnerable
        
        return min(base_damage, agent.state.property_value)
```

### 7.4 Environment → Agent Observation

```python
def prepare_agent_context(
    agent: HouseholdAgent,
    env: Environment,
    year: int
) -> Dict:
    """Prepare observable context for agent."""
    
    return {
        "year": year,
        "flood_occurred": env.last_flood.occurred if env.last_flood else False,
        "subsidy_rate": env.government.state.subsidy_rate,
        "premium_rate": env.insurance.state.premium_rate,
        
        # Household can ONLY see their own damage
        "own_damage": agent.state.cumulative_damage,
        
        # Limited neighbor observation (optional)
        "neighbor_adaptations": count_neighbor_adaptations(agent, env)
    }
```

---

## 8. Future: Disaster Model Modules

> **Note:** 以下模組待後續實作

### 8.1 Module 3: Insurance Claims Processor

```python
class InsuranceClaimsProcessor:
    """Process insurance claims after flood event."""
    
    def process_claim(
        self, 
        agent: HouseholdAgent, 
        damage: float,
        insurance: InsuranceAgent
    ) -> ClaimResult:
        """
        Process insurance claim.
        
        Returns:
            ClaimResult with:
            - approved: bool
            - payout: float
            - denial_reason: Optional[str]
        """
        if not agent.state.has_insurance:
            return ClaimResult(approved=False, payout=0, denial_reason="No policy")
        
        # NFIP: Payout capped at policy limit
        policy_limit = 250_000  # NFIP building limit
        payout = min(damage, policy_limit)
        
        # Deductible
        deductible = 1_000
        payout = max(0, payout - deductible)
        
        return ClaimResult(
            approved=True,
            payout=payout,
            out_of_pocket=damage - payout
        )
```

**Household 可見：**
- `claim_approved: bool`
- `claim_payout: float`
- `out_of_pocket: float`

### 8.2 Module 4: Buyout Program

```python
class BuyoutProgram:
    """Government buyout/acquisition program."""
    
    def check_eligibility(
        self, 
        agent: HouseholdAgent,
        government: GovernmentAgent
    ) -> BuyoutEligibility:
        """
        Check if agent is eligible for buyout.
        
        Eligibility factors:
        - Repetitive loss (2+ floods)
        - Severe repetitive loss
        - Government budget available
        - Priority group
        """
        # Repetitive loss check
        is_repetitive = agent.state.flood_count >= 2
        
        # Priority check
        priority_match = (
            government.state.priority_group == "ALL" or
            (government.state.priority_group == "MG" and agent.state.mg)
        )
        
        # Budget check
        budget_available = government.state.buyout_budget > 0
        
        return BuyoutEligibility(
            eligible=is_repetitive and priority_match and budget_available,
            reason=self._get_reason(...)
        )
    
    def process_application(
        self, 
        agent: HouseholdAgent
    ) -> BuyoutResult:
        """
        Process buyout application.
        
        Returns:
            - approved: bool
            - offer_amount: float (pre-flood fair market value)
            - processing_time_years: int
        """
        pass  # Future implementation
```

**Household 可見：**
- `buyout_eligible: bool`
- `buyout_offer: Optional[float]`

### 8.3 Module Integration Order

1. ✅ **Flood Event Generator** (已有)
2. ✅ **Damage Calculator** (已有)
3. 🔜 **Insurance Claims** (next)
4. 📋 **Buyout Program** (later)

---

## 9. Insurance Type Configuration

```python
@dataclass
class InsuranceAgentState:
    agent_id: str = "InsuranceCo"
    
    # Type configuration
    insurance_type: str = "NFIP"  # "NFIP", "private", "mixed"
    
    # NFIP: Government-backed
    #   - Regulated rates
    #   - Universal coverage
    #   - Federal reinsurance
    
    # Private: Market-based
    #   - Risk-based pricing
    #   - May deny coverage
    #   - Higher limits available
    
    # Metrics
    premium_rate: float = 0.05
    loss_ratio: float = 0.0
    total_policies: int = 0
    risk_pool: float = 1_000_000
```
