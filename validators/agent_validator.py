"""
Generic Agent Validator

Label-based validation - one validator for all agent types.
Rules are configured per agent_type label, not per file.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ValidationLevel(Enum):
    ERROR = "ERROR"      # Must fix, decision rejected
    WARNING = "WARNING"  # Log but allow


@dataclass
class ValidationResult:
    valid: bool
    level: ValidationLevel
    rule: str
    message: str
    agent_id: str
    field: Optional[str] = None
    value: Optional[float] = None
    constraint: Optional[str] = None


class AgentValidator:
    """
    Generic validator for any agent type.
    
    Uses agent_type label to lookup validation rules.
    Rules are defined in VALIDATION_RULES dict.
    
    Usage:
        v = AgentValidator()
        results = v.validate("insurance", "InsuranceCo", decision, state)
    """
    
    # Validation rules per agent_type
    VALIDATION_RULES: Dict[str, Dict] = {
        "insurance": {
            "rate_bounds": {
                "param": "premium_rate",
                "min": 0.02,
                "max": 0.15,
                "level": ValidationLevel.ERROR
            },
            "max_change": {
                "param": "premium_rate",
                "max_delta": 0.15,
                "level": ValidationLevel.WARNING
            },
            "solvency_floor": {
                "param": "solvency",
                "min": 0.0,
                "level": ValidationLevel.ERROR
            },
            "valid_decisions": {
                "values": ["RAISE", "LOWER", "MAINTAIN", "raise_premium", "lower_premium", "maintain_premium"],
                "level": ValidationLevel.ERROR
            }
        },
        "government": {
            "subsidy_bounds": {
                "param": "subsidy_rate",
                "min": 0.20,
                "max": 0.95,
                "level": ValidationLevel.ERROR
            },
            "max_change": {
                "param": "subsidy_rate",
                "max_delta": 0.15,
                "level": ValidationLevel.WARNING
            },
            "budget_reserve": {
                "param": "budget_used",
                "max": 0.90,
                "level": ValidationLevel.WARNING
            },
            "valid_decisions": {
                "values": ["INCREASE", "DECREASE", "MAINTAIN", "OUTREACH",
                          "increase_subsidy", "decrease_subsidy", "maintain_subsidy", "target_mg_outreach"],
                "level": ValidationLevel.ERROR
            }
        },
        "household": {
            "valid_decisions": {
                "values": ["FI", "HE", "EH", "BP", "RL", "Relocate", "DN", "Do Nothing",
                          "1", "2", "3", "4", "5"],
                "level": ValidationLevel.ERROR
            }
        }
    }
    
    def __init__(self):
        self.errors: List[ValidationResult] = []
        self.warnings: List[ValidationResult] = []
    
    def validate(
        self,
        agent_type: str,
        agent_id: str,
        decision: str,
        state: Dict[str, Any],
        prev_state: Dict[str, Any] = None
    ) -> List[ValidationResult]:
        """
        Validate agent decision based on agent_type rules.
        
        Args:
            agent_type: Type label (e.g., "insurance", "government", "household")
            agent_id: Agent identifier
            decision: Decision string
            state: Current agent state
            prev_state: Previous state (for delta checks)
        """
        results = []
        rules = self.VALIDATION_RULES.get(agent_type, {})
        
        if not rules:
            # Unknown agent type - just validate response format
            return self.validate_response_format(agent_id, decision)
        
        # 1. Validate decision is in allowed values
        if "valid_decisions" in rules:
            rule = rules["valid_decisions"]
            normalized = decision.lower().replace("_", "").replace(" ", "")
            valid_normalized = [v.lower().replace("_", "").replace(" ", "") for v in rule["values"]]
            if normalized not in valid_normalized:
                results.append(ValidationResult(
                    valid=False,
                    level=rule["level"],
                    rule="valid_decisions",
                    message=f"Invalid decision '{decision}' for {agent_type}",
                    agent_id=agent_id,
                    field="decision",
                    constraint=str(rule["values"][:4]) + "..."
                ))
        
        # 2. Validate rate/param bounds
        for rule_name, rule in rules.items():
            if rule_name in ["valid_decisions"]:
                continue
                
            param = rule.get("param")
            if not param or param not in state:
                continue
            
            value = state[param]
            
            # Min/Max bounds
            if "min" in rule and value < rule["min"]:
                results.append(ValidationResult(
                    valid=False,
                    level=rule["level"],
                    rule=rule_name,
                    message=f"{param} {value:.2f} below min {rule['min']:.2f}",
                    agent_id=agent_id,
                    field=param,
                    value=value,
                    constraint=f"min={rule['min']}"
                ))
            
            if "max" in rule and value > rule["max"]:
                results.append(ValidationResult(
                    valid=False,
                    level=rule["level"],
                    rule=rule_name,
                    message=f"{param} {value:.2f} above max {rule['max']:.2f}",
                    agent_id=agent_id,
                    field=param,
                    value=value,
                    constraint=f"max={rule['max']}"
                ))
            
            # Delta check
            if "max_delta" in rule and prev_state and param in prev_state:
                delta = abs(value - prev_state[param])
                if delta > rule["max_delta"]:
                    results.append(ValidationResult(
                        valid=False,
                        level=rule["level"],
                        rule=rule_name,
                        message=f"{param} change {delta:.2%} exceeds max {rule['max_delta']:.0%}",
                        agent_id=agent_id,
                        field=param,
                        value=delta,
                        constraint=f"max_delta={rule['max_delta']}"
                    ))
        
        self._categorize_results(results)
        return results
    
    def validate_response_format(
        self,
        agent_id: str,
        response: str,
        required_fields: List[str] = None
    ) -> List[ValidationResult]:
        """Validate LLM response has required fields."""
        results = []
        required = required_fields or ["INTERPRET:", "DECIDE:"]
        
        for field in required:
            if field not in response:
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    rule="response_format",
                    message=f"Missing '{field}' in response",
                    agent_id=agent_id,
                    field="response",
                    constraint=f"required: {required}"
                ))
        
        self._categorize_results(results)
        return results
    
    def validate_adjustment(
        self,
        agent_id: str,
        adjustment: float,
        min_adj: float = 0.0,
        max_adj: float = 0.15
    ) -> List[ValidationResult]:
        """Validate adjustment is within bounds."""
        results = []
        if not (min_adj <= adjustment <= max_adj):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                rule="adjustment_bounds",
                message=f"Adjustment {adjustment:.1%} outside [{min_adj:.0%}, {max_adj:.0%}]",
                agent_id=agent_id,
                field="adjustment",
                value=adjustment,
                constraint=f"[{min_adj}, {max_adj}]"
            ))
        self._categorize_results(results)
        return results
    
    def _categorize_results(self, results: List[ValidationResult]):
        """Sort results into errors and warnings."""
        for r in results:
            if r.level == ValidationLevel.ERROR:
                self.errors.append(r)
            else:
                self.warnings.append(r)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def summary(self) -> Dict:
        """Return validation summary."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "errors": [{"rule": e.rule, "message": e.message} for e in self.errors],
            "warnings": [{"rule": w.rule, "message": w.message} for w in self.warnings]
        }
    
    def reset(self):
        """Clear accumulated results."""
        self.errors = []
        self.warnings = []


# Backward compatibility alias
InstitutionalValidator = AgentValidator
