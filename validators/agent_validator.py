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


from broker.agent_config import load_agent_config, ValidationRule, CoherenceRule

class AgentValidator:
    """
    Generic validator for any agent type.
    
    Uses agent_type label to lookup validation rules from agent_types.yaml.
    """
    
    def __init__(self, config_path: str = None):
        self.config = load_agent_config(config_path)
        self.errors: List[ValidationResult] = []
        self.warnings: List[ValidationResult] = []
    
    def validate(
        self,
        agent_type: str,
        agent_id: str,
        decision: str,
        state: Dict[str, Any],
        prev_state: Dict[str, Any] = None,
        reasoning: Dict[str, str] = None
    ) -> List[ValidationResult]:
        """
        Validate agent decision based on agent_type rules.
        
        Args:
            agent_type: Type label (e.g., "insurance", "government", "household")
            agent_id: Agent identifier
            decision: Decision string
            state: Current agent state
            prev_state: Previous state (for delta checks)
            reasoning: Optional reasoning dictionary (contains PMT labels)
        """
        results = []
        
        # 0. Handle subtypes (e.g., household_mg -> household) if defined in config mapping
        # This mapping should ideally be handled by the Config loader, but we preserve it generically
        base_type = self.config.get_base_type(agent_type)
            
        # 1. Validate decision is in allowed values
        valid_actions = self.config.get_valid_actions(base_type)
        if valid_actions:
            normalized = decision.lower().replace("_", "").replace(" ", "")
            valid_normalized = [v.lower().replace("_", "").replace(" ", "") for v in valid_actions]
            if normalized not in valid_normalized:
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    rule="valid_decisions",
                    message=f"Invalid decision '{decision}' for {agent_type}",
                    agent_id=agent_id,
                    field="decision",
                    constraint=str(valid_actions[:4]) + "..."
                ))
        
        # 2. Validate Cognitive/Construct Coherence (if defined in config)
        coherence_rules = self.config.get_coherence_rules(base_type)
        if coherence_rules and reasoning:
            results.extend(self.validate_coherence(agent_id, base_type, state, reasoning))
        
        # 3. Validate rate/param bounds
        rules = self.config.get_validation_rules(base_type)
        for rule_name, rule in rules.items():
            param = rule.param
            if not param or param not in state:
                continue
            
            value = state[param]
            lv = ValidationLevel.ERROR if rule.level == "ERROR" else ValidationLevel.WARNING
            
            # Min/Max bounds
            if rule.min_val is not None and value < rule.min_val:
                results.append(ValidationResult(
                    valid=False,
                    level=lv,
                    rule=rule_name,
                    message=rule.message or f"{param} {value:.2f} below min {rule.min_val:.2f}",
                    agent_id=agent_id,
                    field=param,
                    value=value,
                    constraint=f"min={rule.min_val}"
                ))
            
            if rule.max_val is not None and value > rule.max_val:
                results.append(ValidationResult(
                    valid=False,
                    level=lv,
                    rule=rule_name,
                    message=rule.message or f"{param} {value:.2f} above max {rule.max_val:.2f}",
                    agent_id=agent_id,
                    field=param,
                    value=value,
                    constraint=f"max={rule.max_val}"
                ))
            
            # Delta check
            if rule.max_delta is not None and prev_state and param in prev_state:
                delta = abs(value - prev_state[param])
                if delta > rule.max_delta:
                    results.append(ValidationResult(
                        valid=False,
                        level=lv,
                        rule=rule_name,
                        message=rule.message or f"{param} change {delta:.2%} exceeds max {rule.max_delta:.0%}",
                        agent_id=agent_id,
                        field=param,
                        value=delta,
                        constraint=f"max_delta={rule.max_delta}"
                    ))
        
        self._categorize_results(results)
        return results
        
        self._categorize_results(results)
        return results

    def validate_coherence(
        self,
        agent_id: str,
        agent_type: str,
        state: Dict[str, Any],
        reasoning: Dict[str, str]
    ) -> List[ValidationResult]:
        """
        Validate logical coherence between Agent State and LLM Cognitive Labels.
        Uses rules from agent_types.yaml.
        """
        results = []
        rules = self.config.get_coherence_rules(agent_type)
        
        # Helper to safely get label (handles brackets like [H])
        def get_label(key):
            # Check both raw key and EVAL_ prefixed key
            val = reasoning.get(key, reasoning.get(f"EVAL_{key}", "")).upper()
            if not val:
                return None
            
            # Extract label from brackets or take first char
            if ']' in val:
                label = val.split(']')[0].replace("[", "").replace("]", "").strip()
            else:
                label = val.strip()
            
            return label

        for rule in rules:
            label = get_label(rule.construct)
            if not label: continue
            
            # Get current value for comparison
            current_val = 0.0
            if rule.state_field:
                current_val = state.get(rule.state_field, 0.5)
            elif rule.state_fields:
                vals = [state.get(f, 0.5) for f in rule.state_fields]
                if rule.aggregation == "average":
                    current_val = sum(vals) / len(vals)
                elif rule.aggregation == "any_true":
                    current_val = 1.0 if any(v > 0.5 for v in vals) else 0.0
            
            
            # 1. State-Label Coherence Check (Numeric Constraints)
            is_coherent = True
            label_matched = False
            
            if current_val >= rule.threshold:
                if rule.expected_levels:
                    # Generic inclusion check (works for L, M, H or FULL, PARTIAL, NONE)
                    found = False
                    for level in rule.expected_levels:
                        if label.startswith(level.upper()):
                            found = True
                            label_matched = True
                            break
                    if not found:
                        is_coherent = False
            
            if not is_coherent:
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.WARNING,
                    rule=f"coherence_{rule.construct.lower()}",
                    message=rule.message or f"Coherence issue with {rule.construct}",
                    agent_id=agent_id,
                    field=rule.construct,
                    value=label,
                    constraint=f"Expected one of {rule.expected_levels} when state >= {rule.threshold}"
                ))

            # 2. Label-Action Coherence Check (PMT Logic)
            # If label matches 'when_above' (label_matched=True), check if decision is blocked
            if label_matched and rule.blocked_skills:
                 # Normalized decision check
                current_decision_norm = reasoning.get("decision", "").lower().replace("_", "")
                
                for blocked in rule.blocked_skills:
                    blocked_norm = blocked.lower().replace("_", "")
                    if blocked_norm in current_decision_norm:
                        results.append(ValidationResult(
                            valid=False,
                            level=ValidationLevel.ERROR, # Strictly reject
                            rule=f"action_logic_{rule.construct.lower()}",
                            message=f"Action '{current_decision_norm}' incompatible with high {rule.construct} ({label})",
                            agent_id=agent_id,
                            field=rule.construct,
                            value=current_decision_norm,
                            constraint=f"Blocked: {rule.blocked_skills} when {rule.construct} is {label}"
                        ))
                        break

            # 3. Text-Action Coherence Check (Semantic Constraints)
            # Checks if specific phrases in reasoning (e.g. "too expensive") contradict decision
            if rule.trigger_phrases and rule.blocked_skills:
                # Concatenate all reasoning text to search for triggers
                full_reasoning = " ".join(str(v) for v in reasoning.values()).lower()
                
                triggered = False
                for phrase in rule.trigger_phrases:
                    if phrase.lower() in full_reasoning:
                        triggered = True
                        break
                
                if triggered:
                    # Normalized decision check
                    current_decision_norm = reasoning.get("decision", "").lower().replace("_", "")
                    
                    for blocked in rule.blocked_skills:
                        blocked_norm = blocked.lower().replace("_", "")
                        # Check if decision matches a blocked skill
                        # (Simple containment check handles aliases roughly, but exact ID match is better if available)
                        if blocked_norm in current_decision_norm:
                            results.append(ValidationResult(
                                valid=False,
                                level=ValidationLevel.ERROR, # Strictly reject contradictions
                                rule=f"semantic_{rule.construct.lower()}",
                                message=rule.message or f"Reasoning contradicts action (claimed '{phrase}')",
                                agent_id=agent_id,
                                field=rule.construct,
                                value=current_decision_norm,
                                constraint=f"Blocked: {rule.blocked_skills}"
                            ))
                            break
            
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
