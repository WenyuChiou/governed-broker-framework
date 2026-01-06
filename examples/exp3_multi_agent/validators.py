"""
Exp3 Validators - Extensible Rule-Based Validation

Architecture:
- ValidationRule: Base class for all rules
- RuleRegistry: Central registry for rules
- Rules tagged with eligible agent types
- Easy to add new rules

Agent Types:
- household_owner
- household_renter
- insurance
- government
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Type
from examples.exp3_multi_agent.parsers import HouseholdOutput, InsuranceOutput, GovernmentOutput


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rule_checks: Dict[str, bool] = field(default_factory=dict)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another result into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            rule_checks={**self.rule_checks, **other.rule_checks}
        )


# =============================================================================
# VALIDATION RULE BASE CLASS
# =============================================================================

class ValidationRule(ABC):
    """
    Base class for validation rules.
    
    Subclasses must define:
    - rule_id: Unique identifier (e.g., "R1", "R9")
    - description: Human-readable explanation
    - eligible_agent_types: Which agents this applies to
    - severity: "error" or "warning"
    """
    
    rule_id: str = "BASE"
    description: str = "Base validation rule"
    eligible_agent_types: Set[str] = {"*"}  # "*" = all agents
    severity: str = "error"  # "error" or "warning"
    
    @abstractmethod
    def check(self, output: Any, state: Dict[str, Any]) -> Optional[str]:
        """
        Check if the rule is violated.
        
        Args:
            output: Parsed LLM output (HouseholdOutput, InsuranceOutput, etc.)
            state: Agent's current state
            
        Returns:
            Error/warning message if violated, None if passed
        """
        pass
    
    def applies_to(self, agent_type: str) -> bool:
        """Check if this rule applies to the given agent type."""
        return "*" in self.eligible_agent_types or agent_type in self.eligible_agent_types


# =============================================================================
# HOUSEHOLD RULES (Literature-backed)
# =============================================================================

class R1_HighTPCPDoNothing(ValidationRule):
    """HIGH TP + HIGH CP → Should NOT do_nothing
    
    Literature: Grothmann & Reusswig (2006) - People at Risk of Flooding
    PMT theory predicts: HIGH threat + HIGH coping → Protection Motivation → Action
    Choosing do_nothing in this case contradicts PMT predictions.
    """
    rule_id = "R1"
    description = "HIGH threat + HIGH coping should lead to protective action (Grothmann & Reusswig, 2006)"
    eligible_agent_types = {"household_owner", "household_renter"}
    severity = "warning"
    literature = "Grothmann, T., & Reusswig, F. (2006). People at risk of flooding. Natural Hazards, 38, 101-120."
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        if output.tp_level == "HIGH" and output.cp_level == "HIGH":
            if output.decision_skill == "do_nothing":
                return "HIGH threat + HIGH coping but chose do_nothing - contradicts PMT (Grothmann & Reusswig, 2006)"
        return None


class R2_LowTPAction(ValidationRule):
    """LOW TP → Taking action is notable (precautionary)
    
    Literature: Rogers (1983) - Protection Motivation Theory
    LOW threat perception typically leads to no action. Taking action despite low threat
    indicates precautionary behavior, which is acceptable but notable.
    """
    rule_id = "R2"
    description = "LOW threat but taking protective action (precautionary, Rogers 1983)"
    eligible_agent_types = {"household_owner", "household_renter"}
    severity = "warning"
    literature = "Rogers, R.W. (1983). Cognitive and physiological processes in fear appeals. Social Psychophysiology."
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        if output.tp_level == "LOW" and output.decision_skill != "do_nothing":
            return "LOW threat but took protective action - precautionary behavior (acceptable per Rogers, 1983)"
        return None


class R3_NoRedundantElevation(ValidationRule):
    """Already elevated or FULL PA → Cannot elevate again
    
    Literature: FEMA guidelines - One-time structural mitigation
    Elevation is a permanent, one-time action. Cannot be repeated.
    """
    rule_id = "R3"
    description = "Prevent redundant elevation (FEMA structural mitigation guidelines)"
    eligible_agent_types = {"household_owner"}  # Only owners can elevate
    severity = "error"
    literature = "FEMA (2022). Hazard Mitigation Assistance Program and Policy Guide."
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        if output.decision_skill == "elevate_house":
            if state.get("elevated", False):
                return "Cannot elevate - house already elevated (once-only per FEMA)"
            if output.pa_level == "FULL":
                return "PA=FULL indicates full protection - elevation unnecessary"
        return None


class R4_RenterNoElevate(ValidationRule):
    """Renter → Cannot elevate_house or apply for buyout
    
    Literature: Property rights - Structural modifications require ownership
    Renters do not own property and cannot make structural changes or sell to government.
    """
    rule_id = "R4"
    description = "Renters cannot modify property they don't own"
    eligible_agent_types = {"household_renter"}
    severity = "error"
    literature = "FEMA HMGP eligibility requirements - property ownership required."
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        owner_only_actions = ["elevate_house", "buyout_program"]
        if output.decision_skill in owner_only_actions:
            return f"Renters cannot {output.decision_skill} - requires property ownership"
        return None


class R5_AffordabilityCheck(ValidationRule):
    """LOW CP + HIGH cost action → Warning
    
    Literature: Bamberg et al. (2017) Meta-analysis
    Coping appraisal (r=0.30) is stronger predictor than threat appraisal (r=0.23).
    LOW CP choosing high-cost action is inconsistent with PMT.
    """
    rule_id = "R5"
    description = "Low coping with high-cost action is inconsistent (Bamberg et al., 2017)"
    eligible_agent_types = {"household_owner", "household_renter"}
    severity = "warning"
    literature = "Bamberg, S., et al. (2017). Threat, coping and flood prevention. J Environmental Psychology, 54, 116-126."
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        high_cost_actions = ["elevate_house", "relocate", "buyout_program"]
        if output.cp_level == "LOW" and output.decision_skill in high_cost_actions:
            return f"LOW coping but chose {output.decision_skill} - inconsistent with PMT (Bamberg et al., 2017)"
        return None


class R6_RelocatedNoAction(ValidationRule):
    """Already relocated → Cannot take any action
    
    Literature: Simulation logic - relocated agents exit the flood zone
    """
    rule_id = "R6"
    description = "Relocated agents cannot take further flood actions"
    eligible_agent_types = {"household_owner", "household_renter"}
    severity = "error"
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        if state.get("relocated", False) and output.decision_skill != "do_nothing":
            return "Already relocated - cannot take further flood actions"
        return None


class R7_FullPARelocate(ValidationRule):
    """FULL PA → Relocating/Buyout may be overreacting
    
    Literature: Grothmann & Reusswig (2006) - Non-protective responses
    When fully protected, extreme actions (leaving) suggest non-protective response patterns.
    """
    rule_id = "R7"
    description = "Full protection but choosing to leave (Grothmann & Reusswig, 2006)"
    eligible_agent_types = {"household_owner", "household_renter"}
    severity = "warning"
    literature = "Grothmann & Reusswig (2006) - Non-protective response patterns"
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        leave_actions = ["relocate", "buyout_program"]
        if output.pa_level == "FULL" and output.decision_skill in leave_actions:
            return f"PA=FULL but chose {output.decision_skill} - may be overreacting or non-protective response"
        return None


class R8_PAConsistency(ValidationRule):
    """PA level should match actual state
    
    Literature: LLM accuracy check - ensuring model understands current protection status
    """
    rule_id = "R8"
    description = "PA assessment should match agent's actual protection status"
    eligible_agent_types = {"household_owner", "household_renter"}
    severity = "warning"
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        actual_pa = "NONE"
        if state.get("elevated") and state.get("has_insurance"):
            actual_pa = "FULL"
        elif state.get("elevated") or state.get("has_insurance"):
            actual_pa = "PARTIAL"
        
        if output.pa_level != actual_pa:
            return (f"PA={output.pa_level} doesn't match actual state "
                   f"(elevated={state.get('elevated')}, insured={state.get('has_insurance')}) "
                   f"-> expected {actual_pa}")
        return None


class R9_RenterValidActions(ValidationRule):
    """Renter can only: buy_contents_insurance, relocate, do_nothing
    
    Literature: NFIP - Contents-only coverage for renters
    """
    rule_id = "R9"
    description = "Renters have limited action options (NFIP contents-only coverage)"
    eligible_agent_types = {"household_renter"}
    severity = "error"
    literature = "FEMA NFIP - Renters may purchase contents-only coverage."
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        valid_actions = ["buy_contents_insurance", "relocate", "do_nothing"]
        if output.decision_skill not in valid_actions:
            return f"Renter cannot {output.decision_skill} - valid actions: {', '.join(valid_actions)}"
        return None


class R10_OwnerValidActions(ValidationRule):
    """Owner valid actions check
    
    Owners can: buy_insurance, elevate_house (if not elevated), buyout_program, do_nothing
    """
    rule_id = "R10"
    description = "Owners have specific action options"
    eligible_agent_types = {"household_owner"}
    severity = "error"
    
    def check(self, output: HouseholdOutput, state: Dict) -> Optional[str]:
        valid_actions = ["buy_insurance", "elevate_house", "buyout_program", "do_nothing"]
        if output.decision_skill not in valid_actions:
            return f"Owner cannot {output.decision_skill} - valid actions: {', '.join(valid_actions)}"
        return None


# =============================================================================
# INSURANCE RULES
# =============================================================================

class RI1_LossRatioRaise(ValidationRule):
    """Loss ratio > 1 but lowering premium is risky"""
    rule_id = "RI1"
    description = "High loss ratio should lead to premium increase"
    eligible_agent_types = {"insurance"}
    severity = "warning"
    
    def check(self, output: InsuranceOutput, state: Dict) -> Optional[str]:
        loss_ratio = state.get("loss_ratio", 0)
        if loss_ratio > 1.0 and output.decision == "LOWER":
            return "Loss ratio > 1 but lowering premium - risky for solvency"
        return None


class RI2_LowLossRatioRaise(ValidationRule):
    """Very low loss ratio but raising premium may lose customers"""
    rule_id = "RI2"
    description = "Low loss ratio with premium increase may reduce uptake"
    eligible_agent_types = {"insurance"}
    severity = "warning"
    
    def check(self, output: InsuranceOutput, state: Dict) -> Optional[str]:
        loss_ratio = state.get("loss_ratio", 0)
        if loss_ratio < 0.3 and output.decision == "RAISE":
            return "Very low loss ratio but raising premium - may lose customers"
        return None


# =============================================================================
# GOVERNMENT RULES
# =============================================================================

class RG1_LowBudgetIncrease(ValidationRule):
    """Low budget remaining but increasing subsidy"""
    rule_id = "RG1"
    description = "Budget constraint check for subsidy increase"
    eligible_agent_types = {"government"}
    severity = "warning"
    
    def check(self, output: GovernmentOutput, state: Dict) -> Optional[str]:
        budget_pct = state.get("budget_remaining_pct", 1.0)
        if budget_pct < 0.2 and output.decision == "INCREASE":
            return "Less than 20% budget remaining but increasing subsidy"
        return None


class RG2_LowMGAdoption(ValidationRule):
    """Low MG adoption with ample budget should prioritize MG"""
    rule_id = "RG2"
    description = "Equity concern for MG group"
    eligible_agent_types = {"government"}
    severity = "warning"
    
    def check(self, output: GovernmentOutput, state: Dict) -> Optional[str]:
        mg_adoption = state.get("mg_adoption", 0)
        budget_pct = state.get("budget_remaining_pct", 1.0)
        if mg_adoption < 0.2 and budget_pct > 0.5:
            return "Low MG adoption with ample budget - consider MG priority"
        return None


# =============================================================================
# RULE REGISTRY
# =============================================================================

class RuleRegistry:
    """
    Central registry for validation rules.
    
    Features:
    - Register rules by agent type
    - Get applicable rules for agent
    - Run all rules and aggregate results
    """
    
    def __init__(self):
        self._rules: List[ValidationRule] = []
    
    def register(self, rule: ValidationRule) -> 'RuleRegistry':
        """Register a rule. Returns self for chaining."""
        self._rules.append(rule)
        return self
    
    def register_all(self, rules: List[ValidationRule]) -> 'RuleRegistry':
        """Register multiple rules."""
        self._rules.extend(rules)
        return self
    
    def get_rules(self, agent_type: str) -> List[ValidationRule]:
        """Get all rules applicable to an agent type."""
        return [r for r in self._rules if r.applies_to(agent_type)]
    
    def validate(self, agent_type: str, output: Any, state: Dict) -> ValidationResult:
        """
        Run all applicable rules and return aggregated result.
        
        Args:
            agent_type: Type of agent (household_owner, household_renter, etc.)
            output: Parsed LLM output
            state: Agent's current state
            
        Returns:
            ValidationResult with all errors, warnings, and rule checks
        """
        errors = []
        warnings = []
        rule_checks = {}
        
        for rule in self.get_rules(agent_type):
            message = rule.check(output, state)
            passed = message is None
            rule_checks[rule.rule_id] = passed
            
            if not passed:
                if rule.severity == "error":
                    errors.append(f"{rule.rule_id}: {message}")
                else:
                    warnings.append(f"{rule.rule_id}: {message}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            rule_checks=rule_checks
        )
    
    def list_rules(self) -> Dict[str, List[str]]:
        """List all rules grouped by agent type."""
        result = {}
        for rule in self._rules:
            for agent_type in rule.eligible_agent_types:
                if agent_type not in result:
                    result[agent_type] = []
                result[agent_type].append(f"{rule.rule_id}: {rule.description}")
        return result


# =============================================================================
# DEFAULT REGISTRY (Pre-configured)
# =============================================================================

def create_default_registry() -> RuleRegistry:
    """Create registry with all default rules."""
    registry = RuleRegistry()
    
    # Household rules
    registry.register_all([
        R1_HighTPCPDoNothing(),
        R2_LowTPAction(),
        R3_NoRedundantElevation(),
        R4_RenterNoElevate(),
        R5_AffordabilityCheck(),
        R6_RelocatedNoAction(),
        R7_FullPARelocate(),
        R8_PAConsistency(),
        R9_RenterValidActions(),
        R10_OwnerValidActions(),
    ])
    
    # Insurance rules
    registry.register_all([
        RI1_LossRatioRaise(),
        RI2_LowLossRatioRaise(),
    ])
    
    # Government rules
    registry.register_all([
        RG1_LowBudgetIncrease(),
        RG2_LowMGAdoption(),
    ])
    
    return registry


# =============================================================================
# CONVENIENCE VALIDATORS (Backward compatible)
# =============================================================================

class HouseholdValidator:
    """Convenience wrapper for household validation."""
    
    def __init__(self, registry: Optional[RuleRegistry] = None):
        self.registry = registry or create_default_registry()
    
    def validate(self, output: HouseholdOutput, state: Dict) -> ValidationResult:
        """Validate household output."""
        agent_type = f"household_{output.tenure.lower()}"
        return self.registry.validate(agent_type, output, state)


class InstitutionalValidator:
    """Convenience wrapper for institutional agent validation."""
    
    def __init__(self, registry: Optional[RuleRegistry] = None):
        self.registry = registry or create_default_registry()
    
    def validate_insurance(self, output: InsuranceOutput, state: Dict) -> ValidationResult:
        """Validate insurance output."""
        return self.registry.validate("insurance", output, state)
    
    def validate_government(self, output: GovernmentOutput, state: Dict) -> ValidationResult:
        """Validate government output."""
        return self.registry.validate("government", output, state)
