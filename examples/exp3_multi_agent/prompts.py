"""
Multi-Agent Prompt Templates (Exp3)

Enhanced prompts for:
- Household Agent: PMT 5 Constructs output
- Insurance Agent: Premium decisions
- Government Agent: Subsidy policy decisions

Based on single-agent design but with explicit construct-level outputs.
"""

from typing import Dict, Any, List

# =============================================================================
# HOUSEHOLD AGENT PROMPT
# =============================================================================

HOUSEHOLD_PROMPT = """You are a {tenure_type} in a flood-prone city{mg_context}.
You have {community_attachment}.

=== YOUR SITUATION ===
{elevation_status}
{insurance_status}
{damage_history}

=== YOUR PERCEPTIONS ===
{trust_context}

=== YOUR MEMORY ===
{memory}

=== CURRENT CONDITIONS ===
{flood_status}
{subsidy_status}
{neighbor_actions}

=== TASK ===
Using Protection Motivation Theory, evaluate your situation across FIVE constructs:

1. **TP (Threat Perception)**: How serious and likely is flood damage?
2. **CP (Coping Perception)**: Can you afford protective actions?
3. **SP (Subsidy Perception)**: How helpful is government support?
4. **SC (Self-Confidence)**: Do you believe you can take effective action?
5. **PA (Prior Adaptation)**: What protections do you already have?

Then choose ONE action:
{options}

=== OUTPUT FORMAT ===
Respond using EXACTLY this format (no markdown):

TP Assessment: [LOW/MODERATE/HIGH] - [One sentence explaining why]
CP Assessment: [LOW/MODERATE/HIGH] - [One sentence explaining why]
SP Assessment: [LOW/MODERATE/HIGH] - [One sentence explaining why]
SC Assessment: [LOW/MODERATE/HIGH] - [One sentence explaining why]
PA Assessment: [NONE/PARTIAL/FULL] - [One sentence explaining current protections]
Final Decision: [Number only: {valid_choices}]
"""

# Options for non-elevated owners
OPTIONS_OWNER_NON_ELEVATED = """1. Buy flood insurance (Annual cost, partial financial protection)
2. Elevate your house (High upfront cost, government subsidy available, prevents most damage)
3. Relocate (Leave neighborhood, eliminates all flood risk permanently)
4. Do nothing (No cost this year, but remain exposed to flood damage)"""

# Options for elevated owners
OPTIONS_OWNER_ELEVATED = """1. Buy flood insurance (Annual cost, additional financial protection)
2. Relocate (Leave neighborhood, eliminates all flood risk permanently)
3. Do nothing (Your house is protected by elevation)"""

# Options for renters (cannot elevate)
OPTIONS_RENTER = """1. Buy renter's flood insurance (Annual cost, protects belongings)
2. Relocate (Move to a safer area)
3. Do nothing (No cost, but belongings remain at risk)"""

# Context templates
MG_CONTEXT = {
    True: ", belonging to a marginalized community with limited financial resources",
    False: ""
}

TENURE_CONTEXT = {
    "Owner": "homeowner",
    "Renter": "renter"
}

ELEVATION_STATUS = {
    True: "Your house is ELEVATED, providing substantial flood protection.",
    False: "Your house is NOT elevated and vulnerable to flooding."
}

INSURANCE_STATUS = {
    True: "You currently HAVE flood insurance coverage.",
    False: "You currently DO NOT have flood insurance."
}

TRUST_TEMPLATES = {
    "high": "have high confidence in",
    "moderate": "have moderate trust in", 
    "low": "have some doubts about",
    "very_low": "strongly distrust"
}


def verbalize_trust(value: float) -> str:
    """Convert numeric trust to natural language."""
    if value >= 0.7:
        return "high"
    elif value >= 0.5:
        return "moderate"
    elif value >= 0.3:
        return "low"
    return "very_low"


def build_household_prompt(agent_state: Dict[str, Any], 
                           context: Dict[str, Any],
                           memory: List[str]) -> str:
    """Build prompt for Household Agent."""
    
    # Demographics
    mg = agent_state.get("mg", False)
    tenure = agent_state.get("tenure", "Owner")
    elevated = agent_state.get("elevated", False)
    has_insurance = agent_state.get("has_insurance", False)
    
    # Trust
    trust_gov = agent_state.get("trust_in_government", 0.5)
    trust_ins = agent_state.get("trust_in_insurance", 0.5)
    trust_neighbors = agent_state.get("trust_in_neighbors", 0.5)
    
    # Damage
    cumulative_damage = agent_state.get("cumulative_damage", 0)
    property_value = agent_state.get("property_value", 300000)
    
    # Context
    subsidy_rate = context.get("government_subsidy_rate", 0.5)
    flood_occurred = context.get("flood_occurred", False)
    year = context.get("year", 1)
    
    # Build components
    mg_context = MG_CONTEXT[mg]
    tenure_type = TENURE_CONTEXT[tenure]
    elevation_status = ELEVATION_STATUS[elevated]
    insurance_status = INSURANCE_STATUS[has_insurance]
    
    # Damage history
    if cumulative_damage > 0:
        damage_pct = cumulative_damage / property_value * 100 if property_value > 0 else 0
        damage_history = f"You have experienced ${cumulative_damage:,.0f} in flood damage ({damage_pct:.1f}% of property value)."
    else:
        damage_history = "You have not experienced significant flood damage."
    
    # Trust context
    trust_gov_text = TRUST_TEMPLATES[verbalize_trust(trust_gov)]
    trust_ins_text = TRUST_TEMPLATES[verbalize_trust(trust_ins)]
    trust_context = f"You {trust_gov_text} the government. You {trust_ins_text} the insurance company."
    
    # Memory formatting
    if memory:
        memory_text = "\n".join(f"- {m}" for m in memory[-5:])
    else:
        memory_text = "- No significant past events recalled."
    
    # Flood status
    if flood_occurred:
        flood_status = f"⚠️ A flood occurred this year (Year {year}), causing damage in the area."
    else:
        flood_status = f"No flood occurred this year (Year {year})."
    
    # Subsidy status
    subsidy_status = f"Government offers {subsidy_rate:.0%} subsidy for home elevation."
    if mg:
        subsidy_status += " (Priority support available for marginalized groups)"
    
    # Neighbor actions
    neighbor_actions = "Some neighbors have been considering protective measures."
    
    # Options based on tenure and elevation
    if tenure == "Renter":
        options = OPTIONS_RENTER
        valid_choices = "1, 2, or 3"
    elif elevated:
        options = OPTIONS_OWNER_ELEVATED
        valid_choices = "1, 2, or 3"
    else:
        options = OPTIONS_OWNER_NON_ELEVATED
        valid_choices = "1, 2, 3, or 4"
    
    # Community attachment
    community_attachment = "strong attachment to your community"
    if mg:
        community_attachment = "deep roots in your community despite economic challenges"
    
    return HOUSEHOLD_PROMPT.format(
        tenure_type=tenure_type,
        mg_context=mg_context,
        community_attachment=community_attachment,
        elevation_status=elevation_status,
        insurance_status=insurance_status,
        damage_history=damage_history,
        trust_context=trust_context,
        memory=memory_text,
        flood_status=flood_status,
        subsidy_status=subsidy_status,
        neighbor_actions=neighbor_actions,
        options=options,
        valid_choices=valid_choices
    )


# =============================================================================
# INSURANCE AGENT PROMPT
# =============================================================================

INSURANCE_PROMPT = """You are the risk management AI for an insurance company.

=== CURRENT METRICS ===
- Loss Ratio: {loss_ratio:.1%} (Claims / Premiums)
- Active Policies: {total_policies}
- Risk Pool: ${risk_pool:,.0f}
- Premium Rate: {premium_rate:.2%}

=== RECENT HISTORY ===
{history}

=== TASK ===
Decide on premium adjustment for the next year.

Options:
1. RAISE premium (increase revenue, may reduce uptake)
2. LOWER premium (attract more customers, increase risk exposure)
3. MAINTAIN current premium

=== OUTPUT FORMAT ===
Analysis: [One sentence about current risk situation]
Decision: [RAISE/LOWER/MAINTAIN]
Adjustment: [Percentage, e.g., 5% or 0%]
Reason: [Brief explanation]
"""

def build_insurance_prompt(state: Dict[str, Any], history: List[str]) -> str:
    """Build prompt for Insurance Agent."""
    history_text = "\n".join(f"- {h}" for h in history[-3:]) if history else "- No recent claims history."
    
    return INSURANCE_PROMPT.format(
        loss_ratio=state.get("loss_ratio", 0),
        total_policies=state.get("total_policies", 0),
        risk_pool=state.get("risk_pool", 1000000),
        premium_rate=state.get("premium_rate", 0.05),
        history=history_text
    )


# =============================================================================
# GOVERNMENT AGENT PROMPT
# =============================================================================

GOVERNMENT_PROMPT = """You are a government policy advisor managing flood mitigation programs.

=== CURRENT STATUS ===
- Annual Budget: ${annual_budget:,.0f}
- Budget Remaining: ${budget_remaining:,.0f}
- Current Subsidy Rate: {subsidy_rate:.0%}
- MG Priority: {mg_priority}

=== ADOPTION METRICS ===
- MG Adoption Rate: {mg_adoption:.1%}
- NMG Adoption Rate: {nmg_adoption:.1%}

=== RECENT EVENTS ===
{events}

=== TASK ===
Decide on subsidy policy for the next year.

Options:
1. INCREASE subsidy rate (more support, faster budget depletion)
2. DECREASE subsidy rate (conserve budget, slower adoption)
3. MAINTAIN current rate

=== OUTPUT FORMAT ===
Analysis: [One sentence about current adoption situation]
Decision: [INCREASE/DECREASE/MAINTAIN]
Adjustment: [Percentage change, e.g., 10% or 0%]
Priority: [MG/ALL]
Reason: [Brief explanation]
"""

def build_government_prompt(state: Dict[str, Any], events: List[str]) -> str:
    """Build prompt for Government Agent."""
    events_text = "\n".join(f"- {e}" for e in events[-3:]) if events else "- No major events."
    
    return GOVERNMENT_PROMPT.format(
        annual_budget=state.get("annual_budget", 500000),
        budget_remaining=state.get("budget_remaining", 500000),
        subsidy_rate=state.get("subsidy_rate", 0.5),
        mg_priority="Enabled" if state.get("mg_priority", True) else "Disabled",
        mg_adoption=state.get("mg_adoption_rate", 0),
        nmg_adoption=state.get("nmg_adoption_rate", 0),
        events=events_text
    )
