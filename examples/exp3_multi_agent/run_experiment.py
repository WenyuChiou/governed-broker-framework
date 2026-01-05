"""
Experiment 3: Multi-Agent Simulation (Main Script)

Runs the annual simulation loop for the Governed Broker Framework (Exp3).
Coordinates:
- 100 Household Agents (Diverse types: MG/NMG × Owner/Renter)
- 1 Insurance Agent
- 2 Government Agents (NJ, NY)
- Environment Modules (Catastrophe, Subsidy, Settlement)

Output:
- Simulation log CSV (Annual summary)
- Agent decision log JSONL (Per-agent decisions with PMT constructs)
"""

import sys
import os
import random
import json
import pandas as pd
from typing import List, Dict
from dataclasses import asdict

# Ensure path
sys.path.insert(0, '.')
sys.path.insert(0, './examples/exp3_multi_agent')

from examples.exp3_multi_agent.agents import (
    HouseholdAgent, InsuranceAgent, GovernmentAgent
)
from examples.exp3_multi_agent.environment import (
    SettlementModule, SettlementReport
)

# Configuration
NUM_HOUSEHOLDS = 100
SIMULATION_YEARS = 10
OUTPUT_DIR = "examples/exp3_multi_agent/results"
SEED = 42

def initialize_agents(seed: int = 42) -> (List[HouseholdAgent], Dict[str, GovernmentAgent], InsuranceAgent):
    """Initializes all agents with a specific distribution."""
    random.seed(seed)
    
    # Multi-Government (NJ, NY)
    govs = {
        "NJ": GovernmentAgent("Gov_NJ"),
        "NY": GovernmentAgent("Gov_NY")
    }
    # NY has higher budget
    govs["NY"].state.annual_budget = 600_000
    govs["NY"].state.budget_remaining = 600_000
    
    # Insurance (Single)
    ins = InsuranceAgent()
    
    # Households
    households = []
    
    # Distribution: 
    # 30% MG Owner, 20% MG Renter
    # 40% NMG Owner, 10% NMG Renter
    # Split 60/40 between NJ/NY
    distribution = [
        (True, "Owner", 30),   # MG Owner
        (True, "Renter", 20),  # MG Renter
        (False, "Owner", 40),  # NMG Owner
        (False, "Renter", 10)  # NMG Renter
    ]
    
    count = 0
    for mg, tenure, num in distribution:
        for i in range(num):
            count += 1
            # Simple attribute randomization
            if tenure == "Owner":
                prop_val = random.gauss(300_000, 50_000)
                income = random.gauss(60_000, 15_000)
            else:
                prop_val = 0 # Renter doesn't own structure
                income = random.gauss(40_000, 10_000)
                
            if mg:
                income *= 0.7 # Lower income for MG on average
                prop_val *= 0.8
            
            # Assign region (60% NJ, 40% NY)
            region = "NJ" if i % 5 < 3 else "NY"
                
            h = HouseholdAgent(
                agent_id=f"H{count:03d}", 
                mg=mg, 
                tenure=tenure, 
                income=income, 
                property_value=prop_val,
                region_id=region
            )
            households.append(h)
            
    return households, govs, ins

def run_simulation():
    """Main simulation loop."""
    print(f"Starting Exp3 Simulation ({SIMULATION_YEARS} years, {NUM_HOUSEHOLDS} households)...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize
    households, govs, ins = initialize_agents(SEED)
    settlement = SettlementModule(seed=SEED)
    
    # Data Logging
    annual_log = []
    decision_log = []  # Per-agent decisions
    
    for year in range(1, SIMULATION_YEARS + 1):
        print(f"\n--- Year {year} ---")
        
        # ==========================================
        # Phase 0: Annual Reset
        # ==========================================
        for hh in households:
            hh.reset_insurance()  # Insurance NOT cumulative
        
        # ==========================================
        # Phase 1: Institutional Decisions
        # ==========================================
        for gov_id, gov in govs.items():
            gov.reset_annual_budget(year)
            gov_decision = gov.decide_policy(year, flood_occurred_prev_year=False)
            print(f"{gov_id} Policy: {gov_decision} (Subsidy: {gov.state.subsidy_rate:.0%})")
            
        ins.reset_annual_metrics()
        ins_decision = ins.decide_strategy(year)
        print(f"Ins Strategy: {ins_decision} (Premium: {ins.state.premium_rate:.1%})")
        
        # ==========================================
        # Phase 2: Household Decisions
        # ==========================================
        actions = {"do_nothing": 0, "buy_insurance": 0, "elevate_house": 0, "relocate": 0}
        
        for hh in households:
            if hh.state.relocated:
                continue # Skip relocated agents
            
            # Get region-specific government
            gov = govs.get(hh.state.region_id, govs["NJ"])
            
            # Context for households
            context = {
                "government_subsidy_rate": gov.state.subsidy_rate,
                "insurance_premium_rate": ins.state.premium_rate,
                "year": year
            }
            
            output = hh.make_decision(year, context)
            decision = output.decision_skill
            actions[decision] += 1
            
            # Log decision with PMT constructs
            decision_log.append(asdict(output))
            
            # Application Logic
            if decision == "elevate_house":
                subsidy_res = settlement.process_mitigation(hh, "elevate_house", gov)
                if subsidy_res['approved']:
                    hh.apply_decision(output)
                else:
                    # Fallback to do_nothing
                    output.decision_skill = "do_nothing"
                    output.decision_number = 4
                    hh.apply_decision(output)
                    actions["elevate_house"] -= 1
                    actions["do_nothing"] += 1
            else:
                hh.apply_decision(output)
                
        print(f"Household Actions (This Year): {actions}")
        
        # ==========================================
        # Cumulative Stats
        # ==========================================
        total_insured = sum(1 for h in households if h.state.has_insurance)
        total_elevated = sum(1 for h in households if h.state.elevated)
        total_relocated = sum(1 for h in households if h.state.relocated)
        print(f"Cumulative: Insured={total_insured}, Elevated={total_elevated}, Relocated={total_relocated}")
        
        # ==========================================
        # Phase 3: Annual Settlement
        # ==========================================
        # Use NJ gov as primary (or could aggregate)
        report = settlement.process_year(year, households, ins, govs["NJ"])
        
        if report.flood_occurred:
            print(f"🌊 FLOOD! Severity: {report.flood_severity:.2f}, Damage: ${report.total_damage:,.0f}, Claims: ${report.total_claims:,.0f}")
        else:
            print("No Flood.")
            
        # ==========================================
        # Annual Logging
        # ==========================================
        log_entry = {
            "Year": year,
            "Flood": report.flood_occurred,
            "Severity": report.flood_severity,
            "TotalDamage": report.total_damage,
            "TotalClaims": report.total_claims,
            "Ins_PremiumRate": ins.state.premium_rate,
            "Ins_LossRatio": ins.state.loss_ratio,
            "Ins_Policies": ins.state.total_policies,
            # Per Year Actions
            "Actions_Insurance": actions["buy_insurance"],
            "Actions_Elevate": actions["elevate_house"],
            "Actions_Relocate": actions["relocate"],
            # Cumulative Stats
            "Cum_Insured": total_insured,
            "Cum_Elevated": total_elevated,
            "Cum_Relocated": total_relocated
        }
        annual_log.append(log_entry)

    # ==========================================
    # Save Results
    # ==========================================
    
    # Annual Summary
    df = pd.DataFrame(annual_log)
    csv_path = os.path.join(OUTPUT_DIR, "simulation_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation Complete. Annual log saved to {csv_path}")
    print(df)
    
    # Decision Log (JSONL with PMT constructs)
    jsonl_path = os.path.join(OUTPUT_DIR, "decision_log.jsonl")
    with open(jsonl_path, 'w') as f:
        for entry in decision_log:
            f.write(json.dumps(entry) + '\n')
    print(f"Decision log saved to {jsonl_path} ({len(decision_log)} entries)")

if __name__ == "__main__":
    run_simulation()
