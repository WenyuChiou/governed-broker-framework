"""
Experiment 3: Multi-Agent Simulation (Main Script)

Runs the annual simulation loop for the Governed Broker Framework (Exp3).
Coordinates:
- 100 Household Agents (Diverse types)
- 1 Insurance Agent
- 1 Government Agent
- Environment Modules (Catastrophe, Customs, Settlement)

Output:
- Simulation log CSV
- Summary metrics
"""

import sys
import os
import random
import csv
import pandas as pd
from typing import List, Dict

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

def initialize_agents(seed: int = 42) -> (List[HouseholdAgent], GovernmentAgent, InsuranceAgent):
    """Initializes all agents with a specific distribution."""
    random.seed(seed)
    
    # Government & Insurance
    gov = GovernmentAgent()
    ins = InsuranceAgent()
    
    # Households
    households = []
    
    # Distribution: 
    # 30% MG Owner, 20% MG Renter
    # 40% NMG Owner, 10% NMG Renter
    distribution = [
        ("MG_Owner", 30), ("MG_Renter", 20),
        ("NMG_Owner", 40), ("NMG_Renter", 10)
    ]
    
    count = 0
    for agent_type, num in distribution:
        for _ in range(num):
            count += 1
            # Simple attribute randomization
            if "Owner" in agent_type:
                prop_val = random.gauss(300_000, 50_000)
                income = random.gauss(60_000, 15_000)
            else:
                prop_val = 0 # Renter doesn't own structure
                income = random.gauss(40_000, 10_000)
                
            if "MG" in agent_type:
                income *= 0.7 # Lower income for MG on average
                prop_val *= 0.8
                
            h = HouseholdAgent(f"H{count:03d}", agent_type, income, prop_val)
            households.append(h)
            
    return households, gov, ins

def run_simulation():
    """Main simulation loop."""
    print(f"Starting Exp3 Simulation ({SIMULATION_YEARS} years, {NUM_HOUSEHOLDS} households)...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize
    households, gov, ins = initialize_agents(SEED)
    settlement = SettlementModule(seed=SEED)
    
    # Data Logging
    data_log = []
    
    for year in range(1, SIMULATION_YEARS + 1):
        print(f"\n--- Year {year} ---")
        
        # Track Flood Status (from previous year for logic, or current year predictions?)
        # Logic: Phase 1 & 2 happen BEFORE the flood event of the current year is known/realized?
        # Usually: Plan -> Event -> Settle.
        # So "flood_occurred_prev_year" checks year-1.
        
        # Need history from previous settlement report?
        prev_flood = False
        if year > 1 and len(data_log) > 0:
            # Check last entry for flood status? Or store in variable.
            # actually we don't have easy access here unless we store it.
            pass # (Logic inside agents handles memory)
            
        # ==========================================
        # Phase 1: Institutional Decisions
        # ==========================================
        gov.reset_annual_budget(year)
        ins.reset_annual_metrics()
        
        # Decisions (Heuristic/LLM)
        # Pass simpler context for now
        gov_decision = gov.decide_policy(year, flood_occurred_prev_year=False) # Simplified arg
        ins_decision = ins.decide_strategy(year)
        
        print(f"Gov Policy: {gov_decision} (Subsidy: {gov.state.subsidy_rate:.0%})")
        print(f"Ins Strategy: {ins_decision} (Premium: {ins.state.premium_rate:.1%})")
        
        # ==========================================
        # Phase 2: Household Decisions
        # ==========================================
        # Context for households
        context = {
            "government_subsidy_rate": gov.state.subsidy_rate,
            "insurance_premium_rate": ins.state.premium_rate,
            "year": year
        }
        
        actions = {"do_nothing": 0, "buy_insurance": 0, "elevate_house": 0, "relocate": 0}
        
        for hh in households:
            if hh.state.relocated:
                continue # Skip relocated agents
                
            decision = hh.make_decision(year, context)
            actions[decision] += 1
            
            # Application Logic (incorporating Subsidy check for Elevation)
            if decision == "elevate_house":
                # Check subsidy
                subsidy_res = settlement.process_mitigation(hh, "elevate_house", gov)
                if subsidy_res['approved']:
                    hh.apply_decision(decision, year)
                else:
                    # Failed to elevate due to budget? Fallback?
                    hh.apply_decision("do_nothing", year) # Or retry
                    actions["elevate_house"] -= 1
                    actions["do_nothing"] += 1
            elif decision == "buy_insurance":
                # Check affordability? (Skip for now)
                hh.apply_decision(decision, year)
            else:
                hh.apply_decision(decision, year)
                
        print(f"Household Actions: {actions}")
        
        # Update metrics for Government (Phase 1 next year needs this)
        # Simple Adoption Rates
        total_mg = sum(1 for h in households if "MG" in h.state.agent_type)
        mg_elevated = sum(1 for h in households if "MG" in h.state.agent_type and h.state.elevated)
        
        total_nmg = NUM_HOUSEHOLDS - total_mg
        nmg_elevated = sum(1 for h in households if "NMG" in h.state.agent_type and h.state.elevated)
        
        gov.update_metrics(
            mg_adopt=mg_elevated/total_mg if total_mg else 0,
            nmg_adopt=nmg_elevated/total_nmg if total_nmg else 0
        )
        
        # ==========================================
        # Phase 3: Annual Settlement
        # ==========================================
        report = settlement.process_year(year, households, ins, gov)
        
        if report.flood_occurred:
            print(f"🌊 FLOOD! Severity: {report.flood_severity:.2f}, Damage: ${report.total_damage:,.0f}, Claims: ${report.total_claims:,.0f}")
        else:
            print("No Flood.")
            
        # ==========================================
        # Logging
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
            "Gov_SubsidyRate": gov.state.subsidy_rate,
            "Gov_BudgetRem": gov.state.budget_remaining,
            "Actions_Elevate": actions["elevate_house"],
            "Actions_Insurance": actions["buy_insurance"],
            "Actions_Relocate": actions["relocate"]
        }
        data_log.append(log_entry)

    # Save Results
    df = pd.DataFrame(data_log)
    csv_path = os.path.join(OUTPUT_DIR, "simulation_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation Complete. Results saved to {csv_path}")
    print(df)

if __name__ == "__main__":
    run_simulation()
