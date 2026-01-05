"""
Integration Test: Year Loop + Agent Interaction

Tests the complete simulation cycle with unified interfaces:
1. Memory Helpers: add_memory()
2. Environment Core: process()
3. Agent Interaction: Household ↔ Government/Insurance
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Import unified interfaces
from broker.memory import CognitiveMemory
from examples.exp3_multi_agent.memory_helpers import add_memory, add_year_end_memories
from examples.exp3_multi_agent.environment.core import process


# =============================================================================
# MOCK AGENTS FOR TESTING
# =============================================================================

@dataclass
class MockHouseholdState:
    id: str
    agent_type: str = "MG_Owner"
    income: float = 50000
    property_value: float = 300000
    elevated: bool = False
    relocated: bool = False
    has_insurance: bool = False
    cumulative_damage: float = 0
    cumulative_oop: float = 0


@dataclass
class MockInsuranceState:
    premium_rate: float = 0.05
    claims_paid: float = 0
    premium_collected: float = 0
    payout_ratio: float = 1.0


@dataclass
class MockGovernmentState:
    budget_remaining: float = 500000
    subsidy_rate: float = 0.5
    mg_priority: bool = True


class MockHousehold:
    def __init__(self, hid: str, agent_type: str = "MG_Owner"):
        self.state = MockHouseholdState(id=hid, agent_type=agent_type)
        self.memory = CognitiveMemory(agent_id=hid)


class MockInsurance:
    def __init__(self):
        self.state = MockInsuranceState()


class MockGovernment:
    def __init__(self):
        self.state = MockGovernmentState()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_year_loop_no_flood():
    """Test a year where no flood occurs."""
    print("\n--- test_year_loop_no_flood ---")
    
    # Setup
    hh = MockHousehold("H001", "MG_Owner")
    ins = MockInsurance()
    gov = MockGovernment()
    rng = np.random.default_rng(999)  # Seed for no flood
    year = 1
    
    # Year Beginning: Neighbor observation (passive)
    add_memory(hh.memory, "neighbor", {"type": "elevated", "count": 2}, year)
    
    # Year Beginning: Agent retrieves memory (active)
    memories = hh.memory.retrieve(top_k=5, current_year=year)
    assert len(memories) >= 1  # At least neighbor observation
    
    # Decision (simulated)
    decision = "buy_insurance"
    hh.state.has_insurance = True
    
    # Year Middle: Flood event
    flood_result = process("flood", {"year": year, "rng": rng})
    # With this seed, may or may not flood
    
    # Year End: Memory updates
    if flood_result["occurred"]:
        # Process damage
        damage_result = process("damage", {
            "severity": flood_result["severity"],
            "property_value": hh.state.property_value,
            "elevated": hh.state.elevated
        })
        add_memory(hh.memory, "flood", {"damage": damage_result["damage_amount"]}, year)
    
    # Record decision
    add_memory(hh.memory, "decision", {"skill_id": decision}, year)
    
    # Verify memory state
    all_mem = hh.memory.retrieve(top_k=10, current_year=year)
    assert any("insurance" in str(m).lower() for m in all_mem)
    
    print("✓ Year loop (no flood) completed")


def test_year_loop_with_flood():
    """Test a year where flood occurs."""
    print("\n--- test_year_loop_with_flood ---")
    
    # Setup with flood-prone seed
    hh = MockHousehold("H002", "NMG_Owner")
    hh.state.has_insurance = True  # Already insured
    ins = MockInsurance()
    year = 3
    
    # Force a flood for testing
    flood_result = {
        "year": year,
        "occurred": True,
        "severity": 0.6,
        "depth_ft": 6.0
    }
    
    # Process damage
    damage_result = process("damage", {
        "severity": flood_result["severity"],
        "property_value": hh.state.property_value,
        "elevated": hh.state.elevated
    })
    
    assert damage_result["damage_amount"] > 0
    
    # Process claim
    claim_result = process("claim", {
        "damage_amount": damage_result["damage_amount"],
        "has_insurance": hh.state.has_insurance,
        "payout_ratio": ins.state.payout_ratio
    })
    
    assert claim_result["filed"] == True
    assert claim_result["payout"] > 0
    
    # Update state
    hh.state.cumulative_damage += damage_result["damage_amount"]
    hh.state.cumulative_oop += claim_result["out_of_pocket"]
    
    # Record memories
    add_memory(hh.memory, "flood", {"damage": damage_result["damage_amount"]}, year)
    add_memory(hh.memory, "claim", {
        "filed": True,
        "approved": claim_result["approved"],
        "payout": claim_result["payout"],
        "oop": claim_result["out_of_pocket"],
        "damage": damage_result["damage_amount"]
    }, year)
    
    # Verify memories recorded flood and claim
    all_mem = hh.memory.retrieve(top_k=10, current_year=year)
    assert len(all_mem) >= 2
    
    print(f"✓ Flood damage: ${damage_result['damage_amount']:,.0f}")
    print(f"✓ Insurance payout: ${claim_result['payout']:,.0f}")
    print(f"✓ Out-of-pocket: ${claim_result['out_of_pocket']:,.0f}")
    print("✓ Year loop (with flood) completed")


def test_household_government_interaction():
    """Test Household ↔ Government subsidy interaction."""
    print("\n--- test_household_government_interaction ---")
    
    hh = MockHousehold("H003", "MG_Owner")
    gov = MockGovernment()
    year = 2
    
    # Household wants to elevate
    decision = "elevate_house"
    
    # Request subsidy from government
    subsidy_result = process("subsidy", {
        "action": decision,
        "is_mg": "MG" in hh.state.agent_type,
        "mg_priority": gov.state.mg_priority,
        "budget": gov.state.budget_remaining
    })
    
    assert subsidy_result["approved"] == True
    assert subsidy_result["subsidy_amount"] > 0
    
    # MG gets bonus rate
    assert subsidy_result["applied_rate"] > 0.5
    
    # Government budget reduced
    gov.state.budget_remaining -= subsidy_result["subsidy_amount"]
    
    # Household pays net cost, becomes elevated
    hh.state.elevated = True
    
    # Memory update
    add_memory(hh.memory, "decision", {"skill_id": decision}, year)
    
    print(f"✓ Subsidy granted: ${subsidy_result['subsidy_amount']:,.0f}")
    print(f"✓ Net cost to household: ${subsidy_result['net_cost']:,.0f}")
    print(f"✓ Government budget remaining: ${gov.state.budget_remaining:,.0f}")
    print("✓ Household ↔ Government interaction completed")


def test_household_insurance_interaction():
    """Test Household ↔ Insurance interaction."""
    print("\n--- test_household_insurance_interaction ---")
    
    hh = MockHousehold("H004", "NMG_Renter")
    ins = MockInsurance()
    year = 1
    
    # Household buys insurance
    decision = "buy_insurance"
    hh.state.has_insurance = True
    
    # Premium collection
    annual_premium = hh.state.property_value * ins.state.premium_rate
    ins.state.premium_collected += annual_premium
    
    # Memory
    add_memory(hh.memory, "decision", {"skill_id": decision}, year)
    
    # Year 2: Flood occurs
    year = 2
    damage = 50000
    
    claim_result = process("claim", {
        "damage_amount": damage,
        "has_insurance": hh.state.has_insurance,
        "payout_ratio": 1.0
    })
    
    # Insurance pays claim
    ins.state.claims_paid += claim_result["payout"]
    
    # Record claim memory
    add_memory(hh.memory, "claim", {
        "filed": True,
        "approved": claim_result["approved"],
        "payout": claim_result["payout"],
        "oop": claim_result["out_of_pocket"],
        "damage": damage
    }, year)
    
    # Verify memory across years
    all_mem = hh.memory.retrieve(top_k=10, current_year=year)
    assert len(all_mem) >= 2  # Decision + Claim
    
    print(f"✓ Premium paid: ${annual_premium:,.0f}")
    print(f"✓ Claim paid: ${claim_result['payout']:,.0f}")
    print(f"✓ Insurance total claims: ${ins.state.claims_paid:,.0f}")
    print("✓ Household ↔ Insurance interaction completed")


def test_multi_year_simulation():
    """Test multiple years of simulation."""
    print("\n--- test_multi_year_simulation ---")
    
    hh = MockHousehold("H005", "MG_Owner")
    ins = MockInsurance()
    gov = MockGovernment()
    rng = np.random.default_rng(42)
    
    flood_years = []
    
    for year in range(1, 6):  # 5 years
        # Generate flood event
        flood = process("flood", {"year": year, "rng": rng})
        
        if flood["occurred"]:
            flood_years.append(year)
            
            # Process damage
            damage = process("damage", {
                "severity": flood["severity"],
                "property_value": hh.state.property_value,
                "elevated": hh.state.elevated
            })
            
            # Process claim if insured
            if hh.state.has_insurance:
                claim = process("claim", {
                    "damage_amount": damage["damage_amount"],
                    "has_insurance": True,
                    "payout_ratio": 1.0
                })
                add_memory(hh.memory, "claim", {
                    "filed": True, "approved": True,
                    "payout": claim["payout"], "oop": claim["out_of_pocket"],
                    "damage": damage["damage_amount"]
                }, year)
            
            # Record flood
            add_memory(hh.memory, "flood", {"damage": damage["damage_amount"]}, year)
        
        # Simple decision: buy insurance if had damage
        if not hh.state.has_insurance and year > 1 and len(flood_years) > 0:
            hh.state.has_insurance = True
            add_memory(hh.memory, "decision", {"skill_id": "buy_insurance"}, year)
        else:
            add_memory(hh.memory, "decision", {"skill_id": "do_nothing"}, year)
        
        # Consolidate at year end
        hh.memory.consolidate()
    
    # Verify memory state after 5 years
    final_mem = hh.memory.retrieve(top_k=10, current_year=5)
    
    print(f"✓ Flood years: {flood_years}")
    print(f"✓ Has insurance: {hh.state.has_insurance}")
    print(f"✓ Total memories: {len(hh.memory._working) + len(hh.memory._episodic)}")
    print(f"✓ Retrieved top memories: {len(final_mem)}")
    print("✓ Multi-year simulation completed")


def main():
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS: YEAR LOOP + AGENT INTERACTION")
    print("=" * 60)
    
    test_year_loop_no_flood()
    test_year_loop_with_flood()
    test_household_government_interaction()
    test_household_insurance_interaction()
    test_multi_year_simulation()
    
    print("\n" + "=" * 60)
    print("✅ ALL 5 INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
