"""
Multi-Agent Memory Test

簡易測試: 驗證 Household, Insurance, Government 三個 Agent 使用 Memory 的狀態
"""
import sys
sys.path.insert(0, '.')

from dataclasses import dataclass, field
from typing import List, Dict, Any
from broker.memory import CognitiveMemory, MemoryProvider


# =============================================================================
# Agent-Specific Memory Classes
# =============================================================================

class HouseholdMemory(CognitiveMemory):
    """Household Agent 記憶"""
    
    PAST_EVENTS = [
        "A flood event about 15 years ago caused $500,000 in city-wide damages",
        "Some residents reported delays when processing their flood insurance claims",
        "The city previously introduced a program offering elevation support"
    ]
    
    def __init__(self, agent_id: str, agent_type: str = "MG_Owner"):
        super().__init__(agent_id)
        self.agent_type = agent_type
        self._initialize_background()
    
    def _initialize_background(self):
        """初始化背景記憶"""
        import random
        event = random.choice(self.PAST_EVENTS)
        self.add_episodic(event, importance=0.4, year=0, tags=["background"])


class InsuranceMemory(CognitiveMemory):
    """Insurance Agent 記憶"""
    
    def __init__(self):
        super().__init__("InsuranceCo")
    
    def add_year_performance(self, year: int, loss_ratio: float, 
                             claims: float, uptake: float):
        """添加年度表現"""
        content = (f"Year {year}: Loss ratio {loss_ratio:.1%}, "
                  f"Claims ${claims:,.0f}, Uptake {uptake:.1%}")
        importance = 0.9 if loss_ratio > 1.0 else 0.5
        return self.add_episodic(content, importance, year, tags=["performance"])


class GovernmentMemory(CognitiveMemory):
    """Government Agent 記憶"""
    
    def __init__(self):
        super().__init__("Government")
    
    def add_policy_record(self, year: int, subsidy_rate: float,
                          mg_adoption: float, flood_occurred: bool):
        """添加政策記錄"""
        flood_str = " [FLOOD]" if flood_occurred else ""
        content = (f"Year {year}{flood_str}: Subsidy {subsidy_rate:.0%}, "
                  f"MG adoption {mg_adoption:.0%}")
        importance = 0.8 if flood_occurred else 0.5
        return self.add_episodic(content, importance, year, tags=["policy"])


# =============================================================================
# Simple Agent State
# =============================================================================

@dataclass
class HouseholdAgent:
    id: str
    agent_type: str
    elevated: bool = False
    has_insurance: bool = False
    memory: HouseholdMemory = None
    
    def __post_init__(self):
        self.memory = HouseholdMemory(self.id, self.agent_type)


@dataclass
class InsuranceAgent:
    id: str = "InsuranceCo"
    premium_rate: float = 0.05
    loss_ratio: float = 0.0
    premium_collected: float = 0
    claims_paid: float = 0
    memory: InsuranceMemory = None
    
    def __post_init__(self):
        self.memory = InsuranceMemory()


@dataclass
class GovernmentAgent:
    id: str = "Government"
    subsidy_rate: float = 0.50
    budget_remaining: float = 500_000
    memory: GovernmentMemory = None
    
    def __post_init__(self):
        self.memory = GovernmentMemory()


# =============================================================================
# Simple Simulation
# =============================================================================

def run_simple_test():
    """簡易模擬: 測試三個 Agent 使用 Memory"""
    
    print("=" * 60)
    print("Multi-Agent Memory Test")
    print("=" * 60)
    
    # 創建 Agents
    household = HouseholdAgent(id="H001", agent_type="MG_Owner")
    insurance = InsuranceAgent()
    government = GovernmentAgent()
    
    print("\n📋 Agents 創建完成")
    print(f"  - Household: {household.id} ({household.agent_type})")
    print(f"  - Insurance: {insurance.id}")
    print(f"  - Government: {government.id}")
    
    # 模擬 3 年
    flood_years = [False, True, False]
    
    for year in range(1, 4):
        flood = flood_years[year - 1]
        print(f"\n{'='*60}")
        print(f"Year {year} {'🌊 FLOOD' if flood else ''}")
        print("=" * 60)
        
        # Phase 1: Institutional Decisions
        print("\n[Phase 1] Institutional Decisions")
        
        # Insurance Decision
        if insurance.loss_ratio > 0.80:
            insurance.premium_rate *= 1.10
            insurance.memory.add_experience(
                f"Year {year}: Raised premium due to high loss ratio",
                importance=0.7, year=year
            )
            print(f"  Insurance: Raised premium to {insurance.premium_rate:.1%}")
        else:
            insurance.memory.add_experience(
                f"Year {year}: Maintained premium rate",
                importance=0.3, year=year
            )
            print(f"  Insurance: Maintained premium at {insurance.premium_rate:.1%}")
        
        # Government Decision
        if flood and year > 1:
            government.subsidy_rate = min(0.80, government.subsidy_rate + 0.10)
            government.memory.add_policy_record(year, government.subsidy_rate, 0.25, True)
            print(f"  Government: Increased subsidy to {government.subsidy_rate:.0%}")
        else:
            government.memory.add_policy_record(year, government.subsidy_rate, 0.30, flood)
            print(f"  Government: Maintained subsidy at {government.subsidy_rate:.0%}")
        
        # Phase 2: Household Decision
        print("\n[Phase 2] Household Decision")
        
        if flood and not household.has_insurance:
            household.has_insurance = True
            household.memory.update_after_decision("buy_insurance", year)
            household.memory.update_after_flood(15000, year)
            print(f"  Household: Bought insurance after flood (damage: $15,000)")
        elif not household.elevated:
            household.memory.add_experience(
                f"Year {year}: Considering elevation but decided to wait",
                importance=0.4, year=year
            )
            print(f"  Household: Decided to wait")
        else:
            household.memory.add_experience(
                f"Year {year}: Already protected, no action needed",
                importance=0.2, year=year
            )
            print(f"  Household: No action (already protected)")
        
        # Phase 3: Settlement
        print("\n[Phase 3] Settlement")
        
        if flood:
            damage = 15000
            if household.has_insurance:
                payout = damage * 0.80
                insurance.claims_paid += payout
                print(f"  Insurance paid ${payout:,.0f} to Household")
            
            insurance.premium_collected += 1000
            insurance.loss_ratio = insurance.claims_paid / max(insurance.premium_collected, 1)
            insurance.memory.add_year_performance(year, insurance.loss_ratio, 
                                                   insurance.claims_paid, 0.35)
        else:
            insurance.premium_collected += 1000
            insurance.loss_ratio = insurance.claims_paid / max(insurance.premium_collected, 1)
            insurance.memory.add_year_performance(year, insurance.loss_ratio, 
                                                   insurance.claims_paid, 0.35)
            print(f"  No flood occurred")
    
    # 最終記憶狀態
    print("\n" + "=" * 60)
    print("Final Memory States")
    print("=" * 60)
    
    print("\n🏠 Household Memory:")
    for mem in household.memory.retrieve(top_k=5, current_year=3):
        print(f"  - {mem}")
    
    print("\n🏢 Insurance Memory:")
    for mem in insurance.memory.retrieve(top_k=5, current_year=3):
        print(f"  - {mem}")
    
    print("\n🏛️ Government Memory:")
    for mem in government.memory.retrieve(top_k=5, current_year=3):
        print(f"  - {mem}")
    
    # 驗證統計
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    hh_mem_count = len(household.memory._working) + len(household.memory._episodic)
    ins_mem_count = len(insurance.memory._working) + len(insurance.memory._episodic)
    gov_mem_count = len(government.memory._working) + len(government.memory._episodic)
    
    print(f"\n✅ Memory counts:")
    print(f"  Household: {hh_mem_count} memories")
    print(f"  Insurance: {ins_mem_count} memories")
    print(f"  Government: {gov_mem_count} memories")
    
    print(f"\n✅ Final states:")
    print(f"  Household has_insurance: {household.has_insurance}")
    print(f"  Insurance loss_ratio: {insurance.loss_ratio:.1%}")
    print(f"  Government subsidy_rate: {government.subsidy_rate:.0%}")
    
    print("\n🎉 Test completed successfully!")
    
    return {
        "household": household,
        "insurance": insurance,
        "government": government
    }


if __name__ == "__main__":
    run_simple_test()
