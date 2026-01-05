"""
Memory Helpers Unit Tests - Unified Interface

Tests the unified add_memory(event_type, data, year) interface.
"""

import sys
sys.path.insert(0, '.')

from broker.memory import CognitiveMemory
from examples.exp3_multi_agent.memory_helpers import (
    add_memory,
    add_year_end_memories,
    EVENT_CONFIG
)


def test_event_config_complete():
    """Test EVENT_CONFIG has all event types."""
    expected = ["decision", "flood", "claim", "neighbor", "policy", "premium"]
    for event_type in expected:
        assert event_type in EVENT_CONFIG, f"Missing: {event_type}"
    print("✓ test_event_config_complete passed")


# =============================================================================
# DECISION TESTS
# =============================================================================

def test_decision_buy_insurance():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "decision", {"skill_id": "buy_insurance"}, year=3)
    assert item is not None
    assert item.importance == 0.7
    assert "insurance" in item.content.lower()
    print("✓ test_decision_buy_insurance passed")


def test_decision_do_nothing():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "decision", {"skill_id": "do_nothing"}, year=2)
    assert item.importance == 0.3
    print("✓ test_decision_do_nothing passed")


# =============================================================================
# FLOOD TESTS
# =============================================================================

def test_flood_major():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "flood", {"damage": 75000}, year=3)
    assert "major" in item.content.lower()
    assert item.importance == 0.9
    assert len(mem._episodic) == 1
    print("✓ test_flood_major passed")


def test_flood_minor():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "flood", {"damage": 5000}, year=2)
    assert "minor" in item.content.lower()
    assert item.importance == 0.6
    print("✓ test_flood_minor passed")


def test_flood_no_damage():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "flood", {"occurred": True, "damage": 0}, year=4)
    assert "not damaged" in item.content.lower()
    assert len(mem._working) == 1
    print("✓ test_flood_no_damage passed")


def test_flood_none():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "flood", {"occurred": False, "damage": 0}, year=5)
    assert item is None
    print("✓ test_flood_none passed")


# =============================================================================
# CLAIM TESTS
# =============================================================================

def test_claim_approved():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "claim", {
        "filed": True, "approved": True, "payout": 40000, "damage": 50000, "oop": 10000
    }, year=3)
    assert item is not None
    assert "approved" in item.tags
    print("✓ test_claim_approved passed")


def test_claim_denied():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "claim", {
        "filed": True, "approved": False, "payout": 0, "damage": 50000, "oop": 50000
    }, year=3)
    assert "denied" in item.content.lower()
    assert item.importance == 0.85
    print("✓ test_claim_denied passed")


def test_claim_not_filed():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "claim", {"filed": False}, year=3)
    assert item is None
    print("✓ test_claim_not_filed passed")


# =============================================================================
# NEIGHBOR TESTS (Year Beginning)
# =============================================================================

def test_neighbor_elevated():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "neighbor", {"type": "elevated", "count": 3}, year=4)
    assert item is not None
    assert "3" in item.content
    assert "neighbor" in item.tags
    print("✓ test_neighbor_elevated passed")


def test_neighbor_zero():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "neighbor", {"type": "elevated", "count": 0}, year=4)
    assert item is None
    print("✓ test_neighbor_zero passed")


# =============================================================================
# POLICY TESTS (Effective Next Year)
# =============================================================================

def test_policy_subsidy():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "policy", {"type": "subsidy_increase", "rate": 0.75}, year=3)
    assert item is not None
    assert "75%" in item.content
    print("✓ test_policy_subsidy passed")


def test_policy_buyout():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "policy", {"type": "buyout_offer", "amount": 200000}, year=5)
    assert item.importance == 0.85
    assert "$200,000" in item.content
    print("✓ test_policy_buyout passed")


# =============================================================================
# PREMIUM TESTS
# =============================================================================

def test_premium_increase():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "premium", {"pct": 0.15}, year=3)
    assert item is not None
    assert "15%" in item.content
    print("✓ test_premium_increase passed")


def test_premium_small():
    mem = CognitiveMemory(agent_id="H001")
    item = add_memory(mem, "premium", {"pct": 0.02}, year=3)
    assert item is None  # <5% not memorable
    print("✓ test_premium_small passed")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def test_year_end_memories():
    mem = CognitiveMemory(agent_id="H001")
    results = add_year_end_memories(
        mem, year=3,
        decision="buy_insurance",
        flood_occurred=True,
        damage=30000,
        claim_filed=True,
        claim_approved=True,
        payout=25000,
        oop=5000
    )
    assert results['decision'] is not None
    assert results['flood'] is not None
    assert results['claim'] is not None
    print("✓ test_year_end_memories passed")


def main():
    print("\n" + "=" * 60)
    print("UNIFIED MEMORY HELPERS TESTS")
    print("=" * 60 + "\n")
    
    test_event_config_complete()
    
    # Decision
    test_decision_buy_insurance()
    test_decision_do_nothing()
    
    # Flood
    test_flood_major()
    test_flood_minor()
    test_flood_no_damage()
    test_flood_none()
    
    # Claim
    test_claim_approved()
    test_claim_denied()
    test_claim_not_filed()
    
    # Neighbor
    test_neighbor_elevated()
    test_neighbor_zero()
    
    # Policy
    test_policy_subsidy()
    test_policy_buyout()
    
    # Premium
    test_premium_increase()
    test_premium_small()
    
    # Convenience
    test_year_end_memories()
    
    print("\n" + "=" * 60)
    print("✅ ALL 18 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
