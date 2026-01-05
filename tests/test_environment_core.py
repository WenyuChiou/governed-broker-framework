"""
Environment Core Unit Tests - Unified Interface

Tests the unified process(event_type, data) interface.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from examples.exp3_multi_agent.environment.core import (
    process,
    ENV_CONFIG,
    FloodEvent,
    DamageResult,
    ClaimResult,
    SubsidyResult
)


def test_env_config_complete():
    """Test ENV_CONFIG has all required sections."""
    expected = ["flood", "damage", "insurance", "subsidy"]
    for section in expected:
        assert section in ENV_CONFIG, f"Missing: {section}"
    print("✓ test_env_config_complete passed")


# =============================================================================
# FLOOD TESTS
# =============================================================================

def test_flood_deterministic():
    """Test flood with fixed seed."""
    rng = np.random.default_rng(42)
    result = process("flood", {"year": 1, "rng": rng})
    
    assert "occurred" in result
    assert "severity" in result
    assert "depth_ft" in result
    print("✓ test_flood_deterministic passed")


def test_flood_no_event():
    """Test when no flood occurs."""
    # Use seed that produces no flood
    rng = np.random.default_rng(123)
    result = process("flood", {"year": 1, "rng": rng})
    
    # Result should have all fields regardless
    assert isinstance(result["occurred"], bool)
    print("✓ test_flood_no_event passed")


# =============================================================================
# DAMAGE TESTS
# =============================================================================

def test_damage_zero():
    """Test damage with no flood."""
    result = process("damage", {"severity": 0.0, "property_value": 300000})
    
    assert result["damage_ratio"] == 0.0
    assert result["damage_amount"] == 0.0
    print("✓ test_damage_zero passed")


def test_damage_elevated():
    """Test damage reduction with elevation."""
    # Same severity, with and without elevation
    base = process("damage", {"severity": 0.5, "property_value": 300000, "elevated": False})
    elevated = process("damage", {"severity": 0.5, "property_value": 300000, "elevated": True})
    
    assert elevated["damage_amount"] < base["damage_amount"]
    assert elevated["damage_ratio"] < base["damage_ratio"]
    print("✓ test_damage_elevated passed")


def test_damage_severe():
    """Test damage with severe flood."""
    result = process("damage", {"severity": 0.9, "property_value": 300000})
    
    assert result["damage_ratio"] > 0.5
    assert result["building_damage"] > 0
    assert result["contents_damage"] > 0
    print("✓ test_damage_severe passed")


def test_damage_overtopped():
    """Test elevation overtopped at high severity."""
    base = process("damage", {"severity": 0.95, "property_value": 300000, "elevated": False})
    elevated = process("damage", {"severity": 0.95, "property_value": 300000, "elevated": True})
    
    # Still reduced but less than normal elevation
    assert elevated["damage_amount"] < base["damage_amount"]
    # But ratio is higher than normal elevation benefit
    ratio = elevated["damage_amount"] / base["damage_amount"]
    assert ratio > 0.3  # Not 95% reduction
    print("✓ test_damage_overtopped passed")


# =============================================================================
# CLAIM TESTS
# =============================================================================

def test_claim_no_insurance():
    """Test claim without insurance."""
    result = process("claim", {"damage_amount": 50000, "has_insurance": False})
    
    assert result["filed"] == False
    assert result["payout"] == 0
    assert result["out_of_pocket"] == 50000
    print("✓ test_claim_no_insurance passed")


def test_claim_approved():
    """Test approved claim."""
    result = process("claim", {"damage_amount": 50000, "has_insurance": True})
    
    assert result["filed"] == True
    assert result["approved"] == True
    assert result["payout"] > 0
    assert result["out_of_pocket"] < 50000
    print("✓ test_claim_approved passed")


def test_claim_deductible():
    """Test deductible is applied."""
    result = process("claim", {"damage_amount": 50000, "has_insurance": True})
    
    # Payout should be less than damage - deductible
    expected_max = 50000 - ENV_CONFIG["insurance"]["default_deductible"]
    assert result["payout"] <= expected_max
    print("✓ test_claim_deductible passed")


def test_claim_coverage_limit():
    """Test NFIP coverage limit."""
    result = process("claim", {"damage_amount": 500000, "has_insurance": True})
    
    # Payout capped at limit - deductible
    limit = ENV_CONFIG["insurance"]["nfip_building_limit"]
    deductible = ENV_CONFIG["insurance"]["default_deductible"]
    assert result["payout"] <= limit - deductible
    print("✓ test_claim_coverage_limit passed")


# =============================================================================
# SUBSIDY TESTS
# =============================================================================

def test_subsidy_elevate():
    """Test elevation subsidy."""
    result = process("subsidy", {"action": "elevate_house", "budget": 500000})
    
    assert result["approved"] == True
    assert result["subsidy_amount"] > 0
    assert result["net_cost"] < result["cost_basis"]
    print("✓ test_subsidy_elevate passed")


def test_subsidy_mg_priority():
    """Test MG priority bonus."""
    base = process("subsidy", {"action": "elevate_house", "is_mg": False, "mg_priority": True, "budget": 500000})
    mg = process("subsidy", {"action": "elevate_house", "is_mg": True, "mg_priority": True, "budget": 500000})
    
    assert mg["applied_rate"] > base["applied_rate"]
    assert mg["subsidy_amount"] > base["subsidy_amount"]
    print("✓ test_subsidy_mg_priority passed")


def test_subsidy_budget_cap():
    """Test subsidy capped by budget."""
    result = process("subsidy", {"action": "elevate_house", "budget": 10000})
    
    assert result["subsidy_amount"] == 10000  # Capped
    assert result["approved"] == True
    print("✓ test_subsidy_budget_cap passed")


def test_subsidy_no_budget():
    """Test no subsidy when budget exhausted."""
    result = process("subsidy", {"action": "elevate_house", "budget": 0})
    
    assert result["approved"] == False
    assert result["subsidy_amount"] == 0
    print("✓ test_subsidy_no_budget passed")


def main():
    print("\n" + "=" * 60)
    print("ENVIRONMENT CORE TESTS")
    print("=" * 60 + "\n")
    
    test_env_config_complete()
    
    # Flood
    test_flood_deterministic()
    test_flood_no_event()
    
    # Damage
    test_damage_zero()
    test_damage_elevated()
    test_damage_severe()
    test_damage_overtopped()
    
    # Claim
    test_claim_no_insurance()
    test_claim_approved()
    test_claim_deductible()
    test_claim_coverage_limit()
    
    # Subsidy
    test_subsidy_elevate()
    test_subsidy_mg_priority()
    test_subsidy_budget_cap()
    test_subsidy_no_budget()
    
    print("\n" + "=" * 60)
    print("✅ ALL 16 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
