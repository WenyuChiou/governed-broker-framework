"""Test Generic Institutional Agent Framework."""

import sys
sys.path.insert(0, '.')

from agents.loader import load_agent_configs, load_agents
from agents.institutional_base import normalize, denormalize


def test_normalization():
    """Test 0-1 normalization."""
    # loss_ratio: 60% in range [0, 150%]
    assert abs(normalize(0.6, 0, 1.5) - 0.4) < 0.01
    
    # solvency: $1M in range [0, $2M]
    assert abs(normalize(1000000, 0, 2000000) - 0.5) < 0.01
    
    # denormalize back
    assert abs(denormalize(0.5, 0, 2000000) - 1000000) < 1
    
    print("✅ Normalization tests passed")


def test_load_configs():
    """Test YAML config loading."""
    configs = load_agent_configs("agents/flood_agents.yaml")
    
    assert len(configs) == 2
    assert configs[0].name == "InsuranceCo"
    assert configs[1].name == "StateGov"
    
    # Check Insurance state params
    ins_config = configs[0]
    assert len(ins_config.state_params) == 4
    assert ins_config.state_params[0].name == "loss_ratio"
    
    # Check objectives
    assert len(ins_config.objectives) == 3
    assert ins_config.objectives[0].name == "maintain_solvency"
    
    print("✅ Config loading tests passed")


def test_agent_instantiation():
    """Test agent creation and state access."""
    agents = load_agents("agents/flood_agents.yaml")
    
    assert "InsuranceCo" in agents
    assert "StateGov" in agents
    
    ins = agents["InsuranceCo"]
    gov = agents["StateGov"]
    
    # All state should be 0-1
    for param, value in ins.get_all_state().items():
        assert 0 <= value <= 1, f"{param} = {value} not in [0,1]"
    
    # Check specific values
    # loss_ratio raw=0.6, range=[0,1.5] -> normalized = 0.4
    assert abs(ins.get_state("loss_ratio") - 0.4) < 0.01
    
    # premium_rate raw=0.05, range=[0.02,0.15] -> normalized = (0.05-0.02)/(0.15-0.02) = 0.23
    assert abs(ins.get_state("premium_rate") - 0.23) < 0.01
    
    print("✅ Agent instantiation tests passed")


def test_objectives_evaluation():
    """Test objective evaluation."""
    agents = load_agents("agents/flood_agents.yaml")
    ins = agents["InsuranceCo"]
    
    obj_eval = ins.evaluate_objectives()
    
    assert "maintain_solvency" in obj_eval
    assert "target_loss_ratio" in obj_eval
    
    # Check structure
    for name, result in obj_eval.items():
        assert "current" in result
        assert "target" in result
        assert "in_range" in result
        assert "weight" in result
    
    print("✅ Objectives evaluation tests passed")


def test_constraint_check():
    """Test constraint validation."""
    agents = load_agents("agents/flood_agents.yaml")
    ins = agents["InsuranceCo"]
    
    # Valid change (within 15%)
    valid, msg = ins.check_constraint("premium_rate", 0.10)
    assert valid, msg
    
    # Invalid change (exceeds 15%)
    valid, msg = ins.check_constraint("premium_rate", 0.20)
    assert not valid
    
    print("✅ Constraint check tests passed")


def test_skill_execution():
    """Test skill execution."""
    agents = load_agents("agents/flood_agents.yaml")
    ins = agents["InsuranceCo"]
    
    initial_rate = ins.get_state("premium_rate")
    
    # Execute raise_premium with 5% adjustment
    success = ins.execute_skill("raise_premium", 0.05)
    assert success
    
    new_rate = ins.get_state("premium_rate")
    assert new_rate > initial_rate
    
    print("✅ Skill execution tests passed")


if __name__ == "__main__":
    test_normalization()
    test_load_configs()
    test_agent_instantiation()
    test_objectives_evaluation()
    test_constraint_check()
    test_skill_execution()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED ✅")
    print("="*50)
