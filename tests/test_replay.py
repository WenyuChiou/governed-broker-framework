"""
Test: Replay produces deterministic results.
"""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_replay_loads_traces():
    """Test that ReplayEngine can load audit traces."""
    from broker.replay import ReplayEngine
    
    # This would need a test audit file
    # For now, just test the class exists and has the right methods
    assert hasattr(ReplayEngine, 'run')
    assert hasattr(ReplayEngine, 'replay_step')
    assert hasattr(ReplayEngine, 'get_run_info')


def test_same_seed_same_result():
    """Test that same seed produces same results."""
    from simulation.engine import ToySimulationEngine
    
    # Run 1
    sim1 = ToySimulationEngine(num_agents=5, seed=42)
    sim1.advance_step()
    state1 = {aid: sim1.get_agent_state(aid) for aid in sim1.agents}
    
    # Run 2 with same seed
    sim2 = ToySimulationEngine(num_agents=5, seed=42)
    sim2.advance_step()
    state2 = {aid: sim2.get_agent_state(aid) for aid in sim2.agents}
    
    # States should match
    for agent_id in state1:
        assert state1[agent_id]["resources"] == state2[agent_id]["resources"]
        assert state1[agent_id]["threat_perception"] == state2[agent_id]["threat_perception"]


if __name__ == "__main__":
    test_replay_loads_traces()
    test_same_seed_same_result()
    print("All replay tests passed!")
