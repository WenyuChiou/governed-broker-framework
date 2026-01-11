import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from broker.social_graph import NeighborhoodGraph
from broker.interaction_hub import InteractionHub
from agents.base_agent import BaseAgent, AgentConfig

def test_social_logic():
    print("Verifying Social Integration Logic...")
    
    # 1. Setup 5 agents in a line
    agent_ids = [f"Agent_{i}" for i in range(5)]
    agents = {}
    for aid in agent_ids:
        config = AgentConfig(name=aid, agent_type="household", 
                             state_params=[], objectives=[], constraints=[], 
                             skills=["do_nothing"], perception=[])
        agent = BaseAgent(config)
        agent.id = aid
        agent.memory = [f"Memory of {aid}"]
        agent.elevated = False
        agents[aid] = agent

    # 2. Setup Neighborhood Graph (K=2, so each has 2 neighbors)
    graph = NeighborhoodGraph(agent_ids, k=2)
    hub = InteractionHub(graph)
    
    # 3. Modify Agent_1 and Agent_2 (Neighbors of Agent_0)
    agents["Agent_1"].elevated = True
    agents["Agent_1"].memory.append("I raised my house!")
    
    # Agent_3 IS NOT a neighbor of Agent_0 in a K=2 graph for N=5
    # Neighbors of 0 are 1 and 4.
    agents["Agent_3"].elevated = True 
    agents["Agent_3"].memory.append("I am far away!")

    # 4. Verify Agent_0 Context
    ctx = hub.build_tiered_context("Agent_0", agents, global_news=["Gov Policy Changed"])
    
    print("\n--- Agent_0 Context Check ---")
    print(f"Spatial Check (Elevated Pct): {ctx['local']['spatial']['elevated_pct']}%")
    # Agent_0 neighbors: 1 and 4. 1 is elevated, 4 is not. -> 50%
    assert ctx['local']['spatial']['elevated_pct'] == 50
    
    print(f"Social Check (Gossip): {ctx['local']['social']}")
    # Should only hear from 1 or 4.
    for gossip in ctx['local']['social']:
        assert "Agent_3" not in gossip, "Leak: Heard gossip from non-neighbor Agent_3!"
        if "Agent_1" in gossip:
            assert "I raised my house!" in gossip
            
    print(f"Global Check: {ctx['global']}")
    assert "Gov Policy Changed" in ctx['global']

    print("\nSocial Integration Test PASSED!")

if __name__ == "__main__":
    test_social_logic()
