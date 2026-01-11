"""
Modular Flood Adaptation Experiment - PR 3 Template
Demonstrates the 'Puzzle' Architecture using ExperimentBuilder.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from broker.experiment import ExperimentBuilder
from broker.social_graph import NeighborhoodGraph
from broker.interaction_hub import InteractionHub
from broker.context_builder import TieredContextBuilder
from examples.single_agent.run_experiment import FloodSimulation, FloodContextBuilder

def run_modular_approach(model: str = "llama3.2:3b", years: int = 10, agents_count: int = 100):
    print(f"--- Launching Modular Experiment: {model} ---")
    
    # 1. Initialize Simulation (World Layer)
    sim = FloodSimulation(num_agents=agents_count)
    agent_ids = list(sim.agents.keys())

    # 2. Setup Social Interaction Layer (PR 2)
    # Using NeighborhoodGraph (K=4) to simulate local street-level influence
    graph = NeighborhoodGraph(agent_ids, k=4)
    hub = InteractionHub(graph)
    
    # 3. Setup Tiered Context Builder (PR 2)
    # This automatically handles Tier 0 (Personal), Tier 1 (Local), and Tier 2 (Global)
    ctx_builder = TieredContextBuilder(
        agents=sim.agents,
        hub=hub,
        global_news=["City Council discusses new flood wall construction."]
    )

    # 4. Assemble the Experiment Using Fluent API (PR 1)
    builder = ExperimentBuilder()
    runner = (
        builder
        .with_model(model)
        .with_years(years)
        .with_agents(sim.agents)
        .with_simulation(sim)
        .with_context_builder(ctx_builder)
        .with_governance("strict", "examples/single_agent/agent_types.yaml")
        .with_output("results_modular")
        .build()
    )

    # 5. LLM
    from examples.single_agent.run_experiment import create_llm_invoke
    llm_invoke = create_llm_invoke(model)

    # 6. Execute
    runner.run(llm_invoke=llm_invoke)
    print(f"--- Modular Experiment Complete! Results in results_modular ---")

if __name__ == "__main__":
    # Example run with mock/toy settings
    run_modular_approach(model="mock", years=2, agents_count=10)
