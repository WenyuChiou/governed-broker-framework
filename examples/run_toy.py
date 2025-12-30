"""
Run Toy Domain Example

Demonstrates the Governed Broker Framework with a minimal domain.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.engine import ToySimulationEngine
from broker.context_builder import SimpleContextBuilder
from broker.audit_writer import AuditWriter, AuditConfig
from broker.types import DecisionRequest, OutcomeType
from validators.base import SchemaValidator, PolicyValidator

# Configuration
NUM_AGENTS = 10
NUM_STEPS = 5
SEED = 42
OUTPUT_DIR = "toy_output"


def mock_llm(prompt: str) -> str:
    """
    Mock LLM that returns deterministic decisions.
    
    In real usage, this would call an actual LLM.
    """
    # Parse context from prompt to make decision
    if "risk_level: high" in prompt.lower() or "shock occurred" in prompt.lower():
        decision = "adapt"
        threat = "High risk detected, need to take action"
    elif "resources: low" in prompt.lower():
        decision = "do_nothing"
        threat = "Limited resources, conserving"
    else:
        decision = "buy_insurance"
        threat = "Moderate risk, taking precautions"
    
    return json.dumps({
        "decision": decision,
        "threat_appraisal": threat,
        "coping_appraisal": "I can handle this situation"
    })


def create_prompt_template() -> str:
    return """You are an agent in a risky environment.

Current State:
- Resources: {resources:.1f}
- Threat Perception: {threat_perception:.2f}
- Vulnerability: {vulnerability:.2f}
- Risk Level: {risk_level:.2f}
- Shock Occurred: {shock_occurred}

Memory:
{memory_str}

Available Actions:
1. do_nothing - No cost, no benefit
2. adapt - Costs 30 resources, reduces vulnerability by 50%
3. buy_insurance - Costs 15 resources, reduces vulnerability by 30%

Respond with JSON:
{{
  "decision": "do_nothing" | "adapt" | "buy_insurance",
  "threat_appraisal": "your assessment of the threat",
  "coping_appraisal": "your assessment of your ability to cope"
}}
"""


def run_toy():
    """Run the toy domain simulation."""
    print("=" * 60)
    print("Governed Broker Framework - Toy Domain Example")
    print("=" * 60)
    
    # Initialize simulation
    sim = ToySimulationEngine(num_agents=NUM_AGENTS, seed=SEED)
    
    # Initialize audit
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    audit = AuditWriter(AuditConfig(
        output_dir=OUTPUT_DIR,
        log_level="full"
    ))
    
    # Validators
    validators = [
        SchemaValidator(required_fields=["decision"]),
        PolicyValidator(allowed_actions=["do_nothing", "adapt", "buy_insurance"])
    ]
    
    # Prompt template
    prompt_template = create_prompt_template()
    
    run_id = f"toy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    step_counter = 0
    
    # Main simulation loop
    for step in range(1, NUM_STEPS + 1):
        print(f"\n--- Step {step} ---")
        sim.advance_step()
        
        env = sim.get_environment_state()
        print(f"Risk Level: {env['risk_level']:.2f}, Shock: {env['shock_occurred']}")
        
        for agent_id, agent in sim.agents.items():
            if not agent.is_active:
                continue
            
            step_counter += 1
            
            # ① Build context
            state = sim.get_agent_state(agent_id)
            context = {
                **state,
                **env,
                "memory_str": "\n".join(f"- {m}" for m in state["memory"])
            }
            
            # ② LLM invocation
            prompt = prompt_template.format(**context)
            raw_output = mock_llm(prompt)
            
            try:
                data = json.loads(raw_output)
                request = DecisionRequest(
                    action_code=data["decision"],
                    reasoning={
                        "threat": data.get("threat_appraisal", ""),
                        "coping": data.get("coping_appraisal", "")
                    },
                    raw_output=raw_output
                )
            except:
                continue
            
            # ③ Validation
            all_valid = True
            validation_results = []
            for v in validators:
                result = v.validate(request, context)
                validation_results.append(result)
                if not result.valid:
                    all_valid = False
            
            if not all_valid:
                outcome = OutcomeType.UNCERTAIN
                action_name = "do_nothing"
            else:
                outcome = OutcomeType.EXECUTED
                action_name = request.action_code
            
            # ⑤ Admissibility check
            admissible = sim.check_admissibility(agent_id, action_name, {})
            
            # ⑥ Execute
            execution_result = None
            if admissible.admissibility_check == "PASSED":
                execution_result = sim.execute(admissible)
            
            # Write audit
            audit.write_trace({
                "run_id": run_id,
                "step_id": step_counter,
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "agent_id": agent_id,
                "memory_pre": state["memory"],
                "llm_output": request.to_dict(),
                "validator_results": [{"valid": v.valid, "errors": v.errors} for v in validation_results],
                "action_request": action_name,
                "admissible_command": admissible.__dict__,
                "execution_result": execution_result.__dict__ if execution_result else None,
                "memory_post": sim.get_agent_state(agent_id).get("memory", []),
                "outcome": outcome.value
            })
    
    # Finalize
    summary = audit.finalize()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total steps: {summary['total_steps']}")
    print(f"Executed: {summary['executed']}")
    print(f"Uncertain: {summary['uncertain']}")
    print(f"Consistency: {summary['consistency_rate']}")
    print(f"\nAudit saved to: {OUTPUT_DIR}/")
    
    # Save final state
    final_state = {
        "run_id": run_id,
        "agents": {aid: sim.get_agent_state(aid) for aid in sim.agents},
        "environment": sim.get_environment_state()
    }
    
    with open(output_path / "final_state.json", "w") as f:
        json.dump(final_state, f, indent=2, default=str)
    
    print(f"Final state saved to: {OUTPUT_DIR}/final_state.json")


if __name__ == "__main__":
    run_toy()
