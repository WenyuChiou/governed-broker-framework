""" 
Lite LLM Verification Script

Minimal script to verify:
1. Framework Core initialization (Registry, Adapter, Validator, Audit)
2. LLM Connectivity (via Ollama or Mock)
3. Skill Proposal Parsing & Validation
4. Audit Trail Generation
"""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from broker.agent_config import AgentTypeConfig
from broker.model_adapter import UnifiedAdapter
from broker.audit_writer import GenericAuditWriter, AuditConfig
from validators.agent_validator import AgentValidator

def call_llm(model: str, prompt: str):
    """Simple Ollama call."""
    if model.lower() == "mock":
        return """
INTERPRET: High risk detected.
PMT_EVAL: TP=H CP=M SP=H SC=M PA=NONE
DECIDE: buy_insurance
REASON: Protection needed.
"""
    
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mock", help="Ollama model name or 'mock'")
    parser.add_argument("--agent-type", type=str, default="household", help="Agent type from agent_types.yaml")
    args = parser.parse_args()

    print(f"--- Lite LLM Verification (Model: {args.model}) ---")
    
    # 1. Initialize Components
    print("[1] Initializing Framework Core...")
    config = AgentTypeConfig.load()
    adapter = UnifiedAdapter(agent_type=args.agent_type)
    validator = AgentValidator()
    audit = GenericAuditWriter(AuditConfig(
        output_dir="tests/verify_logs",
        experiment_name="lite_verification"
    ))
    
    # 2. Mock Agent Context
    print("[2] Building Bounded Context...")
    skills = config.get_valid_actions(args.agent_type)
    context = {
        "agent_id": "test_agent_001",
        "agent_type": args.agent_type,
        "income": 45000,
        "property_value": 250000,
        "cumulative_damage": 5000,
        "trust_gov": 0.5,
        "trust_ins": 0.6,
        "trust_neighbors": 0.4,
        "elevated": False,
        "has_insurance": False,
        "skills": skills
    }
    
    # 3. Build Prompt
    prompt = f"""
You are a {args.agent_type} agent in a disaster simulation.
State: Income=${context['income']}, Damage=${context['cumulative_damage']}, Elevated: {context['elevated']}
Available Skills: {', '.join(skills)}

Respond in structured format:
INTERPRET: ...
PMT_EVAL: TP=[SL|L|M|H] CP=[L|M|H] SP=[L|M|H] SC=[L|M|H] PA=[NONE|PARTIAL|FULL]
DECIDE: [skill_id]
REASON: ...
"""

    # 4. Invoke LLM
    print(f"[3] Calling LLM ({args.model})...")
    raw_response = call_llm(args.model, prompt)
    if "Error" in raw_response:
        print(f"ABORTED: {raw_response}")
        return
    print(f"RAW RECEIVED (First 50 chars): {raw_response.strip()[:50]}...")

    # 5. Parse
    print("[4] Parsing Skill Proposal...")
    proposal = adapter.parse_output(raw_response, context)
    if not proposal:
        print("FAILED: Could not parse response.")
        return
    print(f"SUCCESS: Proposed Skill -> {proposal.skill_name}")

    # 6. Validate
    print("[5] Validating Proposal...")
    validation_results = validator.validate(
        agent_type=args.agent_type,
        agent_id=context["agent_id"],
        decision=proposal.skill_name,
        state=context,
        reasoning=proposal.reasoning
    )
    is_valid = not any(not v.valid for v in validation_results)
    print(f"VALIDATION: {'PASSED' if is_valid else 'FAILED'}")

    # 7. Audit
    print("[6] Writing Audit Trace...")
    trace = {
        "agent_id": context["agent_id"],
        "year": 1,
        "decision": proposal.skill_name,
        "reasoning": proposal.reasoning,
        "raw_context": {k: v for k, v in context.items() if k != "skills"}
    }
    audit.write_trace(args.agent_type, trace, validation_results)
    
    print(f"\nVerification Complete. Log written to: tests/verify_logs/lite_verification_{args.agent_type}_audit.jsonl")

if __name__ == "__main__":
    main()
