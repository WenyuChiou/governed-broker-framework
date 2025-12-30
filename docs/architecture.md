# Governed Broker Framework - Architecture

## Overview

A generalizable governance middleware for LLM-driven Agent-Based Models (ABMs).

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Agent Layer                          │
│  • READ-ONLY access to state                                │
│  • Produces structured JSON proposals                       │
│  • CANNOT execute actions directly                          │
└───────────────────────┬─────────────────────────────────────┘
                        │ ① Bounded Context
                        │ ② Structured Output
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Governed Broker Layer                       │
│  • Validation (schema, policy, feasibility, theory)         │
│  • Retry management                                         │
│  • Audit logging                                            │
│  • NO STATE MUTATION                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │ ④ Action Request (Intent)
                        │ ⑤ Admissible Command
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               Simulation Engine Layer                       │
│  • ONLY layer that mutates state                            │
│  • Executes admissible commands                             │
│  • Updates memory deterministically                         │
│  • ALL causality happens here                               │
└─────────────────────────────────────────────────────────────┘
```

## 6-Step Decision Lifecycle

| Step | Layer | Action | Notes |
|------|-------|--------|-------|
| ① | Broker | Signal Read | Builds bounded context (READ-ONLY) |
| ② | LLM | Structured Output | JSON reasoning + decision |
| ③ | Broker | Validation | Schema, policy, feasibility, theory |
| ④ | Broker | Request Submit | Intent only, NOT execution |
| ⑤ | Engine | Admissibility | System constraints check |
| ⑥ | Engine | Execution | **ONLY** state mutation point |

## Component Responsibilities

### Broker
- `BrokerEngine`: Main orchestrator
- `ContextBuilder`: Builds bounded context
- `AuditWriter`: Writes JSONL traces
- `RetryManager`: Handles validation failures

### Interfaces
- `ReadInterface`: Read-only state access
- `ActionRequestInterface`: Submit intents (④)
- `ExecutionInterface`: System-only execution (⑥)

### Validators
- `SchemaValidator`: JSON schema compliance
- `PolicyValidator`: Role-based access
- `FeasibilityValidator`: Constraint checking
- `LeakageValidator`: No hidden state access
- `MemoryIntegrityValidator`: No memory writes

## Key Invariants

1. **LLM is READ-ONLY**: Cannot modify state
2. **Broker never mutates**: Validation and audit only
3. **Engine owns causality**: All state transitions
4. **Audit everything**: Every step produces trace
5. **Replay deterministic**: Same seed + trace = same result
