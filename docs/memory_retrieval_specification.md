# Memory and Retrieval Module Specification

## 1. Overview: Memory as State vs Retrieval as Tool

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY / RETRIEVAL DISTINCTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MEMORY (State)                   RETRIEVAL (Tool)                         │
│   ─────────────────                ────────────────                         │
│   - Persistent storage             - Active extraction                      │
│   - Updated by events              - Invoked during reasoning               │
│   - Owned by agent                 - Returns subset to prompt               │
│   - Deterministic                  - May have scoring/ranking               │
│                                                                             │
│   Agent owns memory →              Agent calls retrieval →                  │
│   Memory accumulates               Retrieval returns relevant subset        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Principle

> **Memory is WHERE experiences are stored.**  
> **Retrieval is HOW experiences are accessed for decision-making.**

---

## 2. Memory Classification

### 2.1 Memory Types

| Type | Duration | Capacity | Decay | Use Case |
|------|----------|----------|-------|----------|
| **Working Memory** | Short-term | 5-10 items | None (evicted) | Recent decisions, current context |
| **Episodic Memory** | Long-term | 50+ items | Yes (time-based) | Significant experiences (floods, losses) |

### 2.2 Memory Item Structure

```python
@dataclass
class MemoryItem:
    content: str                 # Text description of experience
    importance: float = 0.5      # 0.0 - 1.0 (determines consolidation)
    year: int = 0                # Simulation year when created
    timestamp: datetime          # Real timestamp (for ordering)
    tags: List[str] = []         # Classification tags

# Examples:
MemoryItem(
    content="Year 3: A flood occurred causing $50,000 in damages",
    importance=0.9,      # High -> goes to episodic
    year=3,
    tags=["flood", "damage"]
)

MemoryItem(
    content="Year 4: I decided to buy_insurance",
    importance=0.5,      # Medium -> working memory
    year=4,
    tags=["decision"]
)
```

### 2.3 Memory Layers Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     MEMORY LAYERS                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Event occurs                                               │
│       ↓                                                      │
│   ┌──────────────┐                                           │
│   │ Importance   │                                           │
│   │ Assessment   │                                           │
│   └──────────────┘                                           │
│       ↓                                                      │
│   importance >= 0.7?                                         │
│       │                                                      │
│   Yes ↓           No ↓                                       │
│   ┌─────────────┐  ┌─────────────┐                           │
│   │  EPISODIC   │  │   WORKING   │                           │
│   │   Memory    │  │   Memory    │                           │
│   │ (permanent) │  │ (temp, FIFO)│                           │
│   └─────────────┘  └─────────────┘                           │
│                           │                                  │
│                    consolidate() at year end                 │
│                           │                                  │
│                    importance >= 0.7?                        │
│                           ↓                                  │
│                    Move to Episodic                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Retrieval Mechanism

### 3.1 Retrieval Scoring

```python
def retrieve(self, top_k: int = 5, current_year: int = 0) -> List[str]:
    """
    Two-phase retrieval:
    1. Working Memory (priority - most recent/important)
    2. Episodic Memory (decay-weighted importance)
    """
    
    # Phase 1: Working Memory
    # Scoring: timestamp (recent first) × importance
    working_scored = sorted(
        self._working,
        key=lambda x: (x.timestamp, x.importance),
        reverse=True
    )
    
    # Phase 2: Episodic Memory (supplement)
    # Scoring: decay(years_passed) × importance
    for item in self._episodic:
        years_passed = current_year - item.year
        decay = DECAY_RATE ** years_passed  # 0.95^years
        score = decay * importance
```

### 3.2 Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Maximum memories returned |
| `current_year` | 0 | For decay calculation |
| `DECAY_RATE` | 0.95 | Per-year decay factor |

### 3.3 Retrieval as Tool (Skill Registry)

```yaml
# skill_registry.yaml
memory_tools:
  - skill_id: retrieve_memory
    description: "Retrieve relevant past experiences for decision making"
    eligible_agent_types: ["*"]  # All agents can use
    implementation_mapping: "memory.retrieve"
    parameters:
      top_k: 5
```

---

## 4. Memory Operations

### 4.1 Add Operations

| Method | Auto-Routes To | Importance Threshold |
|--------|----------------|---------------------|
| `add_experience()` | Working or Episodic | >= 0.7 → Episodic |
| `add_working()` | Working only | N/A |
| `add_episodic()` | Episodic only | N/A |
| `update_after_flood()` | Episodic | Fixed: 0.9 |
| `update_after_decision()` | Working | 0.7 (if not do_nothing) |

### 4.2 Event-Driven Memory Updates

```python
# After flood event
if damage > 0:
    agent.memory.update_after_flood(damage=50000, year=3)
    # → Episodic: "Year 3: A flood occurred causing $50,000 in damages"

# After decision
agent.memory.update_after_decision(decision="buy_insurance", year=3)
    # → Working: "Year 3: I decided to buy_insurance" (importance=0.7)

agent.memory.update_after_decision(decision="do_nothing", year=4)
    # → Working: "Year 4: I decided to do_nothing" (importance=0.3)
```

### 4.3 Consolidation (Year-End)

```python
def consolidate(self) -> int:
    """Move high-importance working memories to episodic."""
    transferred = 0
    for item in self._working:
        if item.importance >= CONSOLIDATION_THRESHOLD:
            self._episodic.append(item)
            transferred += 1
    # Working memory remains intact (will evict naturally)
    return transferred
```

---

## 5. Configuration

### 5.1 Memory Parameters

```python
class CognitiveMemory:
    # Capacity
    WORKING_CAPACITY = 10      # Max working memory items
    EPISODIC_CAPACITY = 50     # Max episodic memory items
    
    # Consolidation
    CONSOLIDATION_THRESHOLD = 0.7  # Move to episodic if >= this
    
    # Retrieval
    DECAY_RATE = 0.95          # Per-year decay for episodic
```

### 5.2 Importance Guidelines

| Event Type | Suggested Importance | Destination |
|------------|---------------------|-------------|
| Major flood (damage > $50k) | 0.9 | Episodic |
| Minor flood (damage < $10k) | 0.6 | Working |
| Elevation decision | 0.8 | Episodic |
| Insurance decision | 0.7 | Episodic |
| Do nothing | 0.3 | Working |
| Policy change observation | 0.5 | Working |
| Neighbor adaptation | 0.4 | Working |

---

## 6. Integration Points

### 6.1 With Prompt (Context)

```python
# prompts.py
def build_household_prompt(state, context, memory: CognitiveMemory):
    memories = memory.retrieve(top_k=5, current_year=context['year'])
    
    prompt = f"""
=== YOUR MEMORY ===
{chr(10).join(f'- {m}' for m in memories)}
"""
```

### 6.2 With Audit (Logging)

```python
# audit_writer.py
trace = {
    "context": {
        "memory_retrieved": memory.retrieve(top_k=5),
        "memory_working_count": len(memory._working),
        "memory_episodic_count": len(memory._episodic)
    }
}
```

### 6.3 With State (Persistence)

```python
# Memory is part of agent state
class HouseholdAgent:
    state: HouseholdAgentState
    memory: CognitiveMemory  # Persistent across years
    
    def serialize(self) -> Dict:
        return {
            "state": asdict(self.state),
            "memory": {
                "working": [asdict(m) for m in self.memory._working],
                "episodic": [asdict(m) for m in self.memory._episodic]
            }
        }
```

---

## 7. Test Specification

### 7.1 Unit Tests

```python
def test_memory_add_working():
    """Test working memory add and eviction."""
    mem = CognitiveMemory(agent_id="H001")
    
    # Add items up to capacity
    for i in range(12):
        mem.add_working(f"Event {i}", importance=0.5)
    
    assert len(mem._working) == 10  # Capacity limit
    assert "Event 0" not in [m.content for m in mem._working]  # Evicted

def test_memory_add_episodic():
    """Test episodic memory add."""
    mem = CognitiveMemory(agent_id="H001")
    mem.add_episodic("Major flood", importance=0.9, year=3)
    
    assert len(mem._episodic) == 1
    assert mem._episodic[0].tags == []

def test_memory_consolidation():
    """Test transfer from working to episodic."""
    mem = CognitiveMemory(agent_id="H001")
    mem.add_working("Low importance", importance=0.3)
    mem.add_working("High importance", importance=0.8)
    
    transferred = mem.consolidate()
    assert transferred == 1
    assert len(mem._episodic) == 1

def test_memory_retrieval_priority():
    """Test working memory takes priority."""
    mem = CognitiveMemory(agent_id="H001")
    mem.add_episodic("Old event", importance=0.9, year=1)
    mem.add_working("Recent event", importance=0.5, year=5)
    
    retrieved = mem.retrieve(top_k=2, current_year=5)
    assert retrieved[0] == "Recent event"  # Working first

def test_memory_retrieval_decay():
    """Test episodic memory decay."""
    mem = CognitiveMemory(agent_id="H001")
    mem.add_episodic("Year 1 event", importance=0.9, year=1)
    mem.add_episodic("Year 10 event", importance=0.5, year=10)
    
    retrieved = mem.retrieve(top_k=2, current_year=10)
    # Year 10 event should rank higher despite lower importance
    # due to less decay
```

### 7.2 Integration Tests

```python
def test_memory_with_flood_decision_cycle():
    """Test full cycle: flood → memory → decision → memory."""
    agent = HouseholdAgent(state=..., memory=CognitiveMemory("H001"))
    
    # Year 1: Flood occurs
    agent.memory.update_after_flood(damage=50000, year=1)
    assert len(agent.memory._episodic) == 1
    
    # Year 2: Agent retrieves memory for decision
    memories = agent.memory.retrieve(top_k=5, current_year=2)
    assert "flood" in memories[0].lower()
    
    # Year 2: Agent decides to buy insurance
    agent.memory.update_after_decision("buy_insurance", year=2)
    assert len(agent.memory._working) == 1

def test_memory_serialization():
    """Test memory can be saved and restored."""
    mem = CognitiveMemory(agent_id="H001")
    mem.add_episodic("Event 1", importance=0.9, year=1)
    mem.add_working("Event 2", importance=0.5, year=2)
    
    # Serialize
    data = {
        "working": [{"content": m.content, "importance": m.importance} 
                    for m in mem._working],
        "episodic": [{"content": m.content, "importance": m.importance} 
                     for m in mem._episodic]
    }
    
    # Deserialize
    new_mem = CognitiveMemory(agent_id="H001")
    for item in data["working"]:
        new_mem.add_working(item["content"], item["importance"])
    for item in data["episodic"]:
        new_mem.add_episodic(item["content"], item["importance"])
    
    assert len(new_mem._working) == 1
    assert len(new_mem._episodic) == 1
```

---

## 8. Implementation Checklist

- [x] `MemoryItem` dataclass with metadata
- [x] `CognitiveMemory.add_working()` with FIFO eviction
- [x] `CognitiveMemory.add_episodic()` with capacity control
- [x] `CognitiveMemory.consolidate()` for year-end transfer
- [x] `CognitiveMemory.retrieve()` with decay scoring
- [x] `update_after_flood()` convenience method
- [x] `update_after_decision()` convenience method
- [x] `format_for_prompt()` for LLM integration
- [x] `to_list()` for ContextBuilder compatibility
- [ ] Memory serialization/deserialization
- [ ] Tag-based retrieval (future)
- [ ] Query-based retrieval (future)
