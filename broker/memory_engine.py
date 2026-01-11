from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from agents.base_agent import BaseAgent

class MemoryEngine(ABC):
    """
    Abstract Base Class for managing agent memory and retrieval.
    Decouples 'Thinking' (LLM) from 'Retention' (Storage).
    """
    
    @abstractmethod
    def add_memory(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new memory item for an agent."""
        pass

    @abstractmethod
    def retrieve(self, agent: BaseAgent, query: Optional[str] = None, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant memories for an agent.
        
        Args:
            agent: The agent instance (for accessing custom_attributes/demographics).
            query: Optional semantic query for retrieval.
            top_k: Number of items to retrieve.
        """
        pass

    @abstractmethod
    def clear(self, agent_id: str):
        """Reset memory for an agent."""
        pass


class WindowMemoryEngine(MemoryEngine):
    """
    Standard sliding window memory. Returns the last N items.
    """
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.storage: Dict[str, List[str]] = {}

    def add_memory(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        if agent_id not in self.storage:
            self.storage[agent_id] = []
        self.storage[agent_id].append(content)
        # We don't truncate here to allow retrieve() to handle different window sizes if needed,
        # but for simplicity we can truncate to a maximum reasonable buffer.
        self.storage[agent_id] = self.storage[agent_id][-100:] 

    def retrieve(self, agent: BaseAgent, query: Optional[str] = None, top_k: int = 3) -> List[str]:
        mems = self.storage.get(agent.id, [])
        return mems[-top_k:]

    def clear(self, agent_id: str):
        if agent_id in self.storage:
            self.storage[agent_id] = []


class ImportanceMemoryEngine(MemoryEngine):
    """
    Active Retrieval Engine.
    Prioritizes significant events over routine ones.
    """
    def __init__(self, window_size: int = 3, top_k_significant: int = 2):
        self.window_size = window_size
        self.top_k_significant = top_k_significant
        self.storage: Dict[str, List[Dict[str, Any]]] = {}

    def _score_content(self, content: str) -> float:
        """Heuristic scoring based on keyword importance."""
        score = 1.0 # Baseline
        keywords = {
            "flood": 5.0,
            "relocate": 4.0,
            "elevate": 3.0,
            "insurance": 2.0,
            "damage": 4.5,
            "neighborhood": 1.5
        }
        content_lower = content.lower()
        for kw, weight in keywords.items():
            if kw in content_lower:
                score += weight
        return score

    def add_memory(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        if agent_id not in self.storage:
            self.storage[agent_id] = []
        
        score = metadata.get("significance") if metadata else None
        if score is None:
            score = self._score_content(content)
            
        self.storage[agent_id].append({
            "content": content,
            "score": score,
            "timestamp": len(self.storage[agent_id])
        })

    def retrieve(self, agent: BaseAgent, query: Optional[str] = None, top_k: int = 5) -> List[str]:
        mems = self.storage.get(agent.id, [])
        if not mems:
            return []
            
        # 1. Get most recent (Recency)
        recent = mems[-self.window_size:]
        recent_texts = [m["content"] for m in recent]
        
        # 2. Get most significant (Significance) - excluding those already in recent
        others = mems[:-self.window_size]
        significant = sorted(others, key=lambda x: x["score"], reverse=True)
        
        top_sig = []
        for s in significant:
            if s["content"] not in recent_texts:
                top_sig.append(s["content"])
            if len(top_sig) >= self.top_k_significant:
                break
                
        # Combine: Significant events first (for context) then Recent (for continuity)
        # or vice versa? Usually Significant provides the 'Trauma/History' context.
        return top_sig + recent_texts

    def clear(self, agent_id: str):
        self.storage[agent_id] = []
