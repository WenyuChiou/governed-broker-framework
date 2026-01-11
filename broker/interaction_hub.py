"""
Interaction Hub Component - PR 2
Handles information diffusion across Institutional, Social, and Spatial tiers.
"""
from typing import List, Dict, Any, Optional
from .social_graph import SocialGraph

class InteractionHub:
    """
    Manages the 'Worldview' of agents by aggregating tiered information.
    """
    def __init__(self, graph: SocialGraph):
        self.graph = graph

    def get_spatial_context(self, agent_id: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 1 (Spatial): Aggregated observation of neighbors."""
        neighbor_ids = self.graph.get_neighbors(agent_id)
        if not neighbor_ids:
            return {"elevated_pct": 0, "relocated_pct": 0}
        
        total = len(neighbor_ids)
        elevated = sum(1 for nid in neighbor_ids if getattr(agents[nid], 'elevated', False))
        relocated = sum(1 for nid in neighbor_ids if getattr(agents[nid], 'relocated', False))
        
        return {
            "neighbor_count": total,
            "elevated_pct": round((elevated / total) * 100),
            "relocated_pct": round((relocated / total) * 100)
        }

    def get_social_context(self, agent_id: str, agents: Dict[str, Any], max_gossip: int = 2) -> List[str]:
        """Tier 1 (Social): Gossip/Shared snippets from neighbor memories."""
        neighbor_ids = self.graph.get_neighbors(agent_id)
        gossip = []
        
        # Filter neighbors who have recent memory items
        chatty_neighbors = [nid for nid in neighbor_ids if agents[nid].memory]
        if not chatty_neighbors:
            return []
            
        # Select random snippets from neighbors
        sample_size = min(len(chatty_neighbors), max_gossip)
        for nid in random.sample(chatty_neighbors, sample_size):
            last_event = agents[nid].memory[-1]
            gossip.append(f"Neighbor {nid} mentioned: '{last_event}'")
            
        return gossip

    def build_tiered_context(self, agent_id: str, agents: Dict[str, Any], global_news: List[str] = None) -> Dict[str, Any]:
        """Aggregate all tiers into a unified context slice."""
        return {
            "personal": {
                "id": agent_id,
                "memory": agents[agent_id].memory,
                "status": agents[agent_id].get_adaptation_status() if hasattr(agents[agent_id], 'get_adaptation_status') else {}
            },
            "local": {
                "spatial": self.get_spatial_context(agent_id, agents),
                "social": self.get_social_context(agent_id, agents)
            },
            "global": global_news or []
        }

import random # Required for sampling
