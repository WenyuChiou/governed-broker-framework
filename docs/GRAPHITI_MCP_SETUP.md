# Graphiti MCP Setup Guide

## What is Graphiti MCP?

Graphiti is a **temporal knowledge graph** MCP server that provides:
- Persistent memory across conversations
- Entity and relationship management
- Semantic search capabilities
- Multi-project isolation via `group_id`

## Prerequisites

1. **Docker Desktop** installed and running
2. **Python 3.10+** (optional, for local dev)
3. **OpenAI API Key** or other LLM provider keys

## Quick Setup (Docker)

### Step 1: Start FalkorDB (Graph Database)

```bash
docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
```

### Step 2: Start Graphiti MCP Server

```bash
docker run -d --name graphiti-mcp \
  -p 8000:8000 \
  -e NEO4J_URI=redis://host.docker.internal:6379 \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e GRAPHITI_GROUP_ID=llm-abm \
  graphiti/mcp-server:latest
```

### Step 3: Configure Your IDE

#### For VS Code / Gemini Code Assist

Add to `.vscode/settings.json` or user settings:

```json
{
  "gemini.mcpServers": {
    "graphiti": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

#### For Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "graphiti": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp/"]
    }
  }
}
```

## Available Tools (After Setup)

| Tool | Description |
|------|-------------|
| `add_episode` | Add text/data to knowledge graph |
| `search_facts` | Semantic search for facts |
| `search_nodes` | Search entity nodes |
| `get_entity` | Get specific entity details |
| `delete_episode` | Remove episode from graph |
| `clear_graph` | Clear all data (use with caution) |

## Multi-Project Support

Use `group_id` to isolate different projects:

```yaml
# mcp-projects.yaml
projects:
  - name: llm-abm-flood
    group_id: flood-adaptation
  - name: llm-abm-climate
    group_id: climate-migration
```

## Integration with Governed Broker Framework

Graphiti can enhance the framework by:

1. **Agent Memory Persistence** - Store agent memories in graph format
2. **Cross-Session Context** - Maintain context across simulation runs
3. **Relationship Tracking** - Map agent-neighbor-institution relationships
4. **Temporal Queries** - "What did Agent_42 decide in Year 5?"

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Check Docker container is running |
| 401 Unauthorized | Verify API key in environment |
| Slow queries | Rebuild indices with `rebuild_indices` |

## References

- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [MCP Specification](https://modelcontextprotocol.io/)
- [FalkorDB Docs](https://docs.falkordb.com/)
