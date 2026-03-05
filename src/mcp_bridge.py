"""mcp_bridge.py — MCP JSON-RPC 2.0 client for agent-web.

Discovers and calls tools on green-agent MCP server.
Protocol: POST /mcp?session_id=... with JSON-RPC 2.0 payload.
WebShop+ tools: search(query), click(element_id), checkout()
"""
from __future__ import annotations
import httpx
from src.config import TOOL_TIMEOUT


async def discover_tools(tools_endpoint: str, session_id: str = "") -> list[dict]:
    """Discover MCP tools from green agent."""
    url = f"{tools_endpoint}/mcp"
    if session_id:
        url = f"{url}?session_id={session_id}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    tools = data.get("result", {}).get("tools", [])
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema") or t.get("input_schema") or {
                "type": "object", "properties": {}
            },
        }
        for t in tools
    ]


async def call_tool(
    tools_endpoint: str,
    tool_name: str,
    params: dict,
    session_id: str,
) -> dict:
    """Call a tool on the green agent's MCP server."""
    url = f"{tools_endpoint}/mcp"
    if session_id:
        url = f"{url}?session_id={session_id}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": params},
    }
    try:
        async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        result = data.get("result", data)
        if "content" in result and isinstance(result["content"], list):
            texts = [c.get("text", "") for c in result["content"] if c.get("type") == "text"]
            return {"text": "\n".join(texts), "raw": result}
        return result
    except Exception as e:
        return {"error": str(e)}
