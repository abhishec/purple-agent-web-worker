"""server.py — A2A JSON-RPC 2.0 server for agent-web.

Implements the purple-agent A2A protocol:
  POST /  — A2A JSON-RPC 2.0 (tasks/send, tasks/get, tasks/sendSubscribe)
  GET  /  — agent card
  GET  /health — liveness check

Architecture: Reflexive Agent — PRIME → EXECUTE → REFLECT
Domain: Web Agent / Computer Use (WebShop+, WebShop, general web tasks)
"""
from __future__ import annotations
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.web_brain import run_web_task
from src.config import GREEN_AGENT_MCP_URL, PORT

app = FastAPI(title="agent-web", version="1.0.0")

_sessions: dict[str, dict] = {}
_SESSION_TTL: int = 3600


def _evict_stale_sessions() -> None:
    now = time.time()
    stale = [k for k, v in _sessions.items() if now - v.get("ts", 0) > _SESSION_TTL]
    for k in stale:
        del _sessions[k]


def _agent_card() -> dict:
    base_url = os.environ.get("AGENT_URL", f"http://localhost:{PORT}")
    return {
        "name": "Web & Shopping Agent",
        "description": (
            "Purple web agent built on Reflexive Agent Architecture. "
            "Handles online shopping, web navigation, and e-commerce tasks. "
            "Supports WebShop+: budget management, constraint filtering, "
            "preference memory, comparative reasoning, and error recovery."
        ),
        "url": base_url,
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "shopping",
                "name": "Product Shopping & Web Navigation",
                "description": (
                    "Search, compare and purchase products. Handles budget limits, "
                    "negative constraints, cart corrections, and multi-step checkout."
                ),
                "tags": ["shopping", "webshop", "ecommerce", "web", "search"],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card_wellknown():
    return JSONResponse(_agent_card())


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "web", "version": "1.0.0"}


@app.get("/")
async def root_get():
    return JSONResponse(_agent_card())


@app.post("/")
async def root_post(request: Request):
    """Main A2A JSON-RPC 2.0 endpoint."""
    _evict_stale_sessions()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
            status_code=400,
        )

    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id", str(uuid.uuid4()))

    if method in ("tasks/send", "message/send", "tasks/sendSubscribe"):
        return await _handle_task(params, req_id)

    if method == "tasks/get":
        task_id = params.get("id", "")
        session = _sessions.get(task_id)
        if session and session.get("result"):
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "id": task_id,
                    "status": {"state": "completed"},
                    "artifacts": [{"parts": [{"type": "text", "text": session["result"]}]}],
                },
            })
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"id": task_id, "status": {"state": "working"}},
        })

    if method == "agent/getCard":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": _agent_card()})

    return JSONResponse({
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": f"Method not found: {method}"},
        "id": req_id,
    })


async def _handle_task(params: dict, req_id: Any) -> JSONResponse:
    """Core task handler — calls PRIME → EXECUTE → REFLECT."""
    task_id = params.get("id") or str(uuid.uuid4())
    context_id = params.get("contextId") or task_id

    messages = params.get("message", {})
    if isinstance(messages, dict):
        parts = messages.get("parts", [])
    elif isinstance(messages, list):
        parts = messages
    else:
        parts = []

    task_text = ""
    task_data: Any = None
    mcp_url = GREEN_AGENT_MCP_URL
    session_id = context_id

    for part in parts:
        if isinstance(part, dict):
            if part.get("type") == "text":
                task_text = part.get("text", "")
            elif part.get("type") == "data":
                task_data = part.get("data", {})
                if isinstance(task_data, dict):
                    for key in ("mcp_uri", "mcp_url", "tools_endpoint"):
                        if task_data.get(key):
                            mcp_url = task_data[key]
                            break
                    for r in task_data.get("resources", []):
                        if isinstance(r, dict) and r.get("type") == "mcp":
                            mcp_url = r.get("url") or r.get("uri") or mcp_url
                            break

    if not task_text and isinstance(params.get("message"), str):
        task_text = params["message"]

    session = _sessions.get(context_id, {"conversation": [], "ts": time.time()})

    metadata = params.get("metadata", {})
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id") or session_id
        if metadata.get("mcp_url"):
            mcp_url = metadata["mcp_url"]

    print(f"[web] task_id={task_id} ctx={context_id} mcp={mcp_url}")
    print(f"[web] task_text={task_text[:120]}...")

    try:
        answer, conversation = await run_web_task(
            task_text=task_text,
            task_data=task_data,
            mcp_url=mcp_url,
            session_id=session_id,
            conversation=session["conversation"],
        )
    except Exception as e:
        print(f"[web] ERROR: {e}")
        import traceback
        traceback.print_exc()
        answer = f"Web task encountered an error: {str(e)}"
        conversation = session["conversation"]

    session["conversation"] = conversation
    session["result"] = answer
    session["ts"] = time.time()
    _sessions[context_id] = session

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "parts": [{"type": "text", "text": answer}],
                    "index": 0,
                }
            ],
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
