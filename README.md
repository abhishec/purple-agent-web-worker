# Web & Shopping AI Worker

> One of four **mini AI workers built on BrainOS** — the Reflexive Agent Architecture framework that achieved **3/3 (100%)** on τ²-Bench. Each worker is a lightweight, self-contained cognitive unit that runs the same PRIME → EXECUTE → REFLECT loop tuned to its domain.

**AgentX Phase 2 — Computer Use & Web Agent Track**

---

## What This Worker Does

The Web & Shopping AI Worker connects to any MCP tool server and executes web tasks: shopping, product search, booking, navigation, and form completion. It discovers tools at runtime, classifies the task category, injects budget/constraint scaffolding, runs an agentic search-and-act loop, and verifies checkout completion and budget compliance before returning.

---

## BrainOS Cognitive Loop: PRIME → EXECUTE → REFLECT

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME  ← Reflex Layer
    ├── Category detection      (shopping / booking / search / task / navigate)
    ├── Budget extraction       (regex: "$X", "under X", "at most X")
    ├── Constraint parsing      (allergies, colors, sizes, forbidden attributes)
    ├── RL primer injection     (top-3 past cases by keyword + category relevance)
    ├── DAAO model selection    (Haiku for simple navigation, Sonnet for multi-step shopping)
    ├── Sequence hint injection (prefix-based tool-call directives per category)
    └── MCP tool discovery      (green agent's tools fetched at runtime)
        │
        ▼
    EXECUTE  ← LLM Cortex (DAAO: Haiku → Sonnet)
    ├── Agentic tool loop:  search_ → click_ / view_ → add_ → checkout_
    ├── Budget accumulator  (deterministic: regex price extraction on every add/buy call)
    ├── Constraint check    (deterministic: regex scan of tool results for forbidden attrs)
    ├── Recovery cascade    (empty search results → inject broadening hint)
    └── L2 Checkout Contract (shopping task ends without checkout → forced retry)
        │
        ▼
    REFLECT  ← Verification Layer
    ├── Checkout confirmation audit
    ├── Budget compliance check (spent vs. limit)
    ├── Constraint violation audit (hard constraints re-verified)
    ├── Quality scoring     (0–1 heuristic: checkout +0.25, budget overrun −0.2, violation −0.25)
    └── RL case recording   (case_log.json, last 20 entries, keyword-indexed)
```

---

## Key BrainOS Concepts Applied

### DAAO — Difficulty-Aware Adaptive Orchestration
Routes each task to the cheapest model that can handle it. Haiku handles simple navigation and lookups (`navigate to`, `open`, `go to`). Sonnet handles multi-item shopping, constraint-heavy searches, and comparative reasoning. Reduces cost on navigation tasks while maintaining quality on complex purchase flows.

### RL Primer Injection
Before each task, loads the last 20 completed cases from `case_log.json`. Scores each case by keyword overlap with the current task (Jaccard on 4+ char words) plus category match bonus (+2.0) plus past quality score. Injects the top-3 most relevant cases into the system prompt. The LLM sees its own past successes (e.g., how it found a product under budget, how it handled a constraint) before attempting the current task.

### Prefix-Based Sequence Hints
Injects an ordered tool-call directive into every system prompt. Uses **prefixes** not hardcoded tool names — `search_` or `find_`, then `view_` or `click_`, then `add_`, then `checkout_` — so the directive works across WebShop+, WebArena, or any shopping MCP server. Per-category seeds:

| Category | Sequence |
|---|---|
| `shopping` | search_ / find_ → view_ / click_ → add_ → checkout_ |
| `booking` | search_ / find_ → view_ / check_ → select_ → confirm_ / book_ |
| `search` | search_ / find_ → view_ / get_ → compare_ → return_ |
| `task` | open_ / navigate_ → fill_ / enter_ → submit_ / click_ |
| `navigate` | go_ / open_ / navigate_ → click_ / select_ |

### L2 Checkout Contract
Shopping tasks must end with a checkout. After execution, checks whether any tool whose name contains `checkout`, `purchase`, `buy`, or `order` was called. If not, re-runs the LLM with an explicit forced-checkout directive injected into the conversation: `"CRITICAL: You must call the checkout tool now to complete the purchase."` This catches cases where the LLM found and added items but forgot the final step.

### Budget Accumulator (Deterministic)
Tracks total spend across the task using deterministic regex — not LLM estimation. On every tool call whose name contains `add`, `cart`, `buy`, or `purchase`, extracts the price from the result text using `$(\d+(?:\.\d{2})?)`. Appends a `[BUDGET ALERT]` to the tool result if accumulated spend exceeds the budget limit. The LLM sees the alert on the next turn and adjusts.

### Constraint Hard-Check (Deterministic)
After every tool call, scans the result text for any forbidden attributes (allergens, colors, sizes, brands, etc.) extracted from the task text. If a constraint word appears verbatim in the result, appends a `[CONSTRAINT WARNING]` to the tool result. Zero LLM cost — pure string matching. Prevents the LLM from silently ignoring hard constraints embedded in product descriptions.

### Recovery Cascade
If a tool result contains `"no results"`, `"0 items"`, `"not found"`, or `"empty"`, appends: `"[RECOVERY HINT: Try broader search terms. Remove specific constraints and search by category. Try alternative product names.]"` The LLM sees this hint on the next turn and tries a broader query automatically.

---

## Supported Task Categories

| Category | Description | Key Contracts |
|---|---|---|
| `shopping` | Multi-item purchase within budget | L2 Checkout + Budget Accumulator + Constraint Check |
| `booking` | Flight, hotel, restaurant reservations | L2 Confirm Contract + Date/time parsing |
| `search` | Product research, price comparison | Recovery Cascade + Constraint Check |
| `task` | Form fill, navigation, UI interaction | Sequence Hints only |
| `navigate` | Site navigation, page access | DAAO fast path (Haiku) |

---

## Competition Targets

**Primary**: `mpnikhil/webshop-plus-green` — WebShop+ (80 tasks, A2A + MCP)
**Secondary**: `mayi0815/webshop-evaluator` — WebShop (text-only gym)

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI application; A2A JSON-RPC 2.0 handler |
| `web_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT; all BrainOS concepts |
| `mcp_bridge.py` | MCP tool bridge; pre-flight parameter validation; schema patching |
| `config.py` | Environment configuration; model constants; timeout settings |

---

## Requirements

Python 3.11+

```
fastapi>=0.115
uvicorn[standard]>=0.30
anthropic>=0.34
httpx>=0.27
pydantic>=2.0
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `GREEN_AGENT_MCP_URL` | Yes | — | MCP tool server base URL |
| `FALLBACK_MODEL` | No | `claude-sonnet-4-6` | Primary execution model |
| `FAST_MODEL` | No | `claude-haiku-4-5` | Fast model for DAAO navigation routing |
| `TOOL_TIMEOUT` | No | `10` | Seconds per tool call |
| `TASK_TIMEOUT` | No | `120` | Seconds per task |
| `RL_CACHE_DIR` | No | `/app` | Directory for `case_log.json` |

---

## Docker

```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-web:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
           -p 9010:9010 \
           public.ecr.aws/d9m7h3k5/agentbench-web:latest
```

---

## API

All requests use A2A JSON-RPC 2.0.

| Endpoint | Method | Description |
|---|---|---|
| `/` | POST | `tasks/send` — submit a web/shopping task |
| `/.well-known/agent-card.json` | GET | Agent capability declaration |
| `/health` | GET | Health check → `{"status":"ok","agent":"web"}` |

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "id": "task-001",
  "params": {
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": [{ "text": "Buy a pair of running shoes under $120, size 10, no synthetic materials." }]
    },
    "metadata": {
      "tools_endpoint": "https://mcp.example.com",
      "session_id": "worker-abc"
    }
  }
}
```

---

## Tech Stack

- **Runtime:** Python 3.11, FastAPI, uvicorn
- **LLM:** Anthropic Claude — Haiku for navigation (DAAO fast path); Sonnet for multi-step shopping and constraint reasoning
- **Architecture:** BrainOS PRIME / EXECUTE / REFLECT cognitive loop
- **Budget tracking:** Deterministic regex price extraction (not LLM-based)
- **Constraint enforcement:** Deterministic string matching (not LLM-based)
- **RL:** RL case log (JSON) + quality scoring + RL primer injection
- **Tool bridge:** MCP HTTP with pre-flight validation
- **Storage:** Local JSON (`case_log.json` — last 20 entries, keyword-indexed)

---

## License

Apache 2.0
