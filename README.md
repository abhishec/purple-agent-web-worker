# Web & Shopping AI Worker

Purple web agent for the **AgentX Phase 2 — Computer Use & Web Agent Track** (2nd Sprint, March 23 – April 12, 2026).

Built on the **Reflexive Agent Architecture** — the same dual-process cognitive design that achieved **3/3 (100%)** on τ²-Bench airline domain in Sprint 1.

---

## Architecture

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME  ← Reflex Layer
    ├── Task category detection (budget / constraints / error-recovery / comparative)
    ├── Budget extraction & constraint parsing
    ├── MCP tool discovery (green agent's tools)
    ├── MCP URI extraction from A2A resources
    └── Shopping strategy injection
        │
        ▼
    EXECUTE  ← LLM Cortex (Claude)
    ├── Shopping loop: search → click → add to cart → checkout
    ├── Budget tracking (never exceed limit)
    ├── Constraint filtering (hard constraints enforced)
    └── Checkout completion (always calls checkout tool)
        │
        ▼
    REFLECT  ← Verification Layer
    ├── Checkout confirmation
    ├── Budget compliance check
    └── Constraint violation audit
```

## Supported Task Categories (WebShop+)

| Category | Description |
|----------|-------------|
| Budget Management | Multi-item shopping within spending limits |
| Negative Constraints | Avoid forbidden attributes (allergens, restrictions) |
| Preference Memory | Cross-session consistency and recall |
| Comparative Reasoning | Explore options and justify choices |
| Error Recovery | Fix mistakes in existing cart state |

## Competition Targets

**Primary**: `mpnikhil/webshop-plus-green` — WebShop+ (80 tasks, A2A + MCP)
**Secondary**: `mayi0815/webshop-evaluator` — WebShop (text-only gym)

## Deployment

```bash
docker build -t agent-web .
docker run -p 9010:9010 \
  -e ANTHROPIC_API_KEY=... \
  -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
  agent-web
```

## API

```
GET  /health                         → {"status":"ok"}
GET  /.well-known/agent-card.json   → Agent card
POST /                               → A2A JSON-RPC 2.0 (tasks/send)
```
