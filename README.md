# agent-web — BrainOS Mini AI Worker

> **WebShop+ · AgentX Phase 2 — Computer Use & Web Agent Track**
> One of five BrainOS Mini AI Workers — a self-contained web execution cognitive unit built on the **PRIME → EXECUTE → REFLECT** loop with deterministic budget and constraint enforcement.

---

## The Problem

Web and shopping agents fail in ways that are fundamentally different from reasoning failures — they are **compliance failures**.

**Budget overrun.** The agent finds the right products, adds them to the cart, and overshoots the budget by $3. The LLM never tracked spending; it assumed the total was fine. The task fails on a hard constraint the agent never actually checked.

**Constraint bypass.** The product description mentions "synthetic lining" in line 4. The task said "no synthetic materials." The LLM read 200 tokens of product attributes and missed one attribute. Hard constraint silently violated.

**Checkout omission.** The agent found the product, verified the price, added it to the cart — and stopped. It never called the checkout tool. The task required a completed purchase. The LLM decided the shopping was done without executing the final step.

These are not reasoning problems. They are **mechanical enforcement gaps** — things that should be checked deterministically but aren't.

---

## BrainOS Innovation: Deterministic Enforcement Layer

The Web AI Worker wraps every LLM execution turn with deterministic enforcement mechanisms that catch compliance failures before they surface. Budget tracking, constraint verification, and checkout completion are not left to the LLM's judgment — they are enforced mechanically, on every call, at zero LLM cost.

---

## Core Technical Innovations

### 1 — Deterministic Budget Accumulator

Spending is tracked via regex extraction on every tool call — not via LLM estimation or post-hoc reasoning.

On every call whose name contains `add`, `cart`, `buy`, or `purchase`:
1. Extract price from result text: `$(\d+(?:\.\d{2})?)`
2. Accumulate to running total
3. If accumulated spend > budget limit: append `[BUDGET ALERT: $X spent of $Y limit]` to tool result

The LLM sees the budget alert on the next turn and self-corrects. Zero API cost. The accumulator cannot be fooled by LLM reasoning errors about running totals.

### 2 — Constraint Hard-Check (Zero Cost)

After every tool call, result text is scanned for forbidden attributes extracted from the task:

```
task text  →  extract constraint words (allergies, colors, sizes, brands)
tool result →  scan for exact constraint word matches
match found →  append [CONSTRAINT WARNING: "synthetic" found in product description]
```

The LLM sees the warning on the next turn and abandons the product. Pure string matching. Zero API cost. This is the only reliable way to catch constraint violations buried in long product attribute lists.

### 3 — L2 Checkout Contract (Mechanical Completion)

After execution completes, a deterministic check verifies that any tool whose name contains `checkout`, `purchase`, `buy`, or `order` was called. If not, re-run with a forced-checkout directive injected into the conversation:

> *"CRITICAL: You must call the checkout tool now to complete the purchase."*

This catches the most common shopping agent failure — completing the search and add phase but missing the final purchase step — without requiring task redesign or additional prompting.

### 4 — DAAO: Difficulty-Aware Adaptive Orchestration

Tasks are routed to the cheapest capable model before execution begins.

**Fast path (Haiku):** Simple navigation — `"navigate to"`, `"open"`, `"go to"`. Single-step, no shopping logic required.

**Deep path (Sonnet):** Multi-item shopping with budget constraints, comparative reasoning across products, booking flows with date/time parsing, tasks with multiple hard constraints.

Routing is deterministic (keyword + task structure). Zero cost to evaluate.

### 5 — Prefix-Based Sequence Hints (Protocol-Agnostic)

Category-specific tool-call directives injected into every system prompt, using prefixes not tool names:

| Category | Injected Sequence |
|---|---|
| `shopping` | `search_` / `find_` → `view_` / `click_` → `add_` → `checkout_` |
| `booking` | `search_` / `find_` → `view_` / `check_` → `select_` → `confirm_` / `book_` |
| `search` | `search_` / `find_` → `view_` / `get_` → `compare_` → `return_` |
| `task` | `open_` / `navigate_` → `fill_` / `enter_` → `submit_` / `click_` |
| `navigate` | `go_` / `open_` / `navigate_` → `click_` / `select_` |

Works against WebShop+, WebArena, or any shopping MCP server — no tool name assumptions.

### 6 — Recovery Cascade

When a tool returns `"no results"`, `"0 items"`, `"not found"`, or `"empty"`, a recovery hint is appended to the tool result:

> *"[RECOVERY HINT: Try broader search terms. Remove specific constraints and search by category. Try alternative product names.]"*

The most common web agent dead-end — an overspecific product search returning nothing — is automatically recovered without human re-prompting.

### 7 — RL Primer Injection

Top-3 past cases most relevant to the current task (Jaccard keyword overlap + category match) are injected as examples before each execution. The agent sees how it previously handled budget constraints, what search strategies worked for similar products, how it navigated specific checkout flows. No retraining.

---

## Supported Task Categories

| Category | Contracts Applied |
|---|---|
| `shopping` | L2 Checkout + Budget Accumulator + Constraint Check + Sequence Hints |
| `booking` | L2 Confirm Contract + Sequence Hints + Recovery Cascade |
| `search` | Recovery Cascade + Constraint Check + DAAO |
| `task` | Sequence Hints (navigate→fill→submit) |
| `navigate` | DAAO fast path (Haiku only) |

---

## Cognitive Loop: PRIME → EXECUTE → REFLECT

```
PRIME
├── Category detection      (shopping / booking / search / task / navigate)
├── Budget extraction       (regex: "$X", "under X", "at most X")
├── Constraint parsing      (allergies, colors, sizes, forbidden attributes)
├── RL primer injection     (top-3 past cases by keyword + category)
├── DAAO model selection    (Haiku navigation; Sonnet shopping/booking)
├── Sequence hint injection (prefix-based directives per category)
└── MCP tool discovery      (green agent tools fetched at runtime)

EXECUTE
├── Agentic tool loop:  search_ → click_ / view_ → add_ → checkout_
├── Budget accumulator  (deterministic price extraction on every add/buy)
├── Constraint check    (deterministic regex scan on every tool result)
├── Recovery cascade    (empty results → broadening hint)
└── L2 Checkout Contract (shopping without checkout → forced retry)

REFLECT
├── Checkout confirmation audit
├── Budget compliance check (spent vs. limit)
├── Constraint violation re-audit (hard constraints verified)
├── Quality scoring     (checkout +0.25, budget overrun −0.2, violation −0.25)
└── RL case recording   (case_log.json, last 20 entries, keyword-indexed)
```

---

## Competition Targets

- **Primary:** WebShop+ (`mpnikhil/webshop-plus-green`) — 80 tasks, A2A + MCP
- **Secondary:** WebShop (`mayi0815/webshop-evaluator`) — text-only

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI; A2A JSON-RPC 2.0 handler |
| `web_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT |
| `mcp_bridge.py` | MCP HTTP; pre-flight validation; schema patching |
| `config.py` | Environment config; model constants; timeout settings |

---

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
PORT=9012 python3 src/server.py
```

**Docker:**
```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-web:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
           -p 9012:9012 \
           public.ecr.aws/d9m7h3k5/agentbench-web:latest
```

---

## Tech Stack

- **Runtime:** Python 3.11 · FastAPI · uvicorn
- **LLM:** claude-haiku-4-5-20251001 (navigation, DAAO fast path) · claude-sonnet-4-6 (shopping, constraint reasoning)
- **Budget tracking:** Deterministic regex (not LLM-estimated)
- **Constraint enforcement:** Deterministic string matching (not LLM-based)
- **Architecture:** BrainOS PRIME / EXECUTE / REFLECT
- **Core library:** [brainos-core-light](https://github.com/abhishec/brainoscorelight) v0.3.0
  - `DAAO` — zero-LLM model routing (Haiku navigation, Sonnet shopping)
  - `CheckoutContract` — L2 mechanical checkout enforcement
  - `SequenceHints` — prefix-based tool-call directives (shopping/booking/navigate)
  - `RecoveryCascade` — automatic empty-result recovery
  - `Brain` + `Router` — UCB1 strategy bandit + 5-layer memory
- **RL:** Case log (JSON) · quality scoring · RL primer injection
- **Protocol:** A2A JSON-RPC 2.0

---

## License

Apache 2.0
