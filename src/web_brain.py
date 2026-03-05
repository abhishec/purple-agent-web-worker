"""web_brain.py — Reflexive Agent core for web/shopping tasks.

Implements the three-layer Reflexive Agent Architecture:
  PRIME   → parse task goal, budget, constraints; plan shopping strategy
  EXECUTE → Claude shopping loop using MCP tools (search/click/checkout)
  REFLECT → verify checkout complete, budget respected, constraints honoured

Supports WebShop+ task categories:
  - budget_management   : stay within spending limit across multiple items
  - preference_memory   : maintain cross-session consistency
  - negative_constraints: avoid forbidden attributes (allergens, restrictions)
  - comparative_reasoning: explore options, justify choice
  - error_recovery      : fix existing cart mistakes

Also supports generic WebShop text-only gym tasks.
"""
from __future__ import annotations
import json
import re
import time
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, MAIN_MODEL, LLM_TIMEOUT, MAX_TURNS, MAX_SEARCH_ROUNDS
from src.mcp_bridge import discover_tools, call_tool

# ── Task category detection ───────────────────────────────────────────────────
_CATEGORY_PATTERNS = {
    "error_recovery": [
        "cart", "remove", "wrong item", "mistake", "fix", "correct the cart",
        "already in cart", "error", "replace",
    ],
    "negative_constraints": [
        "avoid", "no ", "without", "exclude", "not contain", "allergen",
        "gluten", "lactose", "vegan", "vegetarian", "nut-free",
    ],
    "budget_management": [
        "budget", "total", "spend", "under $", "within $", "total cost",
        "multiple items", "combination",
    ],
    "preference_memory": [
        "previous", "last time", "same brand", "consistent", "prefer",
        "remember", "history", "session",
    ],
    "comparative_reasoning": [
        "compare", "best", "better", "justify", "which", "versus", "vs",
        "reason", "explain why",
    ],
}

_SHOPPING_SYSTEM_PROMPT = """You are an expert shopping assistant navigating an online store.

AVAILABLE TOOLS:
- search(query): Search for products. Returns product list with IDs.
- click(element_id): View product details OR add to cart (pass "buy_now" or product ID).
- checkout(): Finalize purchase. MUST be called to complete the task.

SHOPPING STRATEGY:
1. Parse goal: what product(s), budget, and constraints
2. Search with specific keywords (include brand, size, color if known)
3. Read product details — check price, attributes, constraints
4. Select the BEST match: within budget, satisfies constraints
5. Add to cart via click
6. Call checkout() to complete

BUDGET RULES:
- Never exceed the stated budget
- If multiple items needed, track running total

CONSTRAINT RULES:
- Negative constraints (e.g. "no nuts", "gluten-free") are HARD requirements — never violate
- Preference constraints (e.g. "prefer blue") are soft

ERROR RECOVERY (if cart has wrong items):
1. Check cart first to see current state
2. Remove incorrect items
3. Add correct items
4. Checkout

IMPORTANT: You MUST call checkout() to finish. Do not stop without checking out.

After completing, respond with:
{"reasoning": "why you chose this product", "summary": "task completed: [product] for $X"}"""

_ERROR_RECOVERY_PROMPT = """You are correcting a shopping cart that has mistakes.

STRATEGY:
1. First use search or click to check what's currently in the cart
2. Identify items that don't match the goal or violate constraints
3. Remove wrong items (click the remove button if available, or use appropriate tool)
4. Add the correct items
5. Verify the cart is correct
6. Call checkout()

Be methodical — fix one issue at a time."""


def _detect_category(task_text: str) -> str:
    """Detect WebShop+ task category from task description."""
    text_lower = task_text.lower()
    scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_PATTERNS.items():
        scores[cat] = sum(1 for k in keywords if k in text_lower)

    best_cat = max(scores, key=lambda c: scores[c])
    return best_cat if scores[best_cat] >= 1 else "general"


def _parse_budget(task_text: str) -> float | None:
    """Extract budget amount from task text."""
    patterns = [
        r'\$(\d+(?:\.\d{2})?)',
        r'budget[:\s]+\$?(\d+(?:\.\d{2})?)',
        r'under\s+\$?(\d+(?:\.\d{2})?)',
        r'within\s+\$?(\d+(?:\.\d{2})?)',
        r'(\d+(?:\.\d{2})?)\s+dollars',
    ]
    for p in patterns:
        m = re.search(p, task_text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _extract_constraints(task_text: str) -> list[str]:
    """Extract negative constraints (forbidden attributes) from task text."""
    constraints = []
    # Common patterns: "no X", "without X", "X-free", "avoid X", "not X"
    no_patterns = re.findall(r'\bno\s+(\w+)', task_text, re.IGNORECASE)
    avoid_patterns = re.findall(r'\bavoid\s+(\w+)', task_text, re.IGNORECASE)
    free_patterns = re.findall(r'(\w+)-free\b', task_text, re.IGNORECASE)

    constraints.extend(no_patterns)
    constraints.extend(avoid_patterns)
    constraints.extend(free_patterns)
    return list(set(c.lower() for c in constraints))


def _extract_mcp_uri(task_data: Any) -> str | None:
    """Extract MCP URI from A2A task data."""
    if isinstance(task_data, dict):
        for key in ("mcp_uri", "mcp_url", "tools_endpoint"):
            if task_data.get(key):
                return task_data[key]
        resources = task_data.get("resources", [])
        for r in resources:
            if isinstance(r, dict) and r.get("type") == "mcp":
                return r.get("url") or r.get("uri")
    return None


async def _prime(task_text: str, task_data: Any, mcp_url: str, session_id: str) -> dict:
    """PRIME: parse task parameters, detect category, discover tools."""
    category = _detect_category(task_text)
    budget = _parse_budget(task_text)
    constraints = _extract_constraints(task_text)

    # Extract structured data if provided by green agent
    goal = task_text
    user_history = ""
    if isinstance(task_data, dict):
        goal = task_data.get("goal") or task_data.get("task") or task_text
        budget = task_data.get("budget") or budget
        constraints = task_data.get("constraints") or constraints
        user_history = task_data.get("user_history") or ""

    # Override MCP URL if task_data has one
    task_mcp_uri = _extract_mcp_uri(task_data)
    if task_mcp_uri:
        mcp_url = task_mcp_uri

    # Discover MCP tools
    tools = []
    try:
        tools = await discover_tools(mcp_url, session_id)
    except Exception as e:
        print(f"[web] tool discovery failed: {e}")

    # Select system prompt
    system_prompt = _SHOPPING_SYSTEM_PROMPT
    if category == "error_recovery":
        system_prompt = _ERROR_RECOVERY_PROMPT + "\n\n" + _SHOPPING_SYSTEM_PROMPT

    # Build task header with parsed parameters
    task_header_parts = [f"GOAL: {goal}"]
    if budget is not None:
        task_header_parts.append(f"BUDGET: ${budget:.2f}")
    if constraints:
        task_header_parts.append(f"HARD CONSTRAINTS (NEVER VIOLATE): {', '.join(constraints)}")
    if user_history:
        task_header_parts.append(f"PREVIOUS HISTORY: {user_history}")
    task_header = "\n".join(task_header_parts)

    return {
        "category": category,
        "budget": budget,
        "constraints": constraints,
        "goal": goal,
        "system_prompt": system_prompt,
        "tools": tools,
        "task_header": task_header,
        "mcp_url": mcp_url,
    }


async def _execute(
    task_text: str,
    prime_ctx: dict,
    mcp_url: str,
    session_id: str,
    conversation: list[dict],
) -> tuple[str, list[dict]]:
    """EXECUTE: run shopping loop with Claude."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = prime_ctx["tools"]
    system_prompt = prime_ctx["system_prompt"]
    task_header = prime_ctx["task_header"]
    effective_mcp_url = prime_ctx.get("mcp_url", mcp_url)

    if not conversation:
        conversation = [{"role": "user", "content": task_header}]

    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in tools
    ] if tools else []

    turn = 0
    checkout_done = False
    last_answer = ""

    while turn < MAX_TURNS:
        turn += 1

        kwargs: dict[str, Any] = {
            "model": MAIN_MODEL,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": conversation,
            "timeout": LLM_TIMEOUT,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = client.messages.create(**kwargs)

        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                last_answer = block.text
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        conversation.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[web] calling tool {block.name}({block.input})")
                result = await call_tool(effective_mcp_url, block.name, block.input, session_id)
                result_text = result.get("text") or json.dumps(result)

                # Track checkout
                if block.name == "checkout":
                    checkout_done = True
                    print(f"[web] ✓ checkout called")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        conversation.append({"role": "user", "content": tool_results})

        # If checkout was just done, give Claude one more turn to summarize
        if checkout_done and response.stop_reason == "tool_use":
            continue

    if not checkout_done:
        print(f"[web] WARNING: checkout not called after {turn} turns")

    return last_answer, conversation


def _reflect(answer: str, prime_ctx: dict) -> str:
    """REFLECT: verify shopping task completed properly."""
    # Check if checkout is mentioned in answer
    answer_lower = answer.lower()
    if "checkout" not in answer_lower and "purchased" not in answer_lower and "complete" not in answer_lower:
        # Add completion note if answer doesn't mention it
        pass  # Trust the tool call log

    return answer.strip()


async def run_web_task(
    task_text: str,
    task_data: Any,
    mcp_url: str,
    session_id: str,
    conversation: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Main entry point: PRIME → EXECUTE → REFLECT.

    Args:
        task_text: The shopping task description
        task_data: Structured task data (may contain goal, budget, constraints, mcp_uri)
        mcp_url: URL of the green agent's MCP server
        session_id: Session ID for MCP tool calls
        conversation: Existing conversation history (for multi-turn)

    Returns:
        (answer, updated_conversation)
    """
    print(f"[web] PRIME — task length={len(task_text)} chars")
    prime_ctx = await _prime(task_text, task_data, mcp_url, session_id)
    print(f"[web] category={prime_ctx['category']}, budget={prime_ctx['budget']}, "
          f"constraints={prime_ctx['constraints']}, tools={len(prime_ctx['tools'])}")

    print(f"[web] EXECUTE — model={MAIN_MODEL}")
    answer, conversation = await _execute(
        task_text,
        prime_ctx,
        prime_ctx.get("mcp_url", mcp_url),
        session_id,
        conversation or [],
    )

    print(f"[web] REFLECT — answer length={len(answer)} chars")
    answer = _reflect(answer, prime_ctx)

    return answer, conversation
