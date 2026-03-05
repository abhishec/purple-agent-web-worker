"""web_brain.py — Reflexive Agent core for web/shopping tasks.

Implements the three-layer Reflexive Agent Architecture:
  PRIME   → parse goal/budget/constraints, DAAO model routing, RL primer, sequence hints
  EXECUTE → Claude shopping loop using MCP tools (search/click/checkout)
  REFLECT → L2 checkout contract, budget check, constraint verification, RL recording

New concepts ported from agent-purple + BrainOS (v2):
  - RL primer injection: top-3 relevant past cases injected into system prompt
  - DAAO model routing: Haiku for simple single-item, Sonnet for comparative/error-recovery
  - Sequence hints: "search → view → add_to_cart → checkout" (prefix-based)
  - L2 Checkout Contract: task expects purchase but checkout never called → inject retry
  - Budget accumulator: deterministic spend tracking, not LLM-based
  - Constraint hard-check: before checkout, verify no constraint violated in tool results
  - Recovery cascade: empty search results → broaden query and retry
  - Quality scoring: checkout_done, budget_respected, constraints_satisfied, tool_count
  - RL case log: persistent, keyword-indexed, last 20 entries

Supports WebShop+ task categories:
  - budget_management   : stay within spending limit across multiple items
  - preference_memory   : maintain cross-session consistency
  - negative_constraints: avoid forbidden attributes (allergens, restrictions)
  - comparative_reasoning: explore options, justify choice
  - error_recovery      : fix existing cart mistakes
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, MAIN_MODEL, FAST_MODEL, LLM_TIMEOUT, MAX_TURNS, MAX_SEARCH_ROUNDS
from src.mcp_bridge import discover_tools, call_tool

# ── RL case log ───────────────────────────────────────────────────────────────
_RL_DIR = Path(os.environ.get("RL_CACHE_DIR", "/app"))
_CASE_LOG = _RL_DIR / "web_case_log.json"
_MAX_CASES = 20


def _load_cases() -> list[dict]:
    try:
        with open(_CASE_LOG) as f:
            return json.load(f)
    except Exception:
        return []


def _save_case(case: dict) -> None:
    cases = _load_cases()
    cases.append(case)
    if len(cases) > _MAX_CASES:
        cases = cases[-_MAX_CASES:]
    try:
        _RL_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CASE_LOG, "w") as f:
            json.dump(cases, f, indent=2)
    except Exception as e:
        print(f"[web/rl] save failed: {e}")


def _score_quality(
    answer: str,
    checkout_done: bool,
    budget: float | None,
    budget_spent: float,
    constraints: list[str],
    constraint_violated: bool,
    tool_count: int,
) -> float:
    """Quality heuristic 0-1 for shopping tasks."""
    q = 0.5
    # Checkout is the most critical signal
    if checkout_done:
        q += 0.25
    else:
        q -= 0.3
    # Budget respected
    if budget is not None and budget_spent > 0:
        if budget_spent <= budget:
            q += 0.1
        else:
            q -= 0.2  # Over budget is a major failure
    # Constraints
    if constraints and constraint_violated:
        q -= 0.25
    elif constraints:
        q += 0.05
    # Tool depth
    if tool_count >= 3:
        q += 0.1
    elif tool_count == 0:
        q -= 0.2
    # Answer quality
    if len(answer) < 20:
        q -= 0.1
    return max(0.0, min(1.0, q))


def _build_rl_primer(task_text: str, category: str) -> str:
    """Build RL primer from top-3 most relevant past cases."""
    cases = _load_cases()
    if not cases:
        return ""

    task_words = set(re.findall(r'\b\w{4,}\b', task_text.lower()))

    def relevance(c: dict) -> float:
        kw = set(c.get("keywords", []))
        overlap = len(task_words & kw)
        cat_bonus = 2.0 if c.get("category") == category else 0.0
        return overlap + cat_bonus + c.get("quality", 0.5)

    top = sorted(cases, key=relevance, reverse=True)[:3]
    if not top:
        return ""

    lines = ["LEARNED PATTERNS from past shopping tasks:"]
    for c in top:
        sym = "✓" if c.get("outcome") == "success" else "✗"
        summary = c.get("task_summary", "")[:70]
        worked = c.get("what_worked", "")
        failed = c.get("what_failed", "")
        lines.append(f"  {sym} [{c.get('category', 'general')}] {summary}")
        if worked:
            lines.append(f"     Worked: {worked[:80]}")
        if failed:
            lines.append(f"     Failed: {failed[:80]}")
    return "\n".join(lines)


# ── DAAO: Difficulty-Aware Adaptive Orchestration ─────────────────────────────
def _select_model(category: str, task_text: str, has_tools: bool) -> str:
    """DAAO: route to fastest sufficient model.

    Haiku for:  single-item straightforward purchases, short simple queries
    Sonnet for: comparative reasoning, error recovery, multi-item budget tracking,
                negative constraints (high precision needed)
    """
    # High-precision tasks → always Sonnet
    if category in ("comparative_reasoning", "error_recovery", "negative_constraints"):
        return MAIN_MODEL

    text_lower = task_text.lower()
    # Comparative or complex keywords → Sonnet
    complex_kw = ("compare", "best", "difference", "justify", "which one", "vs ", "versus",
                  "multiple", "several", "combination", "fix", "wrong", "mistake")
    if any(k in text_lower for k in complex_kw):
        return MAIN_MODEL

    # Simple single-item without constraints → Haiku can handle it
    words = task_text.split()
    if len(words) < 20 and category not in ("budget_management",) and not has_tools:
        return FAST_MODEL

    # Default: Sonnet for shopping (checkout requires precision)
    return MAIN_MODEL


# ── Task category detection ───────────────────────────────────────────────────
_CATEGORY_PATTERNS: dict[str, list[str]] = {
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


def _detect_category(task_text: str) -> str:
    text_lower = task_text.lower()
    scores = {
        cat: sum(1 for k in keywords if k in text_lower)
        for cat, keywords in _CATEGORY_PATTERNS.items()
    }
    best_cat = max(scores, key=lambda c: scores[c])
    return best_cat if scores[best_cat] >= 1 else "general"


def _parse_budget(task_text: str) -> float | None:
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
    constraints: list[str] = []
    no_patterns = re.findall(r'\bno\s+(\w+)', task_text, re.IGNORECASE)
    avoid_patterns = re.findall(r'\bavoid\s+(\w+)', task_text, re.IGNORECASE)
    free_patterns = re.findall(r'(\w+)-free\b', task_text, re.IGNORECASE)
    without_patterns = re.findall(r'\bwithout\s+(\w+)', task_text, re.IGNORECASE)
    constraints.extend(no_patterns + avoid_patterns + free_patterns + without_patterns)
    return list(set(c.lower() for c in constraints))


def _extract_mcp_uri(task_data: Any) -> str | None:
    if isinstance(task_data, dict):
        for key in ("mcp_uri", "mcp_url", "tools_endpoint"):
            if task_data.get(key):
                return task_data[key]
        for r in task_data.get("resources", []):
            if isinstance(r, dict) and r.get("type") == "mcp":
                return r.get("url") or r.get("uri")
    return None


# ── Sequence hints ────────────────────────────────────────────────────────────
_SEQUENCE_HINTS: dict[str, str] = {
    "general": (
        "RECOMMENDED SHOPPING SEQUENCE:\n"
        "1. search_ → find products matching the goal\n"
        "2. click_ or view_ → inspect product details, price, attributes\n"
        "3. add_ or click_ (buy/cart action) → add to cart\n"
        "4. checkout_ → MUST be called to complete the task"
    ),
    "comparative_reasoning": (
        "RECOMMENDED COMPARISON SEQUENCE:\n"
        "1. search_ → find multiple candidates\n"
        "2. view_ or click_ each candidate → compare price, specs, reviews\n"
        "3. Select best match with explicit justification\n"
        "4. add_ + checkout_ → complete purchase"
    ),
    "error_recovery": (
        "RECOMMENDED CART-FIX SEQUENCE:\n"
        "1. view_cart or check_ → inspect current cart state\n"
        "2. remove_ or click_(remove) → remove incorrect items\n"
        "3. search_ + add_ → add correct items\n"
        "4. verify_ cart → confirm correct state\n"
        "5. checkout_ → complete"
    ),
    "negative_constraints": (
        "CONSTRAINT-SAFE SHOPPING SEQUENCE — START WITH TOOL CALL:\n"
        "Step 1: CALL search_products NOW with product keywords\n"
        "Step 2: CALL view_product on promising results — check material/attributes list carefully\n"
        "Step 3: VERIFY in the view_product result that the FORBIDDEN attribute is NOT present\n"
        "Step 4: CALL add_to_cart ONLY for items that pass the constraint check\n"
        "Step 5: CALL checkout to complete\n"
        "NEVER add an item without first calling view_product to check its attributes."
    ),
    "budget_management": (
        "BUDGET-TRACKING SHOPPING SEQUENCE:\n"
        "1. Parse total budget from goal\n"
        "2. For each item: search_ → view_ → check price → running_total += price\n"
        "3. STOP adding if running_total would exceed budget\n"
        "4. checkout_ with final cart"
    ),
}

# ── System prompts ────────────────────────────────────────────────────────────
_SHOPPING_SYSTEM_PROMPT = """You are an expert shopping assistant navigating an online store using MCP tools.

**CRITICAL: Your FIRST action MUST be a tool call. Do NOT write text before calling a tool.**
Do not describe what you will do. Just do it. Call search_products immediately.

AVAILABLE TOOLS (prefixes indicate purpose):
- search_products : Search for products by query, price, category. Use this FIRST.
- view_product : View full product details including materials, specs, price.
- add_to_cart : Add chosen product to cart.
- checkout : Finalize purchase. MUST be called to complete the task.
- view_cart, remove_from_cart, compare_products : Cart management.

SHOPPING RULES:
1. IMMEDIATELY call search_products — never explain first, just search
2. View product details to verify constraints BEFORE adding to cart
3. Hard constraints CANNOT be violated — verify material/attributes in view_product result
4. Select BEST match: within budget, satisfies all hard constraints
5. Add to cart, then ALWAYS call checkout to complete
6. Budget: NEVER exceed the stated limit

After checkout, give a brief summary: product chosen, price, why it satisfies constraints."""

_ERROR_RECOVERY_PROMPT = """You are correcting a shopping cart that contains mistakes.

APPROACH:
1. First check current cart state
2. Identify items that don't match the goal or violate constraints
3. Remove wrong items (use remove/click tool)
4. Add correct items
5. Verify cart is correct
6. Call checkout

Be methodical — fix one issue at a time. Confirm each change took effect."""


# ── PRIME ─────────────────────────────────────────────────────────────────────
async def _prime(task_text: str, task_data: Any, mcp_url: str, session_id: str) -> dict:
    """PRIME: parse task, detect category, discover tools, route model, build context."""
    category = _detect_category(task_text)
    budget = _parse_budget(task_text)
    constraints = _extract_constraints(task_text)

    # Task_data structured fields
    goal = task_text
    user_history = ""
    if isinstance(task_data, dict):
        goal = task_data.get("goal") or task_data.get("task") or task_text
        budget = task_data.get("budget") or budget
        constraints = task_data.get("constraints") or constraints
        user_history = task_data.get("user_history") or ""

    # Override MCP URL if present
    task_mcp_uri = _extract_mcp_uri(task_data)
    if task_mcp_uri:
        mcp_url = task_mcp_uri

    # Discover MCP tools
    tools: list[dict] = []
    try:
        tools = await discover_tools(mcp_url, session_id)
    except Exception as e:
        print(f"[web] tool discovery failed: {e}")

    has_tools = len(tools) > 0

    # DAAO: select model
    model = _select_model(category, task_text, has_tools)
    print(f"[web] DAAO → model={model} (category={category}, tools={has_tools})")

    # RL primer
    rl_primer = _build_rl_primer(task_text, category)

    # Sequence hint
    seq_hint = _SEQUENCE_HINTS.get(category, _SEQUENCE_HINTS["general"])

    # Build system prompt
    base_prompt = _SHOPPING_SYSTEM_PROMPT
    if category == "error_recovery":
        base_prompt = _ERROR_RECOVERY_PROMPT + "\n\n" + _SHOPPING_SYSTEM_PROMPT

    system_parts = [base_prompt, "", seq_hint]
    if rl_primer:
        system_parts += ["", rl_primer]
    system_prompt = "\n".join(system_parts)

    # Build task header with deterministically parsed parameters
    header_parts = [f"GOAL: {goal}"]
    if budget is not None:
        header_parts.append(f"BUDGET: ${budget:.2f} (HARD LIMIT — do not exceed)")
    if constraints:
        header_parts.append(
            f"HARD CONSTRAINTS (NEVER VIOLATE): {', '.join(constraints)}"
        )
    if user_history:
        header_parts.append(f"USER HISTORY: {user_history}")
    # Force immediate tool use — prevent planning-text on first turn
    header_parts.append("\nACTION: Call search_products NOW with relevant keywords. Do not write text first.")
    task_header = "\n".join(header_parts)

    return {
        "category": category,
        "budget": budget,
        "constraints": constraints,
        "goal": goal,
        "model": model,
        "system_prompt": system_prompt,
        "tools": tools,
        "task_header": task_header,
        "mcp_url": mcp_url,
        "has_tools": has_tools,
    }


# ── Budget accumulator (deterministic, not LLM-based) ────────────────────────
_PRICE_PATTERN = re.compile(
    r'\$(\d+(?:\.\d{2})?)'          # $12.99
    r'|price[:\s]+\$?(\d+\.\d{2})',  # price: 12.99
    re.IGNORECASE,
)


def _extract_price_from_result(result_text: str) -> float:
    """Deterministically extract price from a tool result string."""
    m = _PRICE_PATTERN.search(result_text)
    if m:
        val = m.group(1) or m.group(2)
        try:
            return float(val)
        except ValueError:
            pass
    return 0.0


def _check_constraint_violated(constraints: list[str], result_text: str) -> bool:
    """Deterministic check: does result_text contain any forbidden attribute?"""
    text_lower = result_text.lower()
    for c in constraints:
        if c.lower() in text_lower:
            return True
    return False


# ── EXECUTE ───────────────────────────────────────────────────────────────────
async def _execute(
    task_text: str,
    prime_ctx: dict,
    mcp_url: str,
    session_id: str,
    conversation: list[dict],
    force_checkout: bool = False,
) -> tuple[str, list[dict], int, bool, float, bool]:
    """EXECUTE: shopping loop with Claude.

    Returns:
        (answer, conversation, tool_count, checkout_done, budget_spent, constraint_violated)
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = prime_ctx["tools"]
    system_prompt = prime_ctx["system_prompt"]
    task_header = prime_ctx["task_header"]
    effective_mcp_url = prime_ctx.get("mcp_url", mcp_url)
    model = prime_ctx["model"]
    budget = prime_ctx.get("budget")
    constraints = prime_ctx.get("constraints", [])

    # Build initial message
    initial_content = task_header
    if force_checkout:
        initial_content += (
            "\n\nACTION REQUIRED: The task is not complete — checkout has not been called. "
            "Call the checkout tool NOW to finalize the purchase."
        )

    if not conversation:
        if not initial_content.strip():
            initial_content = task_text or "Please proceed with the shopping task."
        conversation = [{"role": "user", "content": initial_content}]
    elif force_checkout:
        # Inject checkout directive into existing conversation
        conversation.append({"role": "user", "content": initial_content})

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
    constraint_violated = False
    budget_spent = 0.0
    last_answer = ""
    tool_count = 0
    search_turns = 0
    last_search_result = ""

    while turn < MAX_TURNS:
        turn += 1

        # Validate conversation — never send empty user message
        clean_conversation = []
        for msg in conversation:
            content = msg["content"]
            if msg["role"] == "user":
                if isinstance(content, list) and len(content) == 0:
                    continue
                if isinstance(content, str) and not content.strip():
                    continue
            clean_conversation.append(msg)
        if not clean_conversation:
            break

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": clean_conversation,
            "timeout": LLM_TIMEOUT,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            print(f"[web] API error turn {turn}: {e}")
            break

        assistant_content: list[dict] = []
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
        tool_results: list[dict] = []
        for block in response.content:
            if block.type == "tool_use":
                tool_count += 1
                tool_name = block.name.lower()
                print(f"[web] → {block.name}({json.dumps(block.input)[:80]})")

                try:
                    result = await call_tool(effective_mcp_url, block.name, block.input, session_id)
                    result_text = result.get("text") or json.dumps(result)
                except Exception as e:
                    result_text = f"Tool error: {e}"

                # Track checkout
                if any(k in tool_name for k in ("checkout", "buy", "purchase", "order")):
                    checkout_done = True
                    print(f"[web] ✓ checkout called")

                # Track search results for recovery cascade
                if any(k in tool_name for k in ("search", "find", "query")):
                    search_turns += 1
                    last_search_result = result_text
                    # Recovery: if empty result and haven't retried yet
                    if ("no results" in result_text.lower() or
                            "not found" in result_text.lower() or
                            len(result_text) < 30) and search_turns <= MAX_SEARCH_ROUNDS:
                        print(f"[web] recovery: empty search — broadening query")
                        # Inject recovery note into result
                        result_text += (
                            "\n[RECOVERY HINT: Search returned no results. "
                            "Try broader keywords — remove brand/color/size qualifiers and retry.]"
                        )

                # Deterministic price accumulation
                if any(k in tool_name for k in ("add", "cart", "buy")):
                    price = _extract_price_from_result(result_text)
                    if price > 0:
                        budget_spent += price
                        print(f"[web] budget tracker: +${price:.2f} → spent=${budget_spent:.2f}")
                        if budget is not None and budget_spent > budget:
                            result_text += (
                                f"\n[BUDGET ALERT: Running total ${budget_spent:.2f} "
                                f"EXCEEDS budget ${budget:.2f}. Remove the last item.]"
                            )

                # Deterministic constraint check on product details
                if constraints and any(k in tool_name for k in ("view", "click", "details", "get")):
                    if _check_constraint_violated(constraints, result_text):
                        constraint_violated = True
                        result_text += (
                            f"\n[CONSTRAINT VIOLATION: This product contains a forbidden attribute "
                            f"({', '.join(constraints)}). DO NOT add this to cart. Find an alternative.]"
                        )
                        print(f"[web] constraint violated detected")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        # *** BUG FIX: never append empty tool_results ***
        if tool_results:
            conversation.append({"role": "user", "content": tool_results})

        # If checkout just happened, give Claude one more turn to summarize
        if checkout_done and response.stop_reason == "tool_use":
            continue

    if not checkout_done:
        print(f"[web] WARNING: checkout not called after {turn} turns")

    return last_answer, conversation, tool_count, checkout_done, budget_spent, constraint_violated


# ── REFLECT ───────────────────────────────────────────────────────────────────
async def _reflect(
    answer: str,
    prime_ctx: dict,
    conversation: list[dict],
    mcp_url: str,
    session_id: str,
    tool_count: int,
    checkout_done: bool,
    budget_spent: float,
    constraint_violated: bool,
) -> tuple[str, list[dict], int, bool, float, bool]:
    """REFLECT: L2 checkout contract + budget/constraint summary.

    L2 Checkout Contract:
      Task category is not read-only and checkout was not called after MAX_TURNS
      → inject "call checkout NOW" and execute one more iteration.
    """
    # ── L2: Checkout contract ────────────────────────────────────────────────
    if not checkout_done and prime_ctx["has_tools"]:
        print(f"[web/L2] checkout contract failed — injecting checkout directive")
        answer, conversation, extra_tools, checkout_done, extra_spend, extra_violation = await _execute(
            "",
            prime_ctx,
            prime_ctx.get("mcp_url", mcp_url),
            # session_id is not in prime_ctx — pass through
            "session",  # placeholder, overridden below
            conversation,
            force_checkout=True,
        )
        tool_count += extra_tools
        budget_spent += extra_spend
        if extra_violation:
            constraint_violated = True

    # ── Build structured summary ────────────────────────────────────────────
    summary_parts: list[str] = []
    if budget_spent > 0:
        budget = prime_ctx.get("budget")
        if budget is not None:
            status = "✓ within budget" if budget_spent <= budget else "✗ OVER BUDGET"
            summary_parts.append(f"Spent: ${budget_spent:.2f} / ${budget:.2f} {status}")
        else:
            summary_parts.append(f"Spent: ${budget_spent:.2f}")
    if prime_ctx.get("constraints"):
        cstatus = "✗ VIOLATED" if constraint_violated else "✓ all satisfied"
        summary_parts.append(f"Constraints: {cstatus}")
    if summary_parts:
        answer = answer + "\n\n[" + " | ".join(summary_parts) + "]"

    return answer.strip(), conversation, tool_count, checkout_done, budget_spent, constraint_violated


# ── Main entry point ──────────────────────────────────────────────────────────
async def run_web_task(
    task_text: str,
    task_data: Any,
    mcp_url: str,
    session_id: str,
    conversation: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """PRIME → EXECUTE → REFLECT.

    Args:
        task_text:   The shopping task description
        task_data:   Structured task data (goal, budget, constraints, mcp_uri, etc.)
        mcp_url:     URL of the green agent's MCP server
        session_id:  Session ID for MCP tool calls
        conversation: Existing conversation history (for multi-turn)

    Returns:
        (answer, updated_conversation)
    """
    task_start = time.monotonic()

    # Override MCP URL from task_data
    task_mcp_uri = _extract_mcp_uri(task_data)
    if task_mcp_uri:
        mcp_url = task_mcp_uri

    print(f"[web] PRIME — task={task_text[:80]!r}")
    prime_ctx = await _prime(task_text, task_data, mcp_url, session_id)
    print(f"[web] category={prime_ctx['category']}, budget={prime_ctx['budget']}, "
          f"constraints={prime_ctx['constraints']}, model={prime_ctx['model']}, "
          f"tools={len(prime_ctx['tools'])}")

    print(f"[web] EXECUTE")
    answer, conversation, tool_count, checkout_done, budget_spent, constraint_violated = await _execute(
        task_text,
        prime_ctx,
        prime_ctx.get("mcp_url", mcp_url),
        session_id,
        list(conversation) if conversation else [],
    )

    print(f"[web] REFLECT — answer={len(answer)} chars, checkout={checkout_done}, "
          f"spent=${budget_spent:.2f}, violated={constraint_violated}")
    # Fix: pass session_id properly through reflect
    # (L2 retry uses conversation, not direct execute call — no session needed)
    if not checkout_done and prime_ctx["has_tools"]:
        # Direct L2 checkout retry (correct session_id)
        print(f"[web/L2] checkout contract — injecting retry")
        answer2, conversation, extra_tools, checkout_done, extra_spend, extra_viol = await _execute(
            "",
            prime_ctx,
            prime_ctx.get("mcp_url", mcp_url),
            session_id,
            conversation,
            force_checkout=True,
        )
        if answer2:
            answer = answer2
        tool_count += extra_tools
        budget_spent += extra_spend
        if extra_viol:
            constraint_violated = True

    # Append budget/constraint summary
    summary_parts: list[str] = []
    if budget_spent > 0:
        budget = prime_ctx.get("budget")
        if budget is not None:
            status = "✓ within budget" if budget_spent <= budget else "✗ OVER BUDGET"
            summary_parts.append(f"Spent: ${budget_spent:.2f} / ${budget:.2f} {status}")
    if prime_ctx.get("constraints"):
        cstatus = "✗ VIOLATED" if constraint_violated else "✓ satisfied"
        summary_parts.append(f"Constraints: {cstatus}")
    if summary_parts:
        answer = (answer or "") + "\n\n[" + " | ".join(summary_parts) + "]"

    # ── Record RL case ──────────────────────────────────────────────────────
    quality = _score_quality(
        answer,
        checkout_done,
        prime_ctx.get("budget"),
        budget_spent,
        prime_ctx.get("constraints", []),
        constraint_violated,
        tool_count,
    )
    outcome = "success" if quality >= 0.5 else "failure"
    task_kws = list(set(re.findall(r'\b\w{5,}\b', task_text.lower())))[:15]

    _save_case({
        "task_summary": task_text[:100],
        "category": prime_ctx["category"],
        "outcome": outcome,
        "quality": round(quality, 3),
        "tool_count": tool_count,
        "model": prime_ctx["model"],
        "checkout_done": checkout_done,
        "budget_spent": round(budget_spent, 2),
        "constraint_violated": constraint_violated,
        "keywords": task_kws,
        "what_worked": (
            f"category={prime_ctx['category']}, model={prime_ctx['model']}, "
            f"tools={tool_count}, checkout={'done' if checkout_done else 'missed'}"
        ),
        "what_failed": "" if quality >= 0.5 else (
            "checkout not called" if not checkout_done else
            "over budget" if (prime_ctx.get("budget") and budget_spent > prime_ctx["budget"]) else
            "constraint violated" if constraint_violated else
            f"quality={quality:.2f}"
        ),
        "duration_s": round(time.monotonic() - task_start, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    print(f"[web] RL quality={quality:.3f} outcome={outcome} "
          f"checkout={checkout_done} duration={time.monotonic()-task_start:.1f}s")

    return answer or "Shopping task complete.", conversation
