# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: AI20k - Lab 3 Team
- **Team Members**: Đỗ Việt Anh (2A202601008), Long
- **Deployment Date**: 2026-06-01

> Provider used for all results below: **DeepSeek-v4-flash** via the cloud
> OpenAI-compatible endpoint (`LLM_ENDPOINT`). Numbers come from a single
> `run_tests.py` ablation run (chatbot + agent-v1 + agent-v2), analysed with
> `analyze_logs.py`.

---

## 1. Executive Summary

*Brief overview of the agent's goal and success rate compared to the baseline chatbot.*

- **Success Rate**: Agent **5/5 (100%)** on graded cases; baseline Chatbot **2/5 (40%)**.
- **Key Outcome**: The ReAct agent solved every tool-dependent multi-step query
  (price + discount + total, stock check, shipping) that the chatbot could not.
  The chatbot, having no tool access, either asks the user for the data or
  hallucinates it. Upgrading the prompt from **v1 → v2** kept accuracy at 100%
  while cutting **average latency per task by ~30%** (12318ms → 8645ms) and
  **average steps from 3.2 → 2.8**.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

The agent (`src/agent/agent.py`) runs a `Thought → Action → Observation` loop over
a growing **scratchpad**:

1. The LLM is asked for the next `Thought` + `Action` (system prompt + scratchpad).
2. The `Action` is parsed (`tool_name(args)`) and the tool is executed.
3. The tool's result is appended as an `Observation` and fed back into the loop.
4. The loop ends when the model emits `Final Answer:` or hits `max_steps`.

```
Question ─▶ Thought ─▶ Action ─▶ [tool] ─▶ Observation ─┐
              ▲                                          │
              └──────────────  (repeat ≤ max_steps)  ◀───┘
                          │
                          ▼
                     Final Answer
```

Robustness features: tolerant Action parsing (`key=value`, JSON, bare positional),
stripping of hallucinated `Observation:` text, `max_steps` guardrail, and (in v2)
a repeated-action guard + stronger parse-error nudge. Every step is logged as a
structured JSON event for analysis.

### 2.2 Tool Definitions (Inventory)
| Tool Name | Input Format | Use Case |
| :--- | :--- | :--- |
| `get_price` | `item_name: str` | Unit price (USD) of a catalog item. |
| `check_stock` | `item_name: str` | Units in stock (ipad is intentionally 0). |
| `get_discount` | `coupon_code: str` | Discount % for a coupon (WINNER, VIP20…). |
| `calc_shipping` | `weight: float, destination: str` | Shipping cost by city. |
| `calculator` | `expression: str` | Safe arithmetic for all totals/discounts. |

### 2.3 LLM Providers Used
- **Primary**: DeepSeek-v4-flash (cloud, OpenAI-compatible endpoint).
- **Fallback**: Local Phi-3-mini-4k (GGUF, CPU via llama-cpp) — provider switch
  is a one-line `.env` change (`DEFAULT_PROVIDER`), enabled by the `LLMProvider`
  ABC + `build_provider()` factory.

---

## 3. Telemetry & Performance Dashboard

*Industry metrics across the full ablation run (39 LLM calls over 18 tasks).*

- **Average Latency (P50)**: 2970 ms
- **P90 / P99 / Max Latency**: 6874 / 11360 / 11360 ms
- **Average Latency (mean)**: 3977 ms
- **Tokens**: 35,052 total; avg 898/call (prompt 589 / completion 308)
- **Estimated Cost of Test Suite**: $0.3505 *(mock pricing: $0.01 / 1k tokens)*

| System | Tasks | LLM calls | Avg steps/task | Avg tokens/task | Avg latency/call (ms) | Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| chatbot  | 6 | 6  | 0.0 | 725  | 6891 | 0 |
| agent-v1 | 6 | 17 | 2.8 | 2415 | 3795 | 3 |
| agent-v2 | 6 | 16 | 2.7 | 2701 | 3078 | 4 |

> Note: chatbot makes 1 long call/task (no loop), so its tokens/task are low but
> its per-call latency is highest. Agents make several short calls per task.

---

## 4. Root Cause Analysis (RCA) - Failure Traces

*Failures captured in `logs/` (event `AGENT_ERROR`), totals: `PARSE_ERROR` ×5, `REPEATED_ACTION` ×2.*

### Case Study A: Reasoning in prose instead of acting (`PARSE_ERROR`)
- **Input**: "I want to buy 2 iphones using coupon WINNER. What is the total price?"
- **Observation**: After fetching the price, the model replied:
  *"The total price for 2 iPhones after applying the 10% WINNER discount is $1,798.20… 999 × 2 × 0.9 = 1798.2"* — **no `Action:` or `Final Answer:` line**.
- **Root Cause**: The v1 prompt did not show the model *how* a turn should look,
  so it computed the math in its head and answered as plain prose. The parser
  found no marker → `PARSE_ERROR`. The loop fed the error back and the model
  recovered, but at the cost of extra steps.
- **Fix (v2)**: Added a worked **few-shot example** and a CRITICAL rule —
  *never do arithmetic yourself; every reply has exactly one `Action:` or
  `Final Answer:` line*.

### Case Study B: Repeating an action (`REPEATED_ACTION`)
- **Trace**: `calc_shipping(weight=4.0, destination="hanoi")` requested twice in a row.
- **Root Cause**: After a valid observation, the model occasionally re-issues the
  same tool call instead of progressing.
- **Fix (v2)**: A repeated-action guard returns *"You already called … use the
  observation or give the Final Answer"* instead of re-executing, preventing
  wasted loops.

---

## 5. Ablation Studies & Experiments

### Experiment 1: Prompt v1 vs Prompt v2
- **Diff**: v2 = v1 + few-shot worked example + stricter format rules
  ("never compute in head", "exactly one marker per reply") + repeated-action
  guard + a stronger parse-error nudge.
- **Result**:

| Metric | agent-v1 | agent-v2 | Change |
| :--- | ---: | ---: | :--- |
| Pass rate | 5/5 (100%) | 5/5 (100%) | = |
| Avg steps / task | 3.2 | 2.8 | **−13%** |
| Avg latency / task (ms) | 12318 | 8645 | **−30%** |
| Avg tokens / task | 2415 | 2701 | +12% (longer prompt) |

- **Interpretation**: v2 converges faster (fewer steps, lower latency) at the
  cost of a slightly larger prompt (the few-shot example). Net win for a
  production setting where latency and reliability matter most.

### Experiment 2: Chatbot vs Agent
| Case | Type | Chatbot | Agent (v1/v2) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| simple-1 (capital of France) | simple | Correct | Correct | Draw |
| simple-2 (define AI agent) | simple | OK (N/A) | OK (N/A) | Draw |
| multi-1 (2 iphones + WINNER) | multi-step | Asks for data | Correct ($1798.20) | **Agent** |
| multi-2 (macbook + airpods + VIP20) | multi-step | Asks for data | Correct ($2076.80) | **Agent** |
| multi-3 (ipad stock) | multi-step | "no inventory access" | Correct (0, can't buy) | **Agent** |
| multi-4 (ship 4kg to Hanoi) | multi-step | Guess (no rate tool) | Correct ($8.00) | **Agent** |

---

## 6. Production Readiness Review

- **Security**: `calculator` evaluates expressions behind a strict character
  whitelist (`0-9 . + - * / ( )`) with `__builtins__` disabled — no arbitrary
  code execution. Tool inputs are normalized before lookup.
- **Guardrails**: `max_steps` caps billing/infinite loops; v2 adds a
  repeated-action guard; unknown tools / bad arguments are caught and fed back
  as observations rather than crashing.
- **Network resilience**: the cloud LLM call is wrapped in a retry loop with
  exponential backoff (default 3 retries, 1s/2s/4s) targeting transient failures
  (`APIConnectionError`, `APITimeoutError`, `RateLimitError`, 5xx). Each retry is
  logged as an `LLM_RETRY` event (and a final `LLM_ERROR` if it gives up), so the
  retry rate is observable in `analyze_logs.py`. The SDK's silent retries are
  disabled so our logged loop is the single source of truth. Tool calls are local
  functions, so they need no network handling.
- **Observability**: every Thought/Action/Observation/error + per-call
  tokens/latency/cost is logged as JSON; `analyze_logs.py` turns it into metrics.
- **Scaling**: move to LangGraph for branching/parallel tool calls; add vector
  retrieval for tool selection when the tool count grows; real pricing table to
  replace the mock cost estimate.
