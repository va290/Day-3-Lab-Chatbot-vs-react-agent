# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Đỗ Việt Anh
- **Student ID**: 2A202601008
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase.*

- **Modules Implemented**:
  - `src/agent/agent.py` — the full ReAct loop (Thought→Action→Observation),
    tolerant Action parsing (`key=value` / JSON / bare positional), `max_steps`
    guardrail, dynamic tool dispatch, and the **v1/v2 prompt versions**.
  - `src/tools/ecommerce_tools.py` — the 5-tool e-commerce toolset
    (`get_price`, `check_stock`, `get_discount`, `calc_shipping`, `calculator`)
    with precise descriptions and a `TOOLS` registry.
  - `src/agent/chatbot.py` — the baseline single-shot chatbot.
  - `src/core/cloud_provider.py` + `src/core/factory.py` — adapter integrating
    the mentor's OpenAI-compatible cloud endpoint into our `LLMProvider` ABC so
    telemetry (tokens/latency/cost) works across providers.
  - `run_tests.py` — the Chatbot-vs-Agent (v1-vs-v2) ablation runner.
  - `analyze_logs.py` — parses JSON logs into the report metrics.
- **Code Highlights**:
  - Action parsing via regex `Action:\s*([A-Za-z_]\w*)\s*\((.*)\)` plus a
    `_parse_args` that handles the three argument formats weak models emit.
  - v2 `get_system_prompt()` adds a worked few-shot example + strict rules;
    the loop adds a repeated-action guard.
- **Documentation**: see `SETUP_AND_TEST.md` for setup, ablation, and log analysis.

---

## II. Debugging Case Study (10 Points)

*A failure I diagnosed using the logging system.*

- **Problem Description**: On *"buy 2 iphones using coupon WINNER"*, the agent
  (v1) logged a `PARSE_ERROR` mid-run — it stalled instead of finishing cleanly.
- **Log Source** (`logs/<date>.log`, event `AGENT_ERROR`):
  ```json
  {"event": "AGENT_ERROR", "data": {"step": 3, "error_code": "PARSE_ERROR",
   "raw": "The total price for 2 iPhones after applying the 10% WINNER discount
   is $1,798.20 ... 999 × 2 × 0.9 = 1798.2"}}
  ```
- **Diagnosis**: This was a **prompt** problem, not a model or tool problem. The
  v1 system prompt told the model the format but never *showed* it, so the model
  did the arithmetic in its head and answered as plain prose with **no
  `Action:`/`Final Answer:` marker**. The parser then failed.
- **Solution (v1 → v2)**: I added a worked **few-shot example** and a CRITICAL
  rule — *never compute arithmetic yourself; every reply contains exactly one
  `Action:` or `Final Answer:` line* — plus a repeated-action guard. Measured
  effect: same 100% accuracy, but average steps dropped 3.2 → 2.8 and average
  latency/task dropped ~30% (12318ms → 8645ms).

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: The `Thought` block forces the model to decide *what to do
   next* before acting, so it fetches real data (price, discount) instead of
   guessing. The chatbot, answering in one shot, has no such checkpoint and just
   asks the user for the numbers or hallucinates them — it failed every
   tool-dependent multi-step case (multi-1/2/3).
2. **Reliability**: The agent is **slower and pricier** on *simple* questions —
   for "capital of France" it spends a full extra LLM call vs the chatbot for the
   same answer. For trivial Q&A the chatbot is the better tool.
3. **Observation**: Feeding the tool result back as an `Observation` is what makes
   multi-step work — e.g. the agent reads `get_discount → 10%` and only then
   calls `calculator("999*2*0.9")`. Without that feedback loop it would be just
   another chatbot.

---

## IV. Future Improvements (5 Points)

- **Scalability**: parallelize independent tool calls (e.g. fetch price + stock
  concurrently) using an async queue; adopt LangGraph for explicit branching.
- **Safety**: add a lightweight "supervisor" check that validates a `Final
  Answer` against the observations before returning it to the user.
- **Performance**: for a many-tool system, retrieve the top-k relevant tools via
  a vector store instead of stuffing every description into the prompt (which
  inflates prompt tokens — we already saw v2's few-shot add ~12% tokens/task).

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_[YOUR_NAME].md` and placing it in this folder.
