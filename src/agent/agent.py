import re
import json
import inspect
from typing import List, Dict, Any, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    A ReAct-style agent that follows the Thought -> Action -> Observation loop.

    The agent keeps a running "scratchpad" of the dialogue. On each step it asks
    the LLM for the next Thought + Action, executes the requested tool, appends
    the resulting Observation to the scratchpad, and repeats until the model
    emits a "Final Answer:" or `max_steps` is exceeded.

    Every step is logged via the telemetry logger so failure traces (parser
    errors, hallucinated tools, infinite loops) can be analysed afterwards.
    """

    # Regexes used to pull structured fields out of free-form LLM text.
    _ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z_]\w*)\s*\((.*)\)", re.DOTALL)
    _FINAL_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.DOTALL)
    _THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*)", re.DOTALL)

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]],
                 max_steps: int = 5, version: str = "v1"):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # "v1" = minimal prompt (baseline); "v2" = improved prompt + loop
        # guardrails addressing the failures observed in v1 (see get_system_prompt).
        self.version = version
        self.history: List[Dict[str, Any]] = []
        # name -> tool dict, for O(1) dispatch
        self._tool_map = {t["name"]: t for t in tools}

    # ------------------------------------------------------------------ prompt
    def get_system_prompt(self) -> str:
        """Instruct the model to follow the ReAct format and list the tools.
        v2 adds a worked few-shot example and stricter rules, derived from the
        v1 failure traces (model computing math in its head, emitting prose
        without an Action:/Final Answer: marker)."""
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self.tools
        )
        tool_names = ", ".join(t["name"] for t in self.tools)

        base = f"""You are a careful reasoning assistant that solves tasks step by step using tools.

You have access to ONLY these tools:
{tool_descriptions}

To use a tool, output EXACTLY this format (one Action per step):
Thought: <your reasoning about what to do next>
Action: <tool_name>(<arguments>)

Rules for Action:
- <tool_name> must be one of: {tool_names}. Never invent a tool.
- Arguments use key=value pairs, e.g. Action: get_price(item_name="iphone")
- Output ONLY the Thought and Action. STOP after the Action line.
- Do NOT write the Observation yourself; the system provides it.

When you have enough information to answer, output:
Thought: <why you are done>
Final Answer: <the complete answer for the user>

Always use the calculator tool for arithmetic instead of computing in your head."""

        if self.version != "v2":
            return base

        # v2: stricter rules + a worked example so weak models lock onto format.
        return base + """

CRITICAL RULES (do not break these):
- NEVER do arithmetic yourself. Even "999 * 2" MUST go through the calculator tool.
- Every reply MUST start with "Thought:" and contain EXACTLY ONE of: an "Action:"
  line OR a "Final Answer:" line. Never both, never neither, never plain prose.
- Only give "Final Answer:" AFTER the calculator has returned the number you report.

Worked example:
Question: What is the total for 3 ipads with coupon VIP20?
Thought: I need the unit price of an ipad first.
Action: get_price(item_name="ipad")
Observation: The unit price of ipad is $599.00.
Thought: Now I need the VIP20 discount.
Action: get_discount(coupon_code="VIP20")
Observation: Coupon VIP20 grants a 20% discount.
Thought: Compute 599 * 3 with 20% off using the calculator.
Action: calculator(expression="599 * 3 * 0.8")
Observation: 599 * 3 * 0.8 = 1437.6
Thought: I have the total.
Final Answer: The total for 3 ipads with coupon VIP20 is $1437.60."""

    # --------------------------------------------------------------------- run
    def run(self, user_input: str) -> str:
        """Execute the ReAct loop and return the final answer string."""
        logger.log_event(
            "AGENT_START",
            {"input": user_input, "model": self.llm.model_name, "version": self.version},
        )
        self.history = []

        system_prompt = self.get_system_prompt()
        # The scratchpad is everything the model has "said" plus observations.
        scratchpad = f"Question: {user_input}\n"
        steps = 0
        final_answer: Optional[str] = None
        last_action_sig: Optional[str] = None  # v2: detect repeated identical actions

        while steps < self.max_steps:
            steps += 1

            # 1) Ask the LLM for the next Thought + Action.
            result = self.llm.generate(scratchpad, system_prompt=system_prompt)
            content = result.get("content", "") or ""

            # Telemetry for this LLM call.
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

            # Defensive: some models hallucinate an Observation. Cut it off so we
            # only keep what the model is actually allowed to produce.
            content = content.split("Observation:")[0].strip()

            thought = self._extract_thought(content)
            logger.log_event("AGENT_THOUGHT", {"step": steps, "thought": thought, "raw": content})

            # 2) Did the model decide it is finished?
            final = self._extract_final_answer(content)
            if final is not None:
                final_answer = final
                self.history.append({"step": steps, "thought": thought, "final_answer": final})
                break

            # 3) Otherwise parse and execute the requested Action.
            parsed = self._parse_action(content)
            if parsed is None:
                # The model produced neither a parseable Action nor a Final Answer.
                logger.log_event(
                    "AGENT_ERROR",
                    {"step": steps, "error_code": "PARSE_ERROR", "raw": content},
                )
                if self.version == "v2":
                    # Stronger, more directive nudge — v1 traces showed the model
                    # often answered in prose without the marker.
                    observation = (
                        "FORMAT ERROR: your reply had no 'Action:' or 'Final Answer:' line. "
                        "If you already know the answer, reply with exactly one line: "
                        "'Final Answer: <answer>'. Otherwise reply with 'Action: tool_name(args)'."
                    )
                else:
                    observation = (
                        "Error: could not parse an Action. Respond with either "
                        "'Action: tool_name(args)' or 'Final Answer: ...'."
                    )
            else:
                tool_name, raw_args = parsed
                # v2: guard against the model repeating the exact same action.
                action_sig = f"{tool_name}({raw_args})"
                if self.version == "v2" and action_sig == last_action_sig:
                    logger.log_event(
                        "AGENT_ERROR",
                        {"step": steps, "error_code": "REPEATED_ACTION", "action": action_sig},
                    )
                    observation = (
                        f"You already called {action_sig} and got the result above. "
                        "Do not repeat it — use the observation to proceed or give the Final Answer."
                    )
                else:
                    observation = self._execute_tool(tool_name, raw_args, step=steps)
                last_action_sig = action_sig

            logger.log_event("AGENT_OBSERVATION", {"step": steps, "observation": observation})
            self.history.append(
                {"step": steps, "thought": thought, "action": content, "observation": observation}
            )

            # 4) Feed the observation back into the scratchpad for the next loop.
            scratchpad += f"{content}\nObservation: {observation}\n"

        if final_answer is None:
            logger.log_event(
                "AGENT_ERROR",
                {"error_code": "MAX_STEPS_EXCEEDED", "steps": steps},
            )
            final_answer = (
                f"Stopped after {steps} steps without reaching a Final Answer "
                f"(max_steps={self.max_steps})."
            )

        logger.log_event("AGENT_END", {"steps": steps, "final_answer": final_answer})
        return final_answer

    # -------------------------------------------------------------- parsing
    def _extract_thought(self, content: str) -> str:
        m = self._THOUGHT_RE.search(content)
        if not m:
            return ""
        # Stop the thought at the next structured marker if present.
        thought = m.group(1)
        for marker in ("Action:", "Final Answer:"):
            thought = thought.split(marker)[0]
        return thought.strip()

    def _extract_final_answer(self, content: str) -> Optional[str]:
        m = self._FINAL_RE.search(content)
        return m.group(1).strip() if m else None

    def _parse_action(self, content: str) -> Optional[Tuple[str, str]]:
        """Return (tool_name, raw_args_string) or None if no Action is present."""
        m = self._ACTION_RE.search(content)
        if not m:
            return None
        tool_name = m.group(1).strip()
        raw_args = m.group(2).strip()
        return tool_name, raw_args

    def _parse_args(self, raw_args: str) -> Dict[str, Any]:
        """
        Parse a tool's argument string into a kwargs dict. Supports several
        formats that weak models tend to emit:
          - key=value pairs:   item_name="iphone", weight=2
          - JSON object:       {"item_name": "iphone"}
          - a bare value:      "iphone"  (mapped to the first param by caller)
        """
        raw_args = raw_args.strip()
        if not raw_args:
            return {}

        # JSON object form.
        if raw_args.startswith("{"):
            try:
                obj = json.loads(raw_args)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # key=value form (split on commas that separate top-level pairs).
        if "=" in raw_args:
            kwargs: Dict[str, Any] = {}
            for part in self._split_top_level(raw_args):
                if "=" not in part:
                    continue
                key, _, value = part.partition("=")
                kwargs[key.strip()] = self._coerce(value.strip())
            if kwargs:
                return kwargs

        # Bare positional value -> let the caller map it to the first parameter.
        return {"__positional__": self._coerce(raw_args)}

    @staticmethod
    def _split_top_level(s: str) -> List[str]:
        """Split on commas that are not inside quotes or brackets."""
        parts, depth, buf, quote = [], 0, [], None
        for ch in s:
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
                buf.append(ch)
            elif ch in "([{":
                depth += 1
                buf.append(ch)
            elif ch in ")]}":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf))
        return parts

    @staticmethod
    def _coerce(value: str) -> Any:
        """Strip quotes and best-effort cast numbers."""
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            return value[1:-1]
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    # ------------------------------------------------------------- execution
    def _execute_tool(self, tool_name: str, raw_args: str, step: int = 0) -> str:
        """Dispatch a parsed Action to the matching tool callable."""
        tool = self._tool_map.get(tool_name)
        if tool is None:
            logger.log_event(
                "AGENT_ERROR",
                {"step": step, "error_code": "HALLUCINATED_TOOL", "tool": tool_name},
            )
            available = ", ".join(self._tool_map)
            return f"Error: tool '{tool_name}' does not exist. Available tools: {available}."

        kwargs = self._parse_args(raw_args)
        func = tool["func"]

        # Map a bare positional value onto the function's first parameter.
        if "__positional__" in kwargs:
            params = list(inspect.signature(func).parameters)
            if not params:
                return f"Error: tool '{tool_name}' takes no arguments."
            kwargs = {params[0]: kwargs["__positional__"]}

        try:
            logger.log_event("AGENT_ACTION", {"step": step, "tool": tool_name, "args": kwargs})
            return str(func(**kwargs))
        except TypeError as e:
            # Wrong/missing argument names -> classic hallucinated-argument failure.
            logger.log_event(
                "AGENT_ERROR",
                {"step": step, "error_code": "BAD_ARGUMENTS", "tool": tool_name, "args": kwargs, "detail": str(e)},
            )
            return f"Error calling {tool_name}: {e}. Check the argument names in the tool description."
        except Exception as e:  # noqa: BLE001 - feed any runtime error back to the agent
            logger.log_event(
                "AGENT_ERROR",
                {"step": step, "error_code": "TOOL_RUNTIME_ERROR", "tool": tool_name, "detail": str(e)},
            )
            return f"Error: {tool_name} raised {type(e).__name__}: {e}."
