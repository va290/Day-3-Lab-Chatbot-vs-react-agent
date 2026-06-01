"""
Test runner: evaluate the baseline Chatbot vs the ReAct Agent on the same set
of cases, using whichever provider is configured in .env (local | openai | google).

Usage:
    python run_tests.py                 # run both, all cases
    python run_tests.py --mode agent    # only the agent
    python run_tests.py --mode chatbot  # only the chatbot

All per-step telemetry is written to logs/<date>.log. A markdown summary table
(handy for the group report) is written to report/last_run_summary.md.
"""

import os
import sys
import time
import argparse

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars can be exported directly
    def load_dotenv(*a, **k):
        return False

from src.core.factory import build_provider
from src.tools import TOOLS
from src.agent.agent import ReActAgent
from src.agent.chatbot import Chatbot


# Each case: id, question, expected substring (loose auto-grade), and whether it
# requires multi-step tool use (where we expect the chatbot to struggle).
TEST_CASES = [
    {"id": "simple-1", "q": "What is the capital of France?",
     "expect": "paris", "multi_step": False},
    {"id": "simple-2", "q": "In one short sentence, what is an AI agent?",
     "expect": None, "multi_step": False},
    {"id": "multi-1", "q": "I want to buy 2 iphones using coupon WINNER. What is the total price?",
     "expect": "1798", "multi_step": True},   # 999 * 2 * 0.90
    {"id": "multi-2", "q": "I want 1 macbook and 3 airpods. Apply coupon VIP20. What is the total?",
     "expect": "2076", "multi_step": True},   # (1999 + 3*199) * 0.80 = 2076.8
    {"id": "multi-3", "q": "How many ipads are available, and can I buy 2?",
     "expect": "0", "multi_step": True},  # ipad stock = 0 -> answer must mention 0
    {"id": "multi-4", "q": "What does it cost to ship a 4kg order to Hanoi?",
     "expect": "8", "multi_step": True},       # 4 * 2.0 = 8.0
]


def grade(answer: str, expected) -> str:
    """Loose substring auto-grade. Numbers are normalized (commas stripped) so
    '$1,798.20' still matches an expected '1798'. This is a heuristic — see the
    report's discussion of why semantic grading of free-form answers is hard."""
    if expected is None:
        return "N/A"
    norm = (answer or "").lower().replace(",", "")
    return "PASS" if expected.lower() in norm else "FAIL"


def run_one(name: str, runner, case: dict) -> dict:
    start = time.time()
    try:
        answer = runner.run(case["q"])
        err = None
    except Exception as e:  # noqa: BLE001
        answer, err = f"<EXCEPTION: {e}>", str(e)
    dur = int((time.time() - start) * 1000)
    verdict = "ERROR" if err else grade(answer, case["expect"])
    steps = len(getattr(runner, "history", [])) if name == "agent" else 1
    print(f"  [{name:7}] {case['id']:9} {verdict:4} ({dur:>6}ms, {steps} step) -> "
          f"{answer[:90].replace(chr(10), ' ')}")
    return {"id": case["id"], "system": name, "verdict": verdict,
            "ms": dur, "steps": steps, "answer": answer}


def write_summary(rows):
    os.makedirs("report", exist_ok=True)
    path = os.path.join("report", "last_run_summary.md")
    by_case = {}
    for r in rows:
        by_case.setdefault(r["id"], {})[r["system"]] = r
    lines = ["# Last Test Run Summary\n",
             "| Case | Chatbot | Agent | Chatbot ms | Agent ms | Agent steps |",
             "| :--- | :--- | :--- | ---: | ---: | ---: |"]
    for cid, sysmap in by_case.items():
        cb = sysmap.get("chatbot", {})
        ag = sysmap.get("agent", {})
        lines.append(f"| {cid} | {cb.get('verdict','-')} | {ag.get('verdict','-')} "
                     f"| {cb.get('ms','-')} | {ag.get('ms','-')} | {ag.get('steps','-')} |")
    # Aggregate pass rates (ignoring N/A).
    for sysname in ("chatbot", "agent"):
        graded = [r for r in rows if r["system"] == sysname and r["verdict"] in ("PASS", "FAIL", "ERROR")]
        passed = sum(1 for r in graded if r["verdict"] == "PASS")
        if graded:
            lines.append(f"\n- **{sysname} pass rate**: {passed}/{len(graded)} "
                         f"({100*passed//len(graded)}%)")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "agent", "chatbot"], default="both")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    load_dotenv()
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai")
    print(f"Building provider: {provider_name} ...")
    llm = build_provider()
    print(f"Model: {llm.model_name}\n")

    chatbot = Chatbot(llm)
    agent = ReActAgent(llm, TOOLS, max_steps=args.max_steps)

    rows = []
    for case in TEST_CASES:
        tag = "MULTI-STEP" if case["multi_step"] else "simple"
        print(f"\n=== {case['id']} ({tag}): {case['q']}")
        if args.mode in ("both", "chatbot"):
            rows.append(run_one("chatbot", chatbot, case))
        if args.mode in ("both", "agent"):
            rows.append(run_one("agent", agent, case))

    write_summary(rows)


if __name__ == "__main__":
    sys.exit(main())
