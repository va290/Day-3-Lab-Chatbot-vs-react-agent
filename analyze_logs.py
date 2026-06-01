"""
Log analyzer: turn the structured JSON telemetry in logs/ into the industry
metrics required by EVALUATION.md and the group report:

  - Token efficiency  (prompt / completion / total per call and per task)
  - Latency           (P50 / P90 / P99 / avg / max)
  - Loop count        (Thought->Action cycles per agent task)
  - Failure analysis  (counts per error_code, with examples)
  - Chatbot vs Agent  (and Agent v1 vs v2) comparison

Usage:
    python analyze_logs.py                 # reads all logs/*.log
    python analyze_logs.py logs/2026-06-01.log
    python analyze_logs.py --out report/analysis.md
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Dict, Any


def load_events(paths: List[str]) -> List[Dict[str, Any]]:
    events = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation); p in [0,100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def build_tasks(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group the flat event stream into tasks (one per CHATBOT/AGENT START..END).
    Runner is single-threaded, so events are sequential per task."""
    tasks: List[Dict[str, Any]] = []
    cur = None
    for ev in events:
        et, data = ev.get("event"), ev.get("data", {})
        if et in ("AGENT_START", "CHATBOT_START"):
            cur = {
                "kind": "agent" if et == "AGENT_START" else "chatbot",
                "version": data.get("version", "-"),
                "input": data.get("input", ""),
                "metrics": [], "errors": [], "steps": 0,
            }
            tasks.append(cur)
        elif et == "LLM_METRIC" and cur is not None:
            cur["metrics"].append(data)
        elif et == "AGENT_ERROR" and cur is not None:
            cur["errors"].append(data)
        elif et == "AGENT_END" and cur is not None:
            cur["steps"] = data.get("steps", len(cur["metrics"]))
            cur = None
        elif et == "CHATBOT_END":
            cur = None
    return tasks


def system_label(task: Dict[str, Any]) -> str:
    return "chatbot" if task["kind"] == "chatbot" else f"agent-{task['version']}"


def md_metric_block(calls: List[Dict[str, Any]]) -> List[str]:
    if not calls:
        return ["_no LLM calls_"]
    lat = [c.get("latency_ms", 0) for c in calls]
    tot = [c.get("total_tokens", 0) for c in calls]
    prm = [c.get("prompt_tokens", 0) for c in calls]
    cmp = [c.get("completion_tokens", 0) for c in calls]
    cost = sum(c.get("cost_estimate", 0) for c in calls)
    n = len(calls)
    return [
        f"- LLM calls: **{n}**",
        f"- Tokens — total: **{sum(tot)}**, avg/call: {sum(tot)//n} "
        f"(prompt {sum(prm)//n} / completion {sum(cmp)//n})",
        f"- Latency ms — avg: {sum(lat)//n}, P50: {percentile(lat,50):.0f}, "
        f"P90: {percentile(lat,90):.0f}, P99: {percentile(lat,99):.0f}, max: {max(lat)}",
        f"- Estimated cost: ${cost:.4f}",
    ]


def analyze(events: List[Dict[str, Any]]) -> str:
    tasks = build_tasks(events)
    all_calls = [m for t in tasks for m in t["metrics"]]
    systems = sorted({system_label(t) for t in tasks})

    n_retry = sum(1 for e in events if e.get("event") == "LLM_RETRY")
    n_llm_err = sum(1 for e in events if e.get("event") == "LLM_ERROR")

    out: List[str] = ["# Telemetry Analysis\n"]
    out.append(f"Parsed **{len(events)}** events → **{len(tasks)}** tasks, "
               f"**{len(all_calls)}** LLM calls.\n")

    # 1) Global metrics ------------------------------------------------------
    out.append("## 1. Overall LLM metrics")
    out += md_metric_block(all_calls)
    out.append(f"- Network resilience: **{n_retry}** LLM_RETRY events, "
               f"**{n_llm_err}** unrecoverable LLM_ERROR(s)")

    # 2) Per-system comparison ----------------------------------------------
    out.append("\n## 2. Comparison by system\n")
    out.append("| System | Tasks | LLM calls | Avg steps/task | Avg tokens/task | Avg latency/call (ms) | Errors |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in systems:
        st = [t for t in tasks if system_label(t) == s]
        calls = [m for t in st for m in t["metrics"]]
        n_tasks = len(st)
        toks = sum(m.get("total_tokens", 0) for m in calls)
        lat = [m.get("latency_ms", 0) for m in calls]
        n_err = sum(len(t["errors"]) for t in st)
        avg_steps = sum(t["steps"] for t in st) / n_tasks if n_tasks else 0
        out.append(
            f"| {s} | {n_tasks} | {len(calls)} | {avg_steps:.1f} | "
            f"{toks // n_tasks if n_tasks else 0} | "
            f"{sum(lat)//len(lat) if lat else 0} | {n_err} |"
        )

    # 3) Loop-count detail (agents) -----------------------------------------
    out.append("\n## 3. Loop count per agent task (Thought→Action cycles)\n")
    out.append("| System | Input (truncated) | Steps | Errors |")
    out.append("| :--- | :--- | ---: | ---: |")
    for t in tasks:
        if t["kind"] != "agent":
            continue
        out.append(f"| {system_label(t)} | {t['input'][:50]} | {t['steps']} | {len(t['errors'])} |")

    # 4) Failure analysis ----------------------------------------------------
    out.append("\n## 4. Failure analysis (by error_code)\n")
    err_counts: Dict[str, int] = {}
    examples: Dict[str, str] = {}
    for t in tasks:
        for e in t["errors"]:
            code = e.get("error_code", "UNKNOWN")
            err_counts[code] = err_counts.get(code, 0) + 1
            snippet = str(e.get("raw") or e.get("action") or e.get("tool") or "").strip()
            # Prefer the first NON-EMPTY snippet as the illustrative example.
            if snippet and not examples.get(code):
                examples[code] = snippet[:120].replace("\n", " ")
    if err_counts:
        out.append("| error_code | count | example |")
        out.append("| :--- | ---: | :--- |")
        for code, c in sorted(err_counts.items(), key=lambda x: -x[1]):
            out.append(f"| `{code}` | {c} | {examples.get(code,'')} |")
    else:
        out.append("_No errors recorded._")

    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="log files (default: logs/*.log)")
    parser.add_argument("--out", default="report/analysis.md", help="markdown output path")
    args = parser.parse_args()

    paths = args.paths or sorted(glob.glob("logs/*.log"))
    if not paths:
        print("No log files found. Run run_tests.py first to generate logs/.")
        return 1

    events = load_events(paths)
    report = analyze(events)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)

    print(report)
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
