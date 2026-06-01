# Setup & Test Guide

How to configure, run, and test the Chatbot-vs-ReAct-Agent stack. Two ways to
run: the **cloud** endpoint (fast, recommended) or a **local Phi-3** model on CPU.

---

## 1. Configure credentials (`.env`)

Copy the example and fill it in. `.env` is git-ignored, so secrets never get
committed.

```bash
cp .env.example .env
```

Set the provider in `.env`:

```env
# cloud | local | openai | google
DEFAULT_PROVIDER=cloud

# --- Cloud (OpenAI-compatible endpoint from the mentor) ---
LLM_ENDPOINT=https://opencode.ai/zen/go/v1
API_KEY=sk-...your-key...
MODEL=deepseek-v4-flash

# --- Local Phi-3 fallback ---
LOCAL_MODEL_PATH=/models/phi3.gguf
```

> ⚠️ Never commit a real `API_KEY`. Both `.env` and `env` are in `.gitignore`.

---

## 2. Run with Docker (recommended)

No local Python setup needed. The daemon usually needs `sudo`.

### Option A — Cloud (slim image, builds in ~30s)

```bash
# Build
sudo docker build -f Dockerfile.cloud -t lab3-agent .

# Smoke test: is the endpoint reachable?
sudo docker run --rm -v "$PWD/.env:/app/.env:ro" lab3-agent \
  python -c "from src.core.factory import build_provider; print(build_provider().generate('Say hello in one sentence.')['content'])"

# Full Chatbot-vs-Agent suite (persists logs/ and report/ to the host)
sudo docker run --rm \
  -v "$PWD/.env:/app/.env:ro" \
  -v "$PWD/logs:/app/logs" \
  -v "$PWD/report:/app/report" \
  lab3-agent
```

> The `-v .../.env:/app/.env:ro` mount is required: `.env` is excluded from the
> image by `.dockerignore`, so it is mounted at runtime instead of baked in.

### Option B — Local Phi-3 on CPU (full image, compiles `llama-cpp`, a few min)

Download `Phi-3-mini-4k-instruct-q4.gguf` first (see `README.md`), then:

```bash
# Set DEFAULT_PROVIDER=local in .env
sudo docker build -f Dockerfile -t lab3-agent-local .

sudo docker run --rm \
  -e DEFAULT_PROVIDER=local -e LOCAL_MODEL_PATH=/models/phi3.gguf \
  -v /path/to/Phi-3-mini-4k-instruct-q4.gguf:/models/phi3.gguf:ro \
  -v "$PWD/logs:/app/logs" -v "$PWD/report:/app/report" \
  lab3-agent-local
```

---

## 3. Run on the host (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or just: pip install openai python-dotenv  (cloud only)

python run_tests.py                     # full suite
python run_tests.py --mode agent        # agent only
python run_tests.py --mode chatbot      # baseline only

python -m src.agent.chatbot             # interactive chatbot REPL

# Endpoint smoke test (uses the configured provider)
python -c "from src.core.factory import build_provider; print(build_provider().generate('hello')['content'])"
```

---

## 4. What the test runner does

`run_tests.py` runs the same cases through **both** the baseline `Chatbot` and
the `ReActAgent`:

| Case        | Type       | Checks |
| :---------- | :--------- | :----- |
| `simple-1`  | simple     | factual recall (capital of France) |
| `simple-2`  | simple     | open definition (graded N/A) |
| `multi-1/2` | multi-step | price + discount + calculator math |
| `multi-3`   | multi-step | out-of-stock handling (ipad) |
| `multi-4`   | multi-step | shipping cost lookup |

Grading is a **loose substring heuristic** (commas stripped from numbers). The
expected result: the chatbot **fails** multi-step cases (it cannot look up real
prices/stock), while the agent **passes** them by calling tools — which is the
whole point of the lab.

Outputs:
- `logs/<date>.log` — structured JSON telemetry, one event per line.
- `report/last_run_summary.md` — markdown pass/fail + latency table.

---

## 5. Reading the logs (for the report)

Every line in `logs/<date>.log` is a JSON event. Key event types:

| Event | Meaning |
| :--- | :--- |
| `AGENT_START` / `AGENT_END` | task boundaries + step count |
| `AGENT_THOUGHT` | the model's reasoning + raw output for the step |
| `AGENT_ACTION` | tool name + parsed arguments |
| `AGENT_OBSERVATION` | tool result fed back into the loop |
| `AGENT_ERROR` | `PARSE_ERROR`, `HALLUCINATED_TOOL`, `BAD_ARGUMENTS`, `MAX_STEPS_EXCEEDED` |
| `LLM_METRIC` | per-call tokens, latency_ms, cost_estimate |
| `CHATBOT_START` / `CHATBOT_END` | baseline calls |

Quick aggregation examples:

```bash
# All failure events
grep AGENT_ERROR logs/*.log

# Total tokens + average latency across the run
python -c "
import json, glob
m=[json.loads(l)['data'] for f in glob.glob('logs/*.log') for l in open(f)
   if l.strip().startswith('{') and json.loads(l)['event']=='LLM_METRIC']
print('calls:', len(m))
print('total tokens:', sum(x['total_tokens'] for x in m))
print('avg latency ms:', sum(x['latency_ms'] for x in m)//len(m))
print('est cost:', round(sum(x['cost_estimate'] for x in m), 4))
"
```

---

## 6. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `pull access denied for lab3-agent` | Image not built yet — run the `docker build` step first. |
| `BrokenPipeError` / `Connection broken` during build | Flaky PyPI network — just re-run `docker build` (layers cache). |
| `Missing required env values` | `.env` not mounted (`-v "$PWD/.env:/app/.env:ro"`) or keys not set. |
| Agent `PARSE_ERROR` steps | The model emitted prose instead of `Action:`/`Final Answer:`. The loop feeds the error back and the model usually recovers; tighten the system prompt for v2. |
| Local model `FileNotFoundError` | `LOCAL_MODEL_PATH` / mount path does not point at the `.gguf` file. |
