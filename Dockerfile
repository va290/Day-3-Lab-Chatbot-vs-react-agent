# CPU-only image for running the Lab 3 agent with a local Phi-3 GGUF model.
FROM python:3.11-slim

# Build tools needed to compile llama-cpp-python's native backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the full Chatbot-vs-Agent test suite.
CMD ["python", "run_tests.py"]
