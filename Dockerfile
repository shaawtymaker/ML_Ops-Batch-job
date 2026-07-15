# --- MLOps Task: reproducible, self-contained batch job image ---
FROM python:3.9-slim

# Keep Python output unbuffered so logs stream immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data + config into the image (no host-path dependencies)
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Default one-command run — no hard-coded absolute paths, all via CLI flags.
# Produces /app/metrics.json and /app/run.log inside the container and
# prints the final metrics JSON to stdout.
CMD ["python", "run.py", "--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]
