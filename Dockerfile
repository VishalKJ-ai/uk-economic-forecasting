# =============================================================================
# UK Economic Forecasting Pipeline — Dockerfile
# Multi-stage build for minimal final image size
# =============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (cached layer) ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────
COPY config/ config/
COPY data/sample/ data/sample/
COPY src/ src/
COPY setup.py .
COPY README.md .

# Create output directories
RUN mkdir -p data/raw data/processed models outputs/figures outputs/forecasts

# ── Entry point ─────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["--mode", "sample"]
