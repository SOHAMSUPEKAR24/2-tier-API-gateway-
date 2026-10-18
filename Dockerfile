# ==============================================================================
# Dockerfile: Adaptive Two-Tier Malware Detection API Gateway
# Reproducible build for Tier 1 screening and Tier 2 deep verification
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from locked requirements
COPY requirements.lock /app/requirements.lock
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.lock

# Copy codebase
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
