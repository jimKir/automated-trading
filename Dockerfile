# ============================================================
#  Trading System — Docker Image
#  Base: Python 3.11 slim
# ============================================================
FROM python:3.11-slim

# System dependencies + Java (required for H2O AutoML)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        procps \
        tzdata \
        openjdk-21-jre-headless \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Java env for H2O
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Build metadata (passed by CI)
ARG BUILD_SHA=unknown
ARG BUILD_TIMESTAMP=unknown
ARG BUILD_VERSION=dev-unknown
ENV BUILD_SHA=${BUILD_SHA}
ENV BUILD_TIMESTAMP=${BUILD_TIMESTAMP}
ENV BUILD_VERSION=${BUILD_VERSION}

# Set timezone
ENV TZ=Europe/Athens
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python dependencies (fully locked — see requirements.in for source)
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy application code
COPY . .

# Create dirs
RUN mkdir -p results/daily logs diagnostics /tmp/databento_cache /tmp/alpaca_signal_cache /tmp/signal_cache

EXPOSE 8080

CMD ["python", "main.py", "paper"]
# Rotated credentials deployed
