# ============================================================
#  Trading System — Docker Image
#  Base: Python 3.11 slim
# ============================================================
FROM python:3.11-slim

# System dependencies + Java (required for H2O AutoML)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tzdata \
        openjdk-21-jre-headless \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Java env for H2O
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Set timezone
ENV TZ=Europe/Athens
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3

# Copy application code
COPY . .

# Create dirs
RUN mkdir -p results/daily logs diagnostics /tmp/databento_cache /tmp/alpaca_signal_cache /tmp/signal_cache

EXPOSE 8080

CMD ["python", "main.py", "paper"]
# Rotated credentials deployed
