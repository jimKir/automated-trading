# ============================================================
#  Trading System — Docker Image
#  Base: Python 3.11 slim (matches AWS Lambda / ECS defaults)
# ============================================================
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone (configurable via TZ env var at runtime)
ENV TZ=Europe/Athens
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3  # AWS SDK for SES / S3 reporting

# Copy application code
COPY . .

# Create results directories
RUN mkdir -p results/daily logs

# Expose health check port
EXPOSE 8080

# Default: paper trading mode
# Override CMD in docker-compose or ECS task definition
CMD ["python", "main.py", "paper"]
