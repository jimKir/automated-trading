# docker/

Docker configuration for the market data platform (separate from the root Dockerfile used by the trading system).

## `Dockerfile`

Python 3.11-slim container for the market data platform. Exposes port 8080 (HTTP health) and 9090 (Prometheus metrics). Entry point: `market-data` CLI.

## `docker-compose.yml`

Local development stack with 4 services:

| Service | Port | Purpose |
|---|---|---|
| market-data | 8080, 9090 | Main data platform |
| prefect-server | 4200 | Workflow orchestration UI |
| prometheus | 9090 | Metrics scraping (15s interval) |
| grafana | 3000 | Dashboards (default admin/admin) |

## `prometheus.yml`

Prometheus configuration scraping `market-data:9090/metrics` every 15 seconds.

## Note

The **trading system** uses the root `Dockerfile` and `docker-compose.yml`. This directory's configs are for the **market data platform** component only.
