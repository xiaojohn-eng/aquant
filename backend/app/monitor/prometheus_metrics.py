"""
Prometheus Metrics Integration

Production-grade monitoring for the AQuant system.
Collects: API latency, GPU utilization, strategy performance, data quality.

Reference:
- FastAPI Prometheus middleware (Aliyun, 2025)
- Prometheus GPU exporter (Tencent Cloud, 2025)
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from fastapi import FastAPI, Request
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest, multiprocess)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

# API metrics
API_REQUEST_COUNT = Counter(
    "aquant_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"]
)

API_REQUEST_LATENCY = Histogram(
    "aquant_api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

API_ACTIVE_REQUESTS = Gauge(
    "aquant_api_active_requests",
    "Currently active API requests"
)

# Strategy metrics
DAILY_RECOMMENDATIONS = Gauge(
    "aquant_daily_recommendations",
    "Number of stocks recommended today"
)

STRATEGY_SHARPE = Gauge(
    "aquant_strategy_sharpe_ratio",
    "Current strategy Sharpe ratio"
)

STRATEGY_RETURN = Gauge(
    "aquant_strategy_total_return_pct",
    "Current strategy total return %"
)

STRATEGY_DRAWDOWN = Gauge(
    "aquant_strategy_max_drawdown_pct",
    "Current strategy max drawdown %"
)

# GPU metrics
GPU_UTILIZATION = Gauge(
    "aquant_gpu_utilization_pct",
    "GPU utilization percentage",
    ["gpu_id", "name"]
)

GPU_MEMORY_USED = Gauge(
    "aquant_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["gpu_id"]
)

GPU_MEMORY_TOTAL = Gauge(
    "aquant_gpu_memory_total_mb",
    "GPU total memory in MB",
    ["gpu_id"]
)

GPU_TEMPERATURE = Gauge(
    "aquant_gpu_temperature_c",
    "GPU temperature in Celsius",
    ["gpu_id"]
)

GPU_POWER = Gauge(
    "aquant_gpu_power_w",
    "GPU power draw in Watts",
    ["gpu_id"]
)

# Data quality metrics
DATA_SOURCE_STATUS = Gauge(
    "aquant_data_source_up",
    "Data source availability (1=up, 0=down)",
    ["source_name"]
)

DAILY_DATA_POINTS = Counter(
    "aquant_data_points_fetched_total",
    "Total data points fetched",
    ["source_name", "data_type"]
)

DATA_FETCH_LATENCY = Histogram(
    "aquant_data_fetch_duration_seconds",
    "Data fetch latency",
    ["source_name", "data_type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to collect Prometheus metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        API_ACTIVE_REQUESTS.inc()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            endpoint = request.url.path
            method = request.method
            
            API_REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=method,
                status_code=status_code
            ).inc()
            
            API_REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
            API_ACTIVE_REQUESTS.dec()
        
        return response


class MetricsCollector:
    """Helper to update business-level metrics."""
    
    @staticmethod
    def update_gpu_metrics(gpu_status_list: list) -> None:
        """Update GPU metrics from nvidia-ml-py data."""
        for gpu in gpu_status_list:
            gpu_id = str(gpu.get("index", 0))
            name = gpu.get("name", "unknown")
            
            GPU_UTILIZATION.labels(gpu_id=gpu_id, name=name).set(
                gpu.get("utilization_gpu_pct", 0)
            )
            GPU_MEMORY_USED.labels(gpu_id=gpu_id).set(
                gpu.get("memory_used_mb", 0)
            )
            GPU_MEMORY_TOTAL.labels(gpu_id=gpu_id).set(
                gpu.get("memory_total_mb", 0)
            )
            GPU_TEMPERATURE.labels(gpu_id=gpu_id).set(
                gpu.get("temperature_c", 0)
            )
            GPU_POWER.labels(gpu_id=gpu_id).set(
                gpu.get("power_draw_w", 0)
            )
    
    @staticmethod
    def update_strategy_metrics(metrics: dict) -> None:
        """Update strategy performance metrics."""
        STRATEGY_SHARPE.set(metrics.get("sharpe_ratio", 0))
        STRATEGY_RETURN.set(metrics.get("total_return_pct", 0))
        STRATEGY_DRAWDOWN.set(abs(metrics.get("max_drawdown_pct", 0)))
    
    @staticmethod
    def update_data_source(source: str, is_up: bool) -> None:
        """Update data source health status."""
        DATA_SOURCE_STATUS.labels(source_name=source).set(1 if is_up else 0)
    
    @staticmethod
    def record_data_fetch(source: str, data_type: str, count: int, duration: float) -> None:
        """Record data fetch operation."""
        DAILY_DATA_POINTS.labels(source_name=source, data_type=data_type).inc(count)
        DATA_FETCH_LATENCY.labels(source_name=source, data_type=data_type).observe(duration)


def setup_prometheus(app: FastAPI) -> None:
    """Setup Prometheus metrics endpoint and middleware."""
    app.add_middleware(PrometheusMiddleware)
    
    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    logger.info("Prometheus metrics endpoint mounted at /metrics")
