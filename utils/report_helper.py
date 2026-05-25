"""
Report Helper - Utilities for test reporting and metrics collection.

Provides decorators and helpers for capturing performance metrics
and generating test summaries.
"""
import time
import json
import logging
import functools
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and aggregate performance metrics during test execution."""

    def __init__(self):
        self.metrics = []

    def record(self, name, value, unit="ms", tags=None):
        """Record a single metric data point."""
        entry = {
            "name": name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {},
        }
        self.metrics.append(entry)
        logger.debug(f"Metric: {name}={value}{unit} {tags or ''}")

    def get_summary(self, name):
        """Get statistical summary for a named metric."""
        values = [m["value"] for m in self.metrics if m["name"] == name]
        if not values:
            return {}
        sorted_values = sorted(values)
        count = len(sorted_values)
        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[min(int(count * 0.95), count - 1)],
            "p99": sorted_values[min(int(count * 0.99), count - 1)],
        }

    def report(self):
        """Generate a full metrics report."""
        names = set(m["name"] for m in self.metrics)
        report = {}
        for name in sorted(names):
            report[name] = self.get_summary(name)
        return report

    def save_to_file(self, filepath):
        """Save raw metrics to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "raw_metrics": self.metrics,
                "summary": self.report(),
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {filepath}")


# Global metrics collector instance
metrics = MetricsCollector()


def timed(metric_name=None):
    """Decorator to measure and record function execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or func.__name__
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000
                metrics.record(name, round(elapsed_ms, 1), unit="ms",
                               tags={"status": "success"})
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                metrics.record(name, round(elapsed_ms, 1), unit="ms",
                               tags={"status": "error", "error": str(e)})
                raise
        return wrapper
    return decorator
