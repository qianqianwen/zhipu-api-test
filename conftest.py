"""
Pytest Configuration and Shared Fixtures.

This is the central place for:
- Client initialization (with API key from environment)
- Shared fixtures used across all test modules
- Custom pytest hooks for reporting
"""
import os
import pytest
import logging
from core.client import ZhipuClient
from utils.config_loader import load_config
from utils.report_helper import metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--env", action="store", default="base",
        help="Test environment: base, staging"
    )
    parser.addoption(
        "--api-key", action="store", default=None,
        help="Zhipu API key (overrides ZHIPU_API_KEY env var)"
    )


@pytest.fixture(scope="session")
def config(request):
    """Load configuration for the test session."""
    env = request.config.getoption("--env")
    return load_config(env)


@pytest.fixture(scope="session")
def api_key(request):
    """Resolve API key from CLI option or environment variable."""
    key = request.config.getoption("--api-key")
    if not key:
        key = os.getenv("ZHIPU_API_KEY", "")
    if not key:
        pytest.skip("ZHIPU_API_KEY not set. Skipping API tests.")
    return key


@pytest.fixture(scope="session")
def client(api_key, config):
    """
    Create a shared ZhipuClient for the entire test session.

    Scope=session means one client is reused across all tests,
    which is efficient for connection pooling.
    """
    base_url = config["api"]["base_url"]
    c = ZhipuClient(api_key=api_key, base_url=base_url)
    logger.info(f"ZhipuClient initialized: base_url={base_url}")
    return c


@pytest.fixture(scope="function")
def fresh_client(api_key, config):
    """
    Create a fresh client per test function.
    Use when test needs isolated session state.
    """
    return ZhipuClient(
        api_key=api_key,
        base_url=config["api"]["base_url"],
    )


# ==================== Hooks ====================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print performance metrics summary after test session."""
    report = metrics.report()
    if report:
        terminalreporter.write_sep("=", "Performance Metrics Summary")
        for name, summary in report.items():
            terminalreporter.write_line(
                f"  {name}: "
                f"avg={summary['avg']:.0f}ms "
                f"p95={summary['p95']:.0f}ms "
                f"p99={summary['p99']:.0f}ms "
                f"min={summary['min']:.0f}ms "
                f"max={summary['max']:.0f}ms "
                f"(n={summary['count']})"
            )
