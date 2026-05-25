"""
Configuration Loader - Multi-environment config management.

Supports loading base config with environment-specific overrides.
API keys are resolved from environment variables for security.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


def deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(env=None):
    """
    Load configuration for the given environment.

    Priority: env-specific config > base config > environment variables

    Args:
        env: Environment name (e.g., "staging"). If None, uses TEST_ENV env var.

    Returns:
        Dict with merged configuration.
    """
    env = env or os.getenv("TEST_ENV", "base")

    # Load base config
    base_path = os.path.join(CONFIG_DIR, "base.yaml")
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Load env-specific override if exists
    env_path = os.path.join(CONFIG_DIR, f"{env}.yaml")
    if os.path.exists(env_path) and env != "base":
        with open(env_path, "r", encoding="utf-8") as f:
            env_config = yaml.safe_load(f) or {}
        config = deep_merge(config, env_config)

    # Resolve API key from environment variable
    api_key = os.getenv("ZHIPU_API_KEY", "")
    if config.get("api", {}).get("api_key", "").startswith("${"):
        config["api"]["api_key"] = api_key

    if not config["api"]["api_key"]:
        logger.warning(
            "ZHIPU_API_KEY not set. Set it via: export ZHIPU_API_KEY='your-key-here'"
        )

    return config


def get_api_key():
    """Get API key from environment variable."""
    key = os.getenv("ZHIPU_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ZHIPU_API_KEY environment variable is not set. "
            "Get your API key from https://open.bigmodel.cn/"
        )
    return key


def get_base_url(env=None):
    """Get API base URL for the given environment."""
    config = load_config(env)
    return config["api"]["base_url"]
