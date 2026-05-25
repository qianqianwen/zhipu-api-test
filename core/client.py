"""
Zhipu AI API Client - Unified wrapper for all API interactions.

Encapsulates authentication, request/response handling, retry logic,
and logging so that test cases only focus on business logic.
"""
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ZhipuClient:
    """Core API client for Zhipu MaaS platform."""

    def __init__(self, api_key, base_url="https://open.bigmodel.cn/api/paas/v4",
                 timeout=(10, 60), max_retries=3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Session with retry and connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

        # Retry on 5xx and connection errors (not on 4xx)
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request(self, method, path, **kwargs):
        """Unified request method with timing and logging."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)

        logger.info(f">>> {method} {url}")
        if "json" in kwargs:
            logger.debug(f"    Request body: {kwargs['json']}")

        start = time.time()
        resp = self.session.request(method, url, **kwargs)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(f"<<< {resp.status_code} ({elapsed_ms:.0f}ms)")
        if resp.status_code >= 400:
            logger.warning(f"    Error response: {resp.text[:500]}")

        # Attach timing info for assertions
        resp.elapsed_ms = elapsed_ms
        return resp

    # ==================== Chat Completions ====================

    def chat_completion(self, messages, model=None, stream=False, **kwargs):
        """
        Call /chat/completions endpoint.

        Args:
            messages: List of message dicts, e.g. [{"role": "user", "content": "hi"}]
            model: Model name. Defaults to glm-4-flash.
            stream: If True, returns SSE stream response.
            **kwargs: Extra parameters (temperature, top_p, max_tokens, etc.)

        Returns:
            requests.Response object (stream=False) or streaming response (stream=True)
        """
        model = model or "glm-4-flash"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        if stream:
            # For streaming, use iter_lines
            return self._request("POST", "/chat/completions", json=payload,
                                 stream=True, timeout=(10, 120))
        return self._request("POST", "/chat/completions", json=payload)

    # ==================== Embeddings ====================

    def create_embedding(self, input_text, model="embedding-3"):
        """
        Call /embeddings endpoint.

        Args:
            input_text: Text string to embed.
            model: Embedding model name.

        Returns:
            requests.Response object.
        """
        payload = {
            "model": model,
            "input": input_text,
        }
        return self._request("POST", "/embeddings", json=payload)

    # ==================== Models ====================

    def list_models(self):
        """Fetch available model list (GET /models)."""
        return self._request("GET", "/models")

    # ==================== Files (for fine-tuning) ====================

    def upload_file(self, file_path, purpose="fine-tune"):
        """Upload a file for fine-tuning or batch processing."""
        with open(file_path, "rb") as f:
            return self._request(
                "POST", "/files",
                files={"file": f},
                data={"purpose": purpose},
            )

    # ==================== Token Counting ====================

    def count_tokens(self, messages, model=None):
        """Estimate token count for given messages (utility method)."""
        # Rough estimation: ~1.5 tokens per Chinese character, ~0.25 per English word
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return int(total_chars * 1.5)


class ZhipuClientWithoutAuth(ZhipuClient):
    """Client without valid authentication - for negative testing."""

    def __init__(self, base_url="https://open.bigmodel.cn/api/paas/v4"):
        super().__init__(api_key="invalid_api_key_for_testing", base_url=base_url)


class ZhipuClientWithExpiredKey(ZhipuClient):
    """Client with expired/malformed API key - for auth boundary testing."""

    def __init__(self, base_url="https://open.bigmodel.cn/api/paas/v4"):
        super().__init__(api_key="expired.key.abc123", base_url=base_url)
