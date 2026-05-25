"""
Authentication and Authorization Tests.

Covers: API key validation, expired keys, missing auth, rate limiting,
and security boundary testing.
"""
import pytest
from core.client import ZhipuClient, ZhipuClientWithoutAuth, ZhipuClientWithExpiredKey
from core.assertions import ResponseAssertions


class TestAuthBasic:
    """Basic authentication tests."""

    @pytest.mark.smoke
    def test_valid_api_key(self, client):
        """Verify valid API key returns successful response."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages)
        ResponseAssertions.assert_status_ok(resp)

    def test_invalid_api_key(self):
        """Verify invalid API key is rejected."""
        invalid_client = ZhipuClientWithoutAuth()
        messages = [{"role": "user", "content": "你好"}]
        resp = invalid_client.chat_completion(messages)
        assert resp.status_code == 401, (
            f"Expected 401 for invalid key, got {resp.status_code}"
        )

    def test_expired_api_key(self):
        """Verify expired/malformed API key is rejected."""
        expired_client = ZhipuClientWithExpiredKey()
        messages = [{"role": "user", "content": "你好"}]
        resp = expired_client.chat_completion(messages)
        assert resp.status_code in [401, 403], (
            f"Expected 401/403 for expired key, got {resp.status_code}"
        )

    def test_empty_api_key(self):
        """Verify empty API key is rejected."""
        empty_client = ZhipuClient(api_key="")
        messages = [{"role": "user", "content": "你好"}]
        resp = empty_client.chat_completion(messages)
        assert resp.status_code in [401, 403], (
            f"Expected 401/403 for empty key, got {resp.status_code}"
        )

    def test_no_auth_header(self):
        """Verify request without Authorization header is rejected."""
        import requests
        resp = requests.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            json={
                "model": "glm-4-flash",
                "messages": [{"role": "user", "content": "你好"}],
            },
            timeout=10,
        )
        assert resp.status_code in [401, 403], (
            f"Expected 401/403 without auth header, got {resp.status_code}"
        )


class TestAuthSecurity:
    """Security-related authentication tests."""

    def test_sql_injection_in_api_key(self):
        """Verify SQL injection in API key doesn't cause server error."""
        malicious_client = ZhipuClient(api_key="' OR 1=1 --")
        messages = [{"role": "user", "content": "你好"}]
        resp = malicious_client.chat_completion(messages)
        # Should return auth error, not 500
        assert resp.status_code in [401, 403], (
            f"Expected auth error for SQL injection key, got {resp.status_code}"
        )

    def test_xss_in_api_key(self):
        """Verify XSS payload in API key is handled safely."""
        xss_client = ZhipuClient(api_key="<script>alert(1)</script>")
        messages = [{"role": "user", "content": "你好"}]
        resp = xss_client.chat_completion(messages)
        assert resp.status_code in [401, 403], (
            f"Expected auth error for XSS key, got {resp.status_code}"
        )

    def test_very_long_api_key(self):
        """Verify excessively long API key is handled gracefully."""
        long_client = ZhipuClient(api_key="a" * 10000)
        messages = [{"role": "user", "content": "你好"}]
        resp = long_client.chat_completion(messages)
        assert resp.status_code in [401, 403, 400, 413], (
            f"Expected error for long key, got {resp.status_code}"
        )


class TestAuthStreamMode:
    """Authentication tests specifically for streaming mode."""

    def test_invalid_key_stream(self):
        """Verify invalid API key is rejected for stream requests too."""
        invalid_client = ZhipuClientWithoutAuth()
        messages = [{"role": "user", "content": "你好"}]
        resp = invalid_client.chat_completion(messages, stream=True)
        assert resp.status_code in [401, 403], (
            f"Expected 401/403 for invalid key in stream mode, got {resp.status_code}"
        )

    @pytest.mark.smoke
    def test_valid_key_stream(self, client):
        """Verify valid API key works for streaming requests."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages, stream=True)
        ResponseAssertions.assert_status_ok(resp)
