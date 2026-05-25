"""
Custom Assertions for Zhipu API Testing.

Provides reusable assertion methods that encapsulate common validation
patterns, making test cases cleaner and more readable.
"""
import logging

logger = logging.getLogger(__name__)


class ResponseAssertions:
    """Assertions for HTTP response validation."""

    @staticmethod
    def assert_status_ok(resp, expected_code=200):
        """Assert response status code matches expected."""
        assert resp.status_code == expected_code, (
            f"Expected status {expected_code}, got {resp.status_code}. "
            f"Response: {resp.text[:500]}"
        )

    @staticmethod
    def assert_status_error(resp, expected_code):
        """Assert response returns expected error code."""
        assert resp.status_code == expected_code, (
            f"Expected error {expected_code}, got {resp.status_code}. "
            f"Response: {resp.text[:500]}"
        )

    @staticmethod
    def assert_json_structure(resp, required_keys):
        """Assert response JSON contains all required keys."""
        data = resp.json()
        for key in required_keys:
            assert key in data, f"Missing required key '{key}' in response: {list(data.keys())}"

    @staticmethod
    def assert_error_response(resp, expected_code=None):
        """Assert response is a proper error with code and message."""
        data = resp.json()
        assert "error" in data, f"Expected error response, got: {data}"
        if expected_code:
            error_code = data["error"].get("code")
            assert error_code == expected_code, (
                f"Expected error code '{expected_code}', got '{error_code}'"
            )


class ChatAssertions:
    """Assertions specific to chat completion responses."""

    @staticmethod
    def assert_chat_response_valid(resp):
        """Validate a complete (non-streaming) chat response."""
        ResponseAssertions.assert_status_ok(resp)
        data = resp.json()

        # Top-level structure
        assert "choices" in data, f"Missing 'choices' in response: {data.keys()}"
        assert len(data["choices"]) > 0, "Empty choices array"

        # Choice structure
        choice = data["choices"][0]
        assert "message" in choice, f"Missing 'message' in choice: {choice.keys()}"
        assert "content" in choice["message"], f"Missing 'content' in message"
        assert len(choice["message"]["content"]) > 0, "Empty content in response"

        # Usage info
        assert "usage" in data, "Missing 'usage' in response"
        usage = data["usage"]
        assert usage.get("total_tokens", 0) > 0, f"Invalid token count: {usage}"

        return data

    @staticmethod
    def assert_chat_content_not_empty(resp):
        """Assert the chat response has non-empty content."""
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content and len(content.strip()) > 0, "Chat response content is empty"
        return content

    @staticmethod
    def assert_chat_role(resp, expected_role="assistant"):
        """Assert the response message has the expected role."""
        data = resp.json()
        role = data["choices"][0]["message"]["role"]
        assert role == expected_role, f"Expected role '{expected_role}', got '{role}'"


class StreamAssertions:
    """Assertions for SSE streaming responses."""

    @staticmethod
    def assert_stream_complete(sse_result):
        """Assert SSE stream completed properly."""
        assert sse_result.has_done_signal, (
            f"Stream did not end with [DONE]. Events: {len(sse_result.events)}, "
            f"Errors: {sse_result.errors}"
        )

    @staticmethod
    def assert_stream_has_content(sse_result):
        """Assert stream produced non-empty content."""
        assert len(sse_result.full_content) > 0, (
            f"Stream produced no content. Events: {len(sse_result.events)}"
        )

    @staticmethod
    def assert_stream_no_errors(sse_result):
        """Assert no errors occurred during stream parsing."""
        assert len(sse_result.errors) == 0, (
            f"Stream had errors: {sse_result.errors}"
        )

    @staticmethod
    def assert_ttft_within(sse_result, max_ms):
        """Assert Time To First Token is within acceptable range."""
        ttft = sse_result.ttft_ms
        assert ttft is not None, "TTFT not measured (no tokens received?)"
        assert ttft <= max_ms, (
            f"TTFT {ttft:.0f}ms exceeds threshold {max_ms}ms"
        )

    @staticmethod
    def assert_total_time_within(sse_result, max_ms):
        """Assert total stream time is within acceptable range."""
        total = sse_result.total_time_ms
        assert total is not None, "Total time not measured"
        assert total <= max_ms, (
            f"Total time {total:.0f}ms exceeds threshold {max_ms}ms"
        )


class PerformanceAssertions:
    """Assertions for performance/latency validation."""

    @staticmethod
    def assert_latency_within(resp, max_ms):
        """Assert response latency is within acceptable range."""
        elapsed = getattr(resp, "elapsed_ms", resp.elapsed.total_seconds() * 1000)
        assert elapsed <= max_ms, (
            f"Latency {elapsed:.0f}ms exceeds threshold {max_ms}ms"
        )

    @staticmethod
    def assert_latency_p99(latencies, max_ms):
        """Assert P99 latency is within threshold."""
        if not latencies:
            return
        sorted_lat = sorted(latencies)
        p99_idx = int(len(sorted_lat) * 0.99)
        p99 = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]
        assert p99 <= max_ms, (
            f"P99 latency {p99:.0f}ms exceeds threshold {max_ms}ms"
        )
