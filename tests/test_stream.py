"""
Streaming (SSE) Output Tests.

Covers: stream completeness, TTFT measurement, content integrity,
disconnection handling, and concurrent streaming.
"""
import pytest
from core.assertions import StreamAssertions, ResponseAssertions
from core.sse_parser import parse_sse_stream
from core.data_factory import PromptFactory


class TestStreamBasic:
    """Basic streaming functionality tests."""

    @pytest.mark.smoke
    def test_stream_basic_response(self, client):
        """Verify streaming returns proper SSE events with [DONE]."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages, stream=True)
        ResponseAssertions.assert_status_ok(resp)

        result = parse_sse_stream(resp)
        StreamAssertions.assert_stream_complete(result)
        StreamAssertions.assert_stream_has_content(result)
        StreamAssertions.assert_stream_no_errors(result)

    def test_stream_content_matches_non_stream(self, client):
        """Verify streaming and non-streaming return semantically similar content."""
        messages = [{"role": "user", "content": "1+1等于几？只回答数字"}]

        # Non-streaming
        resp_sync = client.chat_completion(messages, stream=False)
        content_sync = resp_sync.json()["choices"][0]["message"]["content"]

        # Streaming
        resp_stream = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp_stream)

        # Both should contain "2"
        assert "2" in content_sync, f"Non-stream response missing '2': {content_sync}"
        assert "2" in result.full_content, f"Stream response missing '2': {result.full_content}"

    def test_stream_multi_turn(self, client):
        """Verify streaming works with multi-turn conversations."""
        messages = PromptFactory.multi_turn_message(index=0)
        resp = client.chat_completion(messages, stream=True)
        ResponseAssertions.assert_status_ok(resp)

        result = parse_sse_stream(resp)
        StreamAssertions.assert_stream_complete(result)
        StreamAssertions.assert_stream_has_content(result)


class TestStreamIntegrity:
    """Stream data integrity and completeness tests."""

    def test_stream_done_signal(self, client):
        """Verify stream always ends with [DONE] signal."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        assert result.has_done_signal, (
            f"Missing [DONE] signal. Got {len(result.events)} events. "
            f"Last event: {result.events[-1] if result.events else 'none'}"
        )

    def test_stream_event_format(self, client):
        """Verify each SSE event has proper JSON structure."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        for event in result.events:
            if event.is_done:
                continue
            data = event.json_data
            assert data is not None, f"Invalid JSON in event: {event.data}"
            assert "choices" in data, f"Missing 'choices' in event: {data}"

    def test_stream_no_content_loss(self, client):
        """Verify concatenated stream content is coherent (not garbled)."""
        messages = [{"role": "user", "content": "从1数到10，每个数字占一行"}]
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        content = result.full_content
        assert len(content) > 0, "Empty content"
        # Should contain at least some numbers
        found_numbers = sum(1 for n in "12345" if n in content)
        assert found_numbers >= 3, f"Expected numbers in response, got: {content[:200]}"

    def test_stream_events_ordered(self, client):
        """Verify SSE events arrive in chronological order."""
        messages = PromptFactory.complex_message()
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        timestamps = [e.timestamp for e in result.events]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Events out of order at index {i}: "
                f"{timestamps[i - 1]} > {timestamps[i]}"
            )


class TestStreamPerformance:
    """Streaming performance tests - TTFT and throughput."""

    @pytest.mark.performance
    def test_ttft_acceptable(self, client):
        """Verify Time To First Token is within acceptable range."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        StreamAssertions.assert_stream_complete(result)
        # TTFT should be < 3 seconds for simple prompts
        StreamAssertions.assert_ttft_within(result, max_ms=3000)

    @pytest.mark.performance
    def test_stream_total_time(self, client):
        """Verify total streaming time is acceptable."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        StreamAssertions.assert_stream_complete(result)
        StreamAssertions.assert_total_time_within(result, max_ms=30000)

    @pytest.mark.performance
    def test_ttft_consistency(self, client):
        """Run multiple stream requests and check TTFT consistency."""
        ttft_values = []
        for _ in range(3):
            messages = PromptFactory.random_message()
            resp = client.chat_completion(messages, stream=True)
            result = parse_sse_stream(resp)
            if result.ttft_ms is not None:
                ttft_values.append(result.ttft_ms)

        assert len(ttft_values) >= 2, "Too few successful stream requests"
        avg_ttft = sum(ttft_values) / len(ttft_values)
        max_ttft = max(ttft_values)
        # Max TTFT should not be more than 3x the average (detect outliers)
        assert max_ttft <= avg_ttft * 3, (
            f"TTFT too inconsistent: avg={avg_ttft:.0f}ms, max={max_ttft:.0f}ms, "
            f"all={[f'{v:.0f}' for v in ttft_values]}"
        )


class TestStreamWithParameters:
    """Streaming with various API parameters."""

    def test_stream_with_max_tokens(self, client):
        """Verify streaming respects max_tokens limit."""
        messages = [{"role": "user", "content": "详细介绍Python的历史"}]
        resp = client.chat_completion(messages, stream=True, max_tokens=20)
        result = parse_sse_stream(resp)

        StreamAssertions.assert_stream_complete(result)
        # Content should be relatively short due to max_tokens
        assert len(result.full_content) < 200, (
            f"Content too long for max_tokens=20: {len(result.full_content)} chars"
        )

    def test_stream_with_temperature_zero(self, client):
        """Verify streaming with temperature=0 (deterministic)."""
        messages = [{"role": "user", "content": "1+1等于几？只回答数字"}]

        results = []
        for _ in range(2):
            resp = client.chat_completion(messages, stream=True, temperature=0.01)
            result = parse_sse_stream(resp)
            results.append(result.full_content.strip())

        # With temperature ~0, responses should be very similar
        # (not asserting exact match due to model non-determinism)
        for r in results:
            assert "2" in r, f"Expected '2' in deterministic response, got: {r}"

    def test_stream_with_system_prompt(self, client):
        """Verify streaming works with system prompt."""
        messages = PromptFactory.system_prompt_message(
            system_content="你是一个诗人，请用诗歌形式回答",
            user_content="描述春天"
        )
        resp = client.chat_completion(messages, stream=True)
        result = parse_sse_stream(resp)

        StreamAssertions.assert_stream_complete(result)
        StreamAssertions.assert_stream_has_content(result)
