"""
Chat Completion API Tests.

Covers: basic functionality, multi-turn conversation, parameter validation,
boundary testing, and error handling.
"""
import pytest
from core.assertions import ChatAssertions, ResponseAssertions, PerformanceAssertions
from core.data_factory import PromptFactory, ParameterFactory


class TestChatBasic:
    """Basic chat completion functionality tests."""

    @pytest.mark.smoke
    def test_simple_chat(self, client):
        """Verify basic chat completion returns valid response."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages)
        ChatAssertions.assert_chat_response_valid(resp)

    @pytest.mark.smoke
    def test_chat_response_has_content(self, client):
        """Verify chat response contains meaningful content."""
        messages = [{"role": "user", "content": "1+1等于几？"}]
        resp = client.chat_completion(messages)
        content = ChatAssertions.assert_chat_content_not_empty(resp)
        assert "2" in content, f"Expected '2' in response, got: {content}"

    def test_chat_response_role(self, client):
        """Verify response message role is 'assistant'."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages)
        ChatAssertions.assert_chat_role(resp, "assistant")

    def test_chat_usage_tokens(self, client):
        """Verify token usage is reported correctly."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages)
        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


class TestChatMultiTurn:
    """Multi-turn conversation tests."""

    def test_two_turn_conversation(self, client):
        """Verify model maintains context across turns."""
        messages = [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明！"},
            {"role": "user", "content": "我叫什么名字？"},
        ]
        resp = client.chat_completion(messages)
        content = ChatAssertions.assert_chat_content_not_empty(resp)
        assert "小明" in content, f"Model lost context, response: {content}"

    def test_system_prompt(self, client):
        """Verify system prompt affects model behavior."""
        messages = PromptFactory.system_prompt_message(
            system_content="请用英文回答所有问题",
            user_content="你好"
        )
        resp = client.chat_completion(messages)
        ChatAssertions.assert_chat_response_valid(resp)

    @pytest.mark.parametrize("conversation_idx", [0, 1, 2])
    def test_predefined_multi_turn(self, client, conversation_idx):
        """Test predefined multi-turn conversation templates."""
        messages = PromptFactory.multi_turn_message(index=conversation_idx)
        resp = client.chat_completion(messages)
        ChatAssertions.assert_chat_response_valid(resp)


class TestChatParameters:
    """API parameter validation and boundary tests."""

    @pytest.mark.parametrize("temperature", ParameterFactory.temperature_values())
    def test_valid_temperature(self, client, temperature):
        """Verify various valid temperature values are accepted."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, temperature=temperature)
        # Temperature 0-2 should all return valid responses
        if resp.status_code == 200:
            ChatAssertions.assert_chat_response_valid(resp)

    @pytest.mark.parametrize("max_tokens", [1, 10, 50, 100])
    def test_max_tokens_limit(self, client, max_tokens):
        """Verify max_tokens parameter constrains output length."""
        messages = [{"role": "user", "content": "请详细介绍Python编程语言的历史"}]
        resp = client.chat_completion(messages, max_tokens=max_tokens)
        if resp.status_code == 200:
            data = resp.json()
            completion_tokens = data["usage"]["completion_tokens"]
            # Allow some flexibility, but should be roughly within limit
            assert completion_tokens <= max_tokens + 5, (
                f"completion_tokens={completion_tokens} exceeds max_tokens={max_tokens}"
            )

    @pytest.mark.parametrize("top_p", ParameterFactory.top_p_values())
    def test_valid_top_p(self, client, top_p):
        """Verify various valid top_p values are accepted."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, top_p=top_p)
        if resp.status_code == 200:
            ChatAssertions.assert_chat_response_valid(resp)


class TestChatEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_message_content(self, client):
        """Test with empty message content."""
        messages = [{"role": "user", "content": ""}]
        resp = client.chat_completion(messages)
        # Should either return an error or handle gracefully
        assert resp.status_code in [200, 400], (
            f"Unexpected status {resp.status_code} for empty content"
        )

    def test_very_long_input(self, client):
        """Test with very long input text."""
        long_text = "请重复这句话。" * 500  # ~3000 chars
        messages = [{"role": "user", "content": long_text}]
        resp = client.chat_completion(messages)
        # Should either process or return context length error
        assert resp.status_code in [200, 400], (
            f"Unexpected status {resp.status_code} for long input"
        )

    def test_special_characters(self, client):
        """Test with special characters and emoji."""
        messages = [{"role": "user", "content": "解释这些符号的含义：🎉 © ® ™ ½ ¼"}]
        resp = client.chat_completion(messages)
        ChatAssertions.assert_chat_response_valid(resp)

    def test_empty_messages_array(self, client):
        """Test with empty messages array — should return error."""
        resp = client.chat_completion(messages=[])
        assert resp.status_code != 200, "Empty messages should not succeed"

    def test_missing_role_field(self, client):
        """Test with malformed message (missing role)."""
        messages = [{"content": "你好"}]  # Missing 'role'
        resp = client.chat_completion(messages)
        assert resp.status_code in [200, 400], (
            f"Unexpected status {resp.status_code} for missing role"
        )

    def test_invalid_role(self, client):
        """Test with invalid role value."""
        messages = [{"role": "invalid_role", "content": "你好"}]
        resp = client.chat_completion(messages)
        assert resp.status_code in [200, 400], (
            f"Unexpected status {resp.status_code} for invalid role"
        )


class TestChatPerformance:
    """Basic performance tests for chat completion."""

    @pytest.mark.performance
    def test_simple_chat_latency(self, client):
        """Verify simple chat response time is acceptable."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages)
        ResponseAssertions.assert_status_ok(resp)
        PerformanceAssertions.assert_latency_within(resp, max_ms=10000)

    @pytest.mark.performance
    def test_batch_latency_consistency(self, client):
        """Run multiple requests and check latency consistency."""
        latencies = []
        for _ in range(5):
            messages = PromptFactory.random_message()
            resp = client.chat_completion(messages)
            if resp.status_code == 200:
                latencies.append(resp.elapsed_ms)

        assert len(latencies) >= 3, "Too many failed requests"
        PerformanceAssertions.assert_latency_p99(latencies, max_ms=15000)
