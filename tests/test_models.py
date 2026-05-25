"""
Model-related Tests.

Covers: model listing, invalid model handling, model-specific behavior,
and cross-model consistency tests.
"""
import pytest
from core.assertions import ResponseAssertions, ChatAssertions
from core.data_factory import ModelFactory, PromptFactory


class TestModelList:
    """Model listing endpoint tests."""

    @pytest.mark.smoke
    def test_list_models(self, client):
        """Verify /models endpoint returns model list."""
        resp = client.list_models()
        ResponseAssertions.assert_status_ok(resp)
        data = resp.json()
        assert "data" in data, f"Missing 'data' in model list response: {data.keys()}"
        assert len(data["data"]) > 0, "Model list is empty"

    def test_model_list_structure(self, client):
        """Verify each model entry has required fields."""
        resp = client.list_models()
        data = resp.json()
        for model in data["data"]:
            assert "id" in model, f"Missing 'id' in model entry: {model}"


class TestModelValidation:
    """Model name validation tests."""

    def test_invalid_model_name(self, client):
        """Verify invalid model name returns appropriate error."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, model="non-existent-model-xyz")
        assert resp.status_code != 200, (
            f"Expected error for invalid model, got 200"
        )

    def test_empty_model_name(self, client):
        """Verify empty model name is handled."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, model="")
        assert resp.status_code != 200, (
            f"Expected error for empty model name, got 200"
        )

    @pytest.mark.parametrize("model_name", ModelFactory.invalid_models())
    def test_various_invalid_models(self, client, model_name):
        """Test various invalid model names are properly rejected."""
        messages = PromptFactory.simple_message()
        resp = client.chat_completion(messages, model=model_name)
        assert resp.status_code in [400, 404, 422], (
            f"Expected 4xx for model '{model_name}', got {resp.status_code}"
        )


class TestModelSpecific:
    """Model-specific behavior tests."""

    @pytest.mark.smoke
    def test_glm4_flash(self, client):
        """Verify glm-4-flash model works correctly."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages, model="glm-4-flash")
        ChatAssertions.assert_chat_response_valid(resp)

    def test_default_model(self, client):
        """Verify default model (no explicit model param) works."""
        messages = [{"role": "user", "content": "你好"}]
        resp = client.chat_completion(messages)
        ChatAssertions.assert_chat_response_valid(resp)


class TestEmbedding:
    """Embedding API tests."""

    @pytest.mark.smoke
    def test_basic_embedding(self, client):
        """Verify basic embedding generation."""
        resp = client.create_embedding("什么是人工智能")
        ResponseAssertions.assert_status_ok(resp)
        data = resp.json()
        assert "data" in data, f"Missing 'data' in embedding response"
        assert len(data["data"]) > 0, "Empty embedding result"

    def test_embedding_vector_dimension(self, client):
        """Verify embedding vector has expected dimensions."""
        resp = client.create_embedding("测试文本")
        if resp.status_code == 200:
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) > 0, "Embedding vector is empty"

    def test_embedding_empty_input(self, client):
        """Test embedding with empty input."""
        resp = client.create_embedding("")
        # Should either work or return meaningful error
        assert resp.status_code in [200, 400], (
            f"Unexpected status {resp.status_code} for empty embedding input"
        )

    def test_embedding_long_input(self, client):
        """Test embedding with long input text."""
        long_text = "人工智能是计算机科学的一个分支。" * 100
        resp = client.create_embedding(long_text)
        assert resp.status_code in [200, 400], (
            f"Unexpected status {resp.status_code} for long embedding input"
        )
