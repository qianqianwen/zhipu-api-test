"""
Test Data Factory - Generate diverse test data for API testing.

Provides parameterized test data to avoid cache hits and ensure
realistic testing conditions.
"""
import random
import string


class PromptFactory:
    """Generate diverse prompts to avoid cache effects during testing."""

    # Basic conversation templates
    SIMPLE_PROMPTS = [
        "你好",
        "今天天气怎么样？",
        "1+1等于几？",
        "请用一句话介绍自己",
        "什么是人工智能？",
    ]

    # Complex prompts for longer responses
    COMPLEX_PROMPTS = [
        "请详细解释什么是分布式系统的CAP定理，并举例说明",
        "用Python实现一个简单的LRU缓存，要求支持get和put操作",
        "对比分析微服务架构和单体架构的优缺点，至少列出5点",
        "解释TCP三次握手的过程，以及为什么需要三次而不是两次",
        "什么是RESTful API设计规范？请列出关键原则并举例",
    ]

    # Prompts that test edge cases
    EDGE_CASE_PROMPTS = [
        "",                          # Empty
        " ",                         # Whitespace only
        "a" * 10000,                 # Very long input
        "🎉🎊🎈",                   # Emoji only
        "<script>alert(1)</script>", # XSS attempt
        "' OR 1=1 --",              # SQL injection attempt
        "\n\n\n",                    # Newlines only
        "Hello\x00World",           # Null byte
    ]

    # Multi-turn conversation templates
    MULTI_TURN_CONVERSATIONS = [
        [
            {"role": "user", "content": "我想学习Python编程"},
            {"role": "assistant", "content": "很好！Python是一门非常适合入门的编程语言。你有什么编程基础吗？"},
            {"role": "user", "content": "我有一些Java基础"},
        ],
        [
            {"role": "system", "content": "你是一个专业的技术顾问"},
            {"role": "user", "content": "什么是微服务架构？"},
            {"role": "assistant", "content": "微服务架构是将应用拆分为小型独立服务的设计模式。"},
            {"role": "user", "content": "它和单体架构相比有什么优势？"},
        ],
        [
            {"role": "system", "content": "请用简短的语言回答问题"},
            {"role": "user", "content": "什么是API？"},
        ],
    ]

    @classmethod
    def simple_message(cls):
        """Return a random simple user message."""
        prompt = random.choice(cls.SIMPLE_PROMPTS)
        return [{"role": "user", "content": prompt}]

    @classmethod
    def complex_message(cls):
        """Return a random complex user message for longer responses."""
        prompt = random.choice(cls.COMPLEX_PROMPTS)
        return [{"role": "user", "content": prompt}]

    @classmethod
    def edge_case_message(cls, index=None):
        """Return an edge case message by index or random."""
        if index is not None:
            prompt = cls.EDGE_CASE_PROMPTS[index]
        else:
            prompt = random.choice(cls.EDGE_CASE_PROMPTS)
        return [{"role": "user", "content": prompt}]

    @classmethod
    def multi_turn_message(cls, index=None):
        """Return a multi-turn conversation."""
        if index is not None:
            return cls.MULTI_TURN_CONVERSATIONS[index]
        return random.choice(cls.MULTI_TURN_CONVERSATIONS)

    @classmethod
    def random_message(cls, min_length=10, max_length=200):
        """Generate a truly random message to bypass any caching."""
        length = random.randint(min_length, max_length)
        topics = ["技术", "科学", "历史", "数学", "编程", "网络", "数据库", "AI"]
        topic = random.choice(topics)
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        prompt = f"请简要介绍关于{topic}的一个知识点 (ref:{suffix})"
        return [{"role": "user", "content": prompt}]

    @classmethod
    def system_prompt_message(cls, system_content, user_content):
        """Create a message with custom system prompt."""
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def batch_messages(cls, count=10):
        """Generate a batch of diverse messages for load testing."""
        messages = []
        for i in range(count):
            if i % 3 == 0:
                messages.append(cls.simple_message())
            elif i % 3 == 1:
                messages.append(cls.complex_message())
            else:
                messages.append(cls.random_message())
        return messages


class ModelFactory:
    """Provide model names for parameterized testing."""

    CHAT_MODELS = [
        "glm-4-flash",
        "glm-4-plus",
        "glm-4",
    ]

    INVALID_MODELS = [
        "",
        "non-existent-model",
        "gpt-4",       # Wrong provider
        "glm-999",     # Non-existent version
    ]

    @classmethod
    def all_chat_models(cls):
        return cls.CHAT_MODELS

    @classmethod
    def default_model(cls):
        return "glm-4-flash"

    @classmethod
    def invalid_models(cls):
        return cls.INVALID_MODELS


class ParameterFactory:
    """Generate API parameter combinations for boundary testing."""

    @staticmethod
    def temperature_values():
        """Temperature boundary values for testing."""
        return [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]

    @staticmethod
    def temperature_invalid_values():
        """Invalid temperature values for negative testing."""
        return [-1.0, -0.1, 2.1, 100.0]

    @staticmethod
    def max_tokens_values():
        """Various max_tokens values for testing."""
        return [1, 10, 100, 512, 1024, 2048, 4096]

    @staticmethod
    def top_p_values():
        """Top-p sampling values for testing."""
        return [0.0, 0.1, 0.5, 0.9, 1.0]
