"""
SSE (Server-Sent Events) Stream Parser.

Parses streaming responses from Zhipu's /chat/completions endpoint
when stream=True. Validates stream integrity and extracts tokens.
"""
import json
import time
import logging

logger = logging.getLogger(__name__)


class SSEEvent:
    """Represents a single SSE event."""

    def __init__(self, data, event_type="message", event_id=None):
        self.data = data
        self.event_type = event_type
        self.event_id = event_id
        self.timestamp = time.time()

    @property
    def is_done(self):
        return self.data == "[DONE]"

    @property
    def json_data(self):
        if self.is_done:
            return None
        try:
            return json.loads(self.data)
        except (json.JSONDecodeError, TypeError):
            return None

    @property
    def content(self):
        """Extract delta content from the event."""
        data = self.json_data
        if not data:
            return ""
        try:
            return data["choices"][0]["delta"].get("content", "")
        except (KeyError, IndexError):
            return ""

    def __repr__(self):
        return f"SSEEvent(data={self.data[:50]}...)" if len(str(self.data)) > 50 else f"SSEEvent(data={self.data})"


class SSEStreamResult:
    """Aggregated result from parsing a complete SSE stream."""

    def __init__(self):
        self.events = []
        self.full_content = ""
        self.first_token_time = None
        self.last_token_time = None
        self.start_time = None
        self.has_done_signal = False
        self.errors = []

    @property
    def token_count(self):
        """Number of non-empty content events."""
        return sum(1 for e in self.events if e.content)

    @property
    def ttft_ms(self):
        """Time To First Token in milliseconds."""
        if self.start_time and self.first_token_time:
            return (self.first_token_time - self.start_time) * 1000
        return None

    @property
    def total_time_ms(self):
        """Total stream duration in milliseconds."""
        if self.start_time and self.last_token_time:
            return (self.last_token_time - self.start_time) * 1000
        return None

    @property
    def is_complete(self):
        """Whether the stream ended properly with [DONE]."""
        return self.has_done_signal and len(self.errors) == 0

    def summary(self):
        """Return a dict summary for reporting."""
        return {
            "token_count": self.token_count,
            "full_content_length": len(self.full_content),
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms else None,
            "total_time_ms": round(self.total_time_ms, 1) if self.total_time_ms else None,
            "has_done_signal": self.has_done_signal,
            "errors": self.errors,
            "is_complete": self.is_complete,
        }


def parse_sse_stream(response):
    """
    Parse an SSE stream from a requests.Response object.

    Args:
        response: requests.Response with stream=True

    Returns:
        SSEStreamResult with all parsed events and metrics.
    """
    result = SSEStreamResult()
    result.start_time = time.time()

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            # SSE format: "data: {json}" or "data: [DONE]"
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()

                event = SSEEvent(data=data_str)
                result.events.append(event)

                if event.is_done:
                    result.has_done_signal = True
                    result.last_token_time = time.time()
                    logger.debug("Stream completed with [DONE]")
                    break

                content = event.content
                if content:
                    result.full_content += content
                    now = time.time()
                    if result.first_token_time is None:
                        result.first_token_time = now
                    result.last_token_time = now

            elif line.startswith("event:"):
                # Some SSE implementations include event type
                logger.debug(f"SSE event type: {line}")

            elif line.startswith(":"):
                # SSE comment (keep-alive)
                logger.debug(f"SSE comment: {line}")

    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"SSE stream parse error: {e}")

    logger.info(f"SSE stream parsed: {result.summary()}")
    return result
