import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from client import Qwen3Client


class OpenAIClientTest(unittest.TestCase):
    def test_openai_provider_uses_http_backend(self):
        captured = {}

        class DummyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "42"}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return DummyResponse()

        with patch.object(httpx, "post", side_effect=fake_post):
            client = Qwen3Client(
                model="test-model",
                provider="openai",
                api_key="abc123",
                base_url="https://example.com/v1",
            )

            response = client.ask("hello", effort="low", max_tokens=128)

        self.assertEqual(response.content, "42")
        self.assertEqual(response.model, "test-model")
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["headers"]["Authorization"], "Bearer abc123")
        self.assertEqual(captured["json"]["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
