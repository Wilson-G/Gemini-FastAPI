import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.client import GeminiClientWrapper
from gemini_webapi.client import ChatSession


async def _empty_async_generator():
    if False:
        yield None


class TestGeminiClientWrapper(unittest.IsolatedAsyncioTestCase):
    async def test_generate_content_stream_uses_compat_file_data(self):
        client = GeminiClientWrapper(client_id="test-client")
        client._batch_execute = AsyncMock()
        client._prepare_file_data = AsyncMock(return_value=[[["upload-url"], "image.png"]])

        chunks = [
            SimpleNamespace(text_delta="A", text="A", metadata=None),
            SimpleNamespace(text_delta="B", text="AB", metadata=None),
        ]

        async def fake_generate(**kwargs):
            self.assertEqual(kwargs["req_file_data"], [[["upload-url"], "image.png"]])
            self.assertFalse(kwargs["temporary"])
            for chunk in chunks:
                yield chunk

        client._generate = fake_generate
        chat = ChatSession(client)

        received = []
        async for chunk in client.generate_content_stream(
            "prompt",
            files=["/tmp/example.png"],
            chat=chat,
            temporary=False,
        ):
            received.append(chunk.text)

        self.assertEqual(received, ["A", "AB"])
        client._prepare_file_data.assert_awaited_once()
        client._batch_execute.assert_awaited()

    async def test_generate_content_stream_closes_bytesio_inputs(self):
        client = GeminiClientWrapper(client_id="test-client")
        client._batch_execute = AsyncMock()
        client._prepare_file_data = AsyncMock(return_value=None)
        client._generate = lambda **kwargs: _empty_async_generator()
        fake_file = io.BytesIO(b"demo")

        async for _ in client.generate_content_stream("prompt", files=[fake_file]):
            pass

        self.assertTrue(fake_file.closed)


if __name__ == "__main__":
    unittest.main()
