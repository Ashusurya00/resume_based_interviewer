from types import SimpleNamespace

from backend.app.services.gemini_client import GeminiClient


def test_gemini_client_returns_openai_compatible_response(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text='{"question": "Dynamic question"}')

    class FakeGeminiSdkClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(
        "backend.app.services.gemini_client.genai.Client", FakeGeminiSdkClient
    )

    client = GeminiClient("test-key", "gemini-test-model")
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Return JSON"}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gemini-test-model"
    assert captured["contents"] == "Return JSON"
    assert captured["config"]["response_mime_type"] == "application/json"
    assert response.choices[0].message.content == '{"question": "Dynamic question"}'
