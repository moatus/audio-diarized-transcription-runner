from fastapi.testclient import TestClient

from audio_diarized_transcription_runner import app as app_module


def test_options_advertises_two_public_transcription_routes():
    with TestClient(app_module.app) as client:
        response = client.get("/openai:audio-transcriptions/options")

    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "openai:audio-transcriptions"
    assert body["endpoints"]["bounded_transcriptions"] == "POST /v1/audio/transcriptions"
    assert body["endpoints"]["openai_compatible"] == "POST /v1/audio/transcriptions"
    assert body["endpoints"]["true_streaming"] == "WS /v1/audio/transcriptions/stream"
    assert "native" not in body["endpoints"]
    assert "legacy_true_streaming" not in body["endpoints"]
    assert body["openai_compatible"] == {
        "endpoint": "POST /v1/audio/transcriptions",
        "capability": "openai:audio-transcriptions",
        "additive": True,
        "response_formats": ["json", "verbose_json", "text", "srt", "vtt"],
        "timestamp_granularities": ["segment", "word"],
        "feature_flags": ["diarization", "include_diarization"],
        "extension_fields": [
            "diarization",
            "speaker_labeled_text",
            "usage",
            "models",
            "artifacts",
            "speaker",
        ],
    }
    assert body["true_streaming"]["endpoint"] == "WS /v1/audio/transcriptions/stream"
    assert "legacy_endpoint" not in body["true_streaming"]


def test_removed_bounded_native_route_is_not_registered():
    http_paths = {
        route.path
        for route in app_module.app.routes
        if "route" in route.__class__.__name__.lower()
    }
    assert "/v1/audio/transcriptions" in http_paths
    assert "/v1/audio/diarized-transcriptions" not in http_paths
