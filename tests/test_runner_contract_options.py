from fastapi.testclient import TestClient

from audio_diarized_transcription_runner import app as app_module


def test_options_advertises_native_and_openai_compatible_routes():
    with TestClient(app_module.app) as client:
        response = client.get("/audio:diarized-transcription@v0/options")

    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "audio:diarized-transcription@v0"
    assert body["endpoints"]["bounded_transcriptions"] == "POST /v1/audio/transcriptions"
    assert body["endpoints"]["native"] == "POST /v1/audio/diarized-transcriptions"
    assert body["endpoints"]["openai_compatible"] == "POST /v1/audio/transcriptions"
    assert body["endpoints"]["true_streaming"] == "WS /v1/audio/transcriptions/stream"
    assert (
        body["endpoints"]["legacy_true_streaming"]
        == "WS /v1/audio/diarized-transcriptions/stream"
    )
    assert body["openai_compatible"] == {
        "endpoint": "POST /v1/audio/transcriptions",
        "native_capability": "audio:diarized-transcription@v0",
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
    assert (
        body["true_streaming"]["legacy_endpoint"]
        == "WS /v1/audio/diarized-transcriptions/stream"
    )
