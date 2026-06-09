from fastapi.testclient import TestClient

from audio_diarized_transcription_runner import app as app_module
from audio_diarized_transcription_runner.settings import RunnerSettings


def _fake_diarized_result():
    return {
        "id": "dtx_local_test",
        "status": "success",
        "capability": "openai:audio-transcriptions",
        "mode": "local-direct",
        "models": {
            "vad": "vad",
            "speaker_embeddings": "speaker",
            "asr": "asr",
        },
        "duration_seconds": 1.5,
        "text": "speaker_0: hello world",
        "speaker_count": 1,
        "speakers": [{"id": "speaker_0", "talk_seconds": 1.5}],
        "segments": [
            {"speaker": "speaker_0", "start": 0.0, "end": 1.5, "text": "hello world"}
        ],
        "words": [
            {"speaker": "speaker_0", "start": 0.0, "end": 0.4, "word": "hello"},
            {"speaker": "speaker_0", "start": 0.5, "end": 1.0, "word": "world"},
        ],
        "artifacts": {},
        "usage": {"audio_seconds": 2, "work_units": 2},
    }


def test_openai_audio_transcriptions_json_is_basic_openai_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "settings", RunnerSettings(work_dir=tmp_path, device="cpu"))
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )
    monkeypatch.setattr(
        app_module,
        "run_diarized_transcription",
        lambda request, settings: _fake_diarized_result(),
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "nemo-diarized-transcription-meeting-v0"},
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert response.headers["x-livepeer-work-units"] == "2"


def test_openai_audio_transcriptions_verbose_json_is_openai_shape_by_default(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(app_module, "settings", RunnerSettings(work_dir=tmp_path, device="cpu"))
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )
    monkeypatch.setattr(
        app_module,
        "run_diarized_transcription",
        lambda request, settings: _fake_diarized_result(),
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data=[
                ("model", "nemo-diarized-transcription-meeting-v0"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
            ],
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["text"] == "hello world"
    assert body["segments"][0]["text"] == "hello world"
    assert "speaker" not in body["segments"][0]
    assert "diarization" not in body
    assert "x_livepeer" not in body


def test_openai_audio_transcriptions_verbose_json_exposes_additive_diarization(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(app_module, "settings", RunnerSettings(work_dir=tmp_path, device="cpu"))
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )
    monkeypatch.setattr(
        app_module,
        "run_diarized_transcription",
        lambda request, settings: _fake_diarized_result(),
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data=[
                ("model", "nemo-diarized-transcription-meeting-v0"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
                ("diarization", "true"),
            ],
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["text"] == "hello world"
    assert body["segments"][0]["text"] == "hello world"
    assert body["segments"][0]["speaker"] == "speaker_0"
    assert body["words"][0] == {
        "word": "hello",
        "start": 0.0,
        "end": 0.4,
        "speaker": "speaker_0",
    }
    assert body["speaker_labeled_text"] == "speaker_0: hello world"
    assert body["usage"]["work_units"] == 2
    assert body["diarization"]["speaker_count"] == 1
    assert "x_livepeer" not in body
    assert "x_livepeer_diarization" not in body
