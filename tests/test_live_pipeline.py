from pathlib import Path

import pytest

from audio_diarized_transcription_runner.live import (
    LiveAudioIngestRequest,
    LiveDiarizationSessionManager,
    LiveSessionConfig,
    _energy_vad,
    _normalize_online_diarization,
)
from audio_diarized_transcription_runner.pipeline import UnsupportedParameterError
from audio_diarized_transcription_runner.settings import RunnerSettings


class FakeOnlineDiarizer:
    def __init__(self, _cfg):
        self.calls = []
        self.frame_index = 0
        self.frame_start = 0.0
        self.buffer_start = 0.0
        self.buffer_end = 0.0
        self.total_buffer_in_secs = 0.0

    def diarize_step(self, audio_buffer, vad_timestamps):
        vad_pairs = _as_pairs(vad_timestamps)
        self.calls.append(
            {
                "audio_len": len(audio_buffer),
                "vad_timestamps": vad_pairs,
                "frame_index": self.frame_index,
                "frame_start": self.frame_start,
                "buffer_start": self.buffer_start,
                "buffer_end": self.buffer_end,
            }
        )
        return [
            f"{segment[0]} {segment[1]} speaker_{index % 2}"
            for index, segment in enumerate(vad_pairs)
        ]


class FailingSecondStepDiarizer(FakeOnlineDiarizer):
    def diarize_step(self, audio_buffer, vad_timestamps):
        if len(self.calls) == 1:
            raise RuntimeError("diarizer failed")
        return super().diarize_step(audio_buffer, vad_timestamps)


def _as_pairs(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[round(float(segment[0]), 3), round(float(segment[1]), 3)] for segment in value]


def test_energy_vad_merges_speech_frames():
    sample_rate = 16000
    samples = [0.0] * 1600 + [0.08] * 6400 + [0.0] * 1600

    vad = _energy_vad(
        samples=samples,
        sample_rate=sample_rate,
        base_offset=10.0,
        threshold=0.01,
        min_speech_duration=0.1,
        min_silence_duration=0.12,
    )

    assert vad == [[10.09, 10.51]]


def test_live_session_ingests_chunks_and_keeps_online_state(monkeypatch, tmp_path):
    created = []

    def factory(cfg):
        diarizer = FakeOnlineDiarizer(cfg)
        created.append(diarizer)
        return diarizer

    def fake_decode(input_path, normalized_path):
        normalized_path.write_bytes(b"normalized")
        return [0.05] * 16000, 1.0

    monkeypatch.setattr("audio_diarized_transcription_runner.live._decode_audio_to_samples", fake_decode)
    settings = RunnerSettings(work_dir=tmp_path)
    manager = LiveDiarizationSessionManager(settings=settings, diarizer_factory=factory)

    started = manager.create_session(
        LiveSessionConfig(
            session_id="live_test",
            vad_strategy="provided",
            rolling_window_seconds=2.0,
        )
    )
    first = manager.ingest_audio(
        "live_test",
        LiveAudioIngestRequest(
            audio_path=tmp_path / "one.wav",
            filename="one.wav",
            sequence_index=0,
            vad_segments=[{"start": 0.1, "end": 0.8}],
        ),
    )
    second = manager.ingest_audio(
        "live_test",
        LiveAudioIngestRequest(
            audio_path=tmp_path / "two.wav",
            filename="two.wav",
            sequence_index=1,
            vad_segments=[{"start": 0.2, "end": 0.9}],
        ),
    )
    finished = manager.finish_session("live_test")

    assert started["event_type"] == "session.started"
    assert first["chunk"]["start"] == 0.0
    assert first["chunk"]["end"] == 1.0
    assert first["segments"] == [
        {"speaker": "speaker_0", "start": 0.1, "end": 0.8, "text": ""}
    ]
    assert second["segments"] == [
        {"speaker": "speaker_0", "start": 0.1, "end": 0.8, "text": ""},
        {"speaker": "speaker_1", "start": 1.2, "end": 1.9, "text": ""},
    ]
    assert created[0].calls[1]["frame_index"] == 1
    assert created[0].calls[1]["frame_start"] == 1.0
    assert created[0].calls[1]["buffer_start"] == 0.0
    assert finished["status"] == "closed"
    assert Path(finished["final_audio_path"]).exists()


def test_live_session_id_is_contained_and_rejects_path_traversal(tmp_path):
    manager = LiveDiarizationSessionManager(
        settings=RunnerSettings(work_dir=tmp_path),
        diarizer_factory=FakeOnlineDiarizer,
    )

    manager.create_session(LiveSessionConfig(session_id="live.test-01_ok"))
    session = manager.get_session("live.test-01_ok")

    assert session.session_dir.resolve().is_relative_to(
        (tmp_path / "live_sessions").resolve()
    )
    assert session.session_dir.name == "live.test-01_ok"

    with pytest.raises(UnsupportedParameterError) as error:
        manager.create_session(LiveSessionConfig(session_id="../escape"))

    assert "session_id must be 1-128 characters" in str(error.value)
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "live_sessions" / ".." / "escape").resolve().exists()


def test_live_session_energy_vad_avoids_exact_frame_boundary(monkeypatch, tmp_path):
    created = []

    def factory(cfg):
        diarizer = FakeOnlineDiarizer(cfg)
        created.append(diarizer)
        return diarizer

    def fake_decode(input_path, normalized_path):
        normalized_path.write_bytes(b"normalized")
        return [0.05] * 16000, 1.0

    monkeypatch.setattr("audio_diarized_transcription_runner.live._decode_audio_to_samples", fake_decode)
    settings = RunnerSettings(work_dir=tmp_path)
    manager = LiveDiarizationSessionManager(settings=settings, diarizer_factory=factory)

    manager.create_session(
        LiveSessionConfig(
            session_id="energy_test",
            vad_strategy="energy",
            energy_threshold=0.01,
            min_speech_duration_seconds=0.03,
            min_silence_duration_seconds=0.15,
        )
    )
    first = manager.ingest_audio(
        "energy_test",
        LiveAudioIngestRequest(
            audio_path=tmp_path / "one.wav",
            filename="one.wav",
            sequence_index=0,
        ),
    )
    second = manager.ingest_audio(
        "energy_test",
        LiveAudioIngestRequest(
            audio_path=tmp_path / "two.wav",
            filename="two.wav",
            sequence_index=1,
        ),
    )

    assert first["chunk"]["vad_segments"] == [{"start": 0.0, "end": 1.0}]
    assert second["vad_segments"] == [{"start": 0.0, "end": 2.0}]
    assert created[0].calls[0]["vad_timestamps"] == [[0.0, 0.999]]
    assert created[0].calls[1]["vad_timestamps"] == [[0.0, 1.999]]


def test_live_session_rolls_back_state_when_diarizer_fails(monkeypatch, tmp_path):
    def fake_decode(input_path, normalized_path):
        normalized_path.write_bytes(b"normalized")
        return [0.05] * 16000, 1.0

    monkeypatch.setattr("audio_diarized_transcription_runner.live._decode_audio_to_samples", fake_decode)
    settings = RunnerSettings(work_dir=tmp_path)
    manager = LiveDiarizationSessionManager(
        settings=settings,
        diarizer_factory=lambda cfg: FailingSecondStepDiarizer(cfg),
    )

    manager.create_session(
        LiveSessionConfig(
            session_id="rollback_test",
            vad_strategy="energy",
            energy_threshold=0.01,
            min_speech_duration_seconds=0.03,
            min_silence_duration_seconds=0.15,
        )
    )
    manager.ingest_audio(
        "rollback_test",
        LiveAudioIngestRequest(
            audio_path=tmp_path / "one.wav",
            filename="one.wav",
            sequence_index=0,
        ),
    )

    try:
        manager.ingest_audio(
            "rollback_test",
            LiveAudioIngestRequest(
                audio_path=tmp_path / "two.wav",
                filename="two.wav",
                sequence_index=1,
            ),
        )
    except RuntimeError as error:
        assert str(error) == "diarizer failed"
    else:
        raise AssertionError("expected diarizer failure")

    snapshot = manager.get_session("rollback_test").snapshot()
    assert snapshot["duration_seconds"] == 1.0
    assert snapshot["chunk_count"] == 1
    assert snapshot["vad_segments"] == [{"start": 0.0, "end": 1.0}]
    assert snapshot["segments"] == [
        {"speaker": "speaker_0", "start": 0.0, "end": 0.999, "text": ""}
    ]


def test_live_session_final_transcription_uses_live_final_asr_model(monkeypatch, tmp_path):
    seen = {}

    def fake_decode(input_path, normalized_path):
        normalized_path.write_bytes(b"normalized")
        return [0.05] * 16000, 1.0

    def fake_run(request, settings):
        seen["asr"] = settings.default_asr_model
        seen["audio_path"] = request.audio_path
        return {
            "text": "speaker_0: hello",
            "segments": [],
            "words": [],
            "models": {"asr": settings.default_asr_model},
            "duration_seconds": 1.0,
            "usage": {"audio_seconds": 1, "work_units": 1},
        }

    monkeypatch.setattr("audio_diarized_transcription_runner.live._decode_audio_to_samples", fake_decode)
    monkeypatch.setattr(
        "audio_diarized_transcription_runner.live.run_diarized_transcription",
        fake_run,
    )
    settings = RunnerSettings(
        work_dir=tmp_path,
        default_asr_model="stt_en_conformer_ctc_large",
        live_final_asr_model="stt_en_fastconformer_ctc_large",
    )
    manager = LiveDiarizationSessionManager(
        settings=settings,
        diarizer_factory=FakeOnlineDiarizer,
    )

    manager.create_session(LiveSessionConfig(session_id="final_asr_test"))
    manager.ingest_audio(
        "final_asr_test",
        LiveAudioIngestRequest(
            audio_path=tmp_path / "one.wav",
            filename="one.wav",
            sequence_index=0,
        ),
    )
    finished = manager.finish_session("final_asr_test", run_final_transcription_path=True)

    assert seen["asr"] == "stt_en_fastconformer_ctc_large"
    assert seen["audio_path"].name == "session.wav"
    assert finished["final_transcription"]["models"]["asr"] == "stt_en_fastconformer_ctc_large"


def test_normalize_online_diarization_accepts_rttm_like_lines_and_dicts():
    assert _normalize_online_diarization(
        [
            "0.00 1.25 speaker_0",
            {"start_time": 1.25, "end_time": 2.5, "speaker": "speaker_1"},
        ]
    ) == [
        {"speaker": "speaker_0", "start": 0.0, "end": 1.25, "text": ""},
        {"speaker": "speaker_1", "start": 1.25, "end": 2.5, "text": ""},
    ]
