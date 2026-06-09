from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from audio_diarized_transcription_runner import app as app_module
from audio_diarized_transcription_runner.pipeline import UnsupportedParameterError
from audio_diarized_transcription_runner.settings import RunnerSettings
from audio_diarized_transcription_runner.streaming import (
    FakeStreamingASREngine,
    FakeStreamingDiarizationEngine,
    StreamingEnginePair,
    TrueStreamingSessionConfig,
    TrueStreamingSessionManager,
    _StreamingTranscriptStabilizer,
    _polish_streaming_text,
    _strip_eou_token_pieces,
    pcm16_sine_wave,
)


def _fake_engines(_settings):
    return StreamingEnginePair(
        asr=FakeStreamingASREngine("fake-asr"),
        diarization=FakeStreamingDiarizationEngine("fake-diar"),
    )


class TokenPieceStreamingASREngine:
    model_name = "fake-token-piece-asr"

    def __init__(self):
        self._frames = 0
        self._pieces = [
            ["▁lead"],
            ["ers"],
            ["hip"],
            ["▁is"],
            ["▁ready"],
        ]

    def transcribe(self, audio, stream_id):
        pieces = self._pieces[min(self._frames, len(self._pieces) - 1)]
        self._frames += 1
        text = "".join(piece.replace("▁", " ") for piece in pieces)
        return {"text": text, "token_pieces": pieces, "is_final": False}

    def reset_state(self, stream_id):
        self._frames = 0


class EOUTokenPieceStreamingASREngine:
    model_name = "fake-parakeet-eou-asr"

    def __init__(self):
        self._frames = 0
        self._pieces = [
            ["▁hello"],
            ["▁from"],
            ["▁parakeet", "<EOU>"],
        ]

    def transcribe(self, audio, stream_id):
        pieces = self._pieces[min(self._frames, len(self._pieces) - 1)]
        self._frames += 1
        text = "".join(
            piece.replace("▁", " ")
            for piece in pieces
            if piece not in {"<EOU>", "<EOB>"}
        )
        return {
            "text": text,
            "token_pieces": [piece for piece in pieces if piece not in {"<EOU>", "<EOB>"}],
            "is_final": "<EOU>" in pieces or "<EOB>" in pieces,
        }

    def reset_state(self, stream_id):
        self._frames = 0


def _token_piece_engines(_settings):
    return StreamingEnginePair(
        asr=TokenPieceStreamingASREngine(),
        diarization=FakeStreamingDiarizationEngine("fake-diar"),
    )


def _eou_token_piece_engines(_settings):
    return StreamingEnginePair(
        asr=EOUTokenPieceStreamingASREngine(),
        diarization=FakeStreamingDiarizationEngine("fake-diar"),
    )


def test_true_streaming_session_emits_incremental_transcript_and_speaker(tmp_path):
    settings = RunnerSettings(
        work_dir=tmp_path,
        true_streaming_engine="fake",
        true_streaming_max_session_seconds=10,
    )
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines)
    session = manager.create_session(TrueStreamingSessionConfig(session_id="stream_test"))

    events = session.process_audio(pcm16_sine_wave(sample_rate=16000, seconds=0.08))
    transcript = [event for event in events if event["event_type"] == "transcript.segment"]
    speakers = [event for event in events if event["event_type"] == "speaker.update"]
    finished = session.finish()

    assert transcript
    assert transcript[0]["text"] == "fake streaming transcript 1"
    assert transcript[0]["is_provisional"] is True
    assert transcript[0]["speaker"] == "speaker_0"
    assert speakers[0]["speaker"] == "speaker_0"
    assert finished["event_type"] == "transcript.session.finished"
    assert Path(finished["audio_path"]).exists()
    assert Path(finished["transcript_jsonl_path"]).exists()


def test_streaming_transcript_stabilizer_coalesces_subword_fragments():
    stabilizer = _StreamingTranscriptStabilizer(
        min_words=6,
        min_chars=28,
        max_hold_seconds=10.0,
    )
    frames = [
        (0.56, 0.64, "A little"),
        (0.8, 0.88, " bit"),
        (0.88, 0.96, " of"),
        (0.96, 1.04, " a"),
        (1.04, 1.12, " question"),
        (1.2, 1.28, " , but"),
        (3.76, 3.84, " lead"),
        (3.84, 3.92, "ers"),
        (3.92, 4.0, "hip"),
        (4.08, 4.16, " is"),
        (4.16, 4.24, " stepped"),
        (4.24, 4.32, " down"),
        (4.32, 4.4, " ."),
    ]

    emitted = [
        segment.text
        for start, end, text in frames
        if (segment := stabilizer.update(text, chunk_start=start, chunk_end=end)) is not None
    ]

    assert emitted == ["A little bit of a question,", "but leadership is stepped down."]


def test_polish_streaming_text_smooths_contractions_and_acronyms():
    assert _polish_streaming_text("the DAO I mean there 's the P N F .") == (
        "the DAO I mean there's the PNF."
    )


def test_streaming_transcript_stabilizer_uses_sentencepiece_boundaries():
    stabilizer = _StreamingTranscriptStabilizer(
        min_words=4,
        min_chars=20,
        max_hold_seconds=10.0,
    )
    frames = [
        (0.0, 0.08, ["▁lead"]),
        (0.08, 0.16, ["ers"]),
        (0.16, 0.24, ["hip"]),
        (0.24, 0.32, ["▁is"]),
        (0.32, 0.4, ["▁clear"]),
        (0.4, 0.48, ["."]),
    ]

    emitted = [
        segment.text
        for start, end, pieces in frames
        if (
            segment := stabilizer.update_token_pieces(
                pieces,
                chunk_start=start,
                chunk_end=end,
            )
        )
        is not None
    ]

    assert emitted == ["leadership is clear."]


def test_streaming_transcript_stabilizer_uses_hold_limit_for_short_phrases():
    stabilizer = _StreamingTranscriptStabilizer(
        min_words=6,
        min_chars=40,
        max_hold_seconds=0.24,
    )

    assert stabilizer.update(" Hello", chunk_start=0.0, chunk_end=0.08) is None
    assert stabilizer.update(" world", chunk_start=0.08, chunk_end=0.16) is None
    segment = stabilizer.update(" again", chunk_start=0.16, chunk_end=0.24)

    assert segment is not None
    assert segment.text == "Hello world"
    assert segment.start == 0.0


def test_strip_eou_token_pieces_marks_final_without_visible_marker():
    visible, is_final = _strip_eou_token_pieces(
        ["▁hello", "▁from", "▁parakeet", "<EOU>"],
        ("<EOU>", "<EOB>"),
    )

    assert visible == ["▁hello", "▁from", "▁parakeet"]
    assert is_final is True


def test_true_streaming_finish_events_flushes_token_piece_partial(tmp_path):
    settings = RunnerSettings(
        work_dir=tmp_path,
        true_streaming_engine="nemo",
        true_streaming_asr_stabilize_partials=True,
        true_streaming_asr_emit_min_words=10,
        true_streaming_asr_emit_min_chars=100,
        true_streaming_asr_emit_max_hold_seconds=10.0,
        true_streaming_max_session_seconds=10,
    )
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_token_piece_engines)
    session = manager.create_session(TrueStreamingSessionConfig(session_id="stream_token_flush"))

    for _ in range(5):
        events = session.process_audio(pcm16_sine_wave(sample_rate=16000, seconds=0.08))
        assert not [event for event in events if event["event_type"] == "transcript.segment"]

    finish_events = session.finish_events()
    transcript = [event for event in finish_events if event["event_type"] == "transcript.segment"]

    assert [event["event_type"] for event in finish_events] == [
        "transcript.segment",
        "transcript.session.finished",
    ]
    assert transcript[0]["text"] == "leadership is ready"
    assert transcript[0]["is_final"] is True
    assert transcript[0]["status"] == "closed"


def test_true_streaming_eou_frame_flushes_stabilized_transcript(tmp_path):
    settings = RunnerSettings(
        work_dir=tmp_path,
        true_streaming_engine="nemo",
        true_streaming_asr_stabilize_partials=True,
        true_streaming_asr_emit_min_words=10,
        true_streaming_asr_emit_min_chars=100,
        true_streaming_asr_emit_max_hold_seconds=10.0,
        true_streaming_max_session_seconds=10,
    )
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_eou_token_piece_engines)
    session = manager.create_session(TrueStreamingSessionConfig(session_id="stream_eou_flush"))

    transcript_events = []
    for _ in range(3):
        events = session.process_audio(pcm16_sine_wave(sample_rate=16000, seconds=0.08))
        transcript_events.extend(
            event for event in events if event["event_type"] == "transcript.segment"
        )

    assert len(transcript_events) == 1
    assert transcript_events[0]["text"] == "hello from parakeet"
    assert transcript_events[0]["is_final"] is True
    assert transcript_events[0]["text_status"] == "available"
    assert "<EOU>" not in transcript_events[0]["text"]

    session.finish()


def test_true_streaming_manager_rejects_duplicate_active_session_id(tmp_path):
    settings = RunnerSettings(
        work_dir=tmp_path,
        true_streaming_engine="fake",
        true_streaming_max_session_seconds=10,
    )
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines)
    first = manager.create_session(TrueStreamingSessionConfig(session_id="stream_test"))

    with pytest.raises(UnsupportedParameterError) as error:
        manager.create_session(TrueStreamingSessionConfig(session_id="stream_test"))

    assert "true streaming session stream_test already active" in str(error.value)
    first_finished = first.finish()
    assert manager.release_session(first) is False

    second = manager.create_session(TrueStreamingSessionConfig(session_id="stream_test"))
    assert manager.release_session(first) is False
    second_finished = second.finish()
    assert manager.release_session(second) is False
    assert first_finished["transcript_jsonl_path"] != second_finished["transcript_jsonl_path"]
    assert Path(first_finished["transcript_jsonl_path"]).exists()
    assert Path(second_finished["transcript_jsonl_path"]).exists()


def test_true_streaming_session_id_is_contained_and_rejects_path_traversal(tmp_path):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake")
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines)

    manager.create_session(TrueStreamingSessionConfig(session_id="stream.test-01_ok")).finish()
    with pytest.raises(UnsupportedParameterError) as error:
        manager.create_session(TrueStreamingSessionConfig(session_id="../escape"))

    assert "session_id must be 1-128 characters" in str(error.value)


def test_true_streaming_websocket_keeps_one_transport_and_emits_before_finish(monkeypatch, tmp_path):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake", device="cpu")
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(
        app_module,
        "true_streaming_sessions",
        TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines),
    )
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )

    with TestClient(app_module.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/stream?session_id=ws_test"
        ) as websocket:
            started = websocket.receive_json()
            assert started["event_type"] == "session.snapshot"

            websocket.send_bytes(pcm16_sine_wave(sample_rate=16000, seconds=0.08))
            received = [websocket.receive_json() for _ in range(3)]

            assert [event["event_type"] for event in received] == [
                "audio.frame.received",
                "speaker.update",
                "transcript.segment",
            ]
            assert received[2]["text"] == "fake streaming transcript 1"
            assert received[2]["is_provisional"] is True

            websocket.send_json({"type": "finish"})
            finished = websocket.receive_json()
            assert finished["event_type"] == "transcript.session.finished"


def test_true_streaming_legacy_websocket_route_remains_compatible(monkeypatch, tmp_path):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake", device="cpu")
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(
        app_module,
        "true_streaming_sessions",
        TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines),
    )
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )

    with TestClient(app_module.app) as client:
        with client.websocket_connect(
            "/v1/audio/diarized-transcriptions/stream?session_id=ws_legacy"
        ) as websocket:
            started = websocket.receive_json()
            assert started["event_type"] == "session.snapshot"
            websocket.send_json({"type": "finish"})
            finished = websocket.receive_json()
            assert finished["event_type"] == "transcript.session.finished"


def test_true_streaming_websocket_rejects_duplicate_active_session_id(monkeypatch, tmp_path):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake", device="cpu")
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(
        app_module,
        "true_streaming_sessions",
        TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines),
    )
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )

    with TestClient(app_module.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/stream?session_id=ws_duplicate"
        ) as first:
            started = first.receive_json()
            assert started["event_type"] == "session.snapshot"

            with client.websocket_connect(
                "/v1/audio/transcriptions/stream?session_id=ws_duplicate"
            ) as second:
                rejected = second.receive_json()
                with pytest.raises(WebSocketDisconnect) as closed:
                    second.receive_json()

            assert rejected["error"]["type"] == "invalid_request_error"
            assert "already active" in rejected["error"]["message"]
            assert closed.value.code == 1008

            first.send_json({"type": "finish"})
            finished = first.receive_json()
            assert finished["event_type"] == "transcript.session.finished"


@pytest.mark.parametrize("query", ["max_speakers=abc", "sample_rate=abc"])
def test_true_streaming_websocket_rejects_malformed_numeric_query_param(
    monkeypatch,
    tmp_path,
    query,
):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake", device="cpu")
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(
        app_module,
        "true_streaming_sessions",
        TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines),
    )
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )

    with TestClient(app_module.app) as client:
        with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?{query}"
        ) as websocket:
            rejected = websocket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()

    assert rejected["error"]["type"] == "invalid_request_error"
    assert "must be an integer" in rejected["error"]["message"]
    assert closed.value.code == 1008


def test_true_streaming_websocket_releases_session_id_after_disconnect(monkeypatch, tmp_path):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake", device="cpu")
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines)
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(app_module, "true_streaming_sessions", manager)
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )

    with TestClient(app_module.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/stream?session_id=ws_cleanup"
        ) as websocket:
            started = websocket.receive_json()
            assert started["event_type"] == "session.snapshot"

        with client.websocket_connect(
            "/v1/audio/transcriptions/stream?session_id=ws_cleanup"
        ) as websocket:
            restarted = websocket.receive_json()
            assert restarted["event_type"] == "session.snapshot"
            websocket.send_json({"type": "finish"})
            finished = websocket.receive_json()
            assert finished["event_type"] == "transcript.session.finished"


def test_true_streaming_websocket_releases_session_id_after_input_error(monkeypatch, tmp_path):
    settings = RunnerSettings(work_dir=tmp_path, true_streaming_engine="fake", device="cpu")
    manager = TrueStreamingSessionManager(settings=settings, engine_factory=_fake_engines)
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(app_module, "true_streaming_sessions", manager)
    monkeypatch.setattr(
        app_module,
        "assert_runtime_ready",
        lambda _settings: {"cuda_available": False, "torch_version": None},
    )

    with TestClient(app_module.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/stream?session_id=ws_input_error"
        ) as websocket:
            started = websocket.receive_json()
            assert started["event_type"] == "session.snapshot"
            websocket.send_bytes(b"\x00")
            rejected = websocket.receive_json()
            assert rejected["error"]["type"] == "invalid_request_error"
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1003

        with client.websocket_connect(
            "/v1/audio/transcriptions/stream?session_id=ws_input_error"
        ) as websocket:
            restarted = websocket.receive_json()
            assert restarted["event_type"] == "session.snapshot"
            websocket.send_json({"type": "finish"})
            finished = websocket.receive_json()
            assert finished["event_type"] == "transcript.session.finished"
