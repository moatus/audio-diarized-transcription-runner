from audio_diarized_transcription_runner.pipeline import (
    _normalize_segments,
    _normalize_words,
    _render_speaker_labeled_text,
    _segments_to_srt,
    _segments_to_vtt,
)


def test_runner_normalizes_nemo_session_shapes():
    segments = _normalize_segments(
        [
            {
                "speaker": "speaker_0",
                "start_time": 0.12,
                "end_time": 1.34,
                "sentence": "hello there",
            }
        ]
    )
    words = _normalize_words(
        [
            {
                "speaker": "speaker_0",
                "start_time": 0.12,
                "end_time": 0.5,
                "word": "hello",
            }
        ]
    )

    assert segments == [
        {"speaker": "speaker_0", "start": 0.12, "end": 1.34, "text": "hello there"}
    ]
    assert words == [{"speaker": "speaker_0", "start": 0.12, "end": 0.5, "word": "hello"}]
    assert _render_speaker_labeled_text(segments) == "speaker_0: hello there"
    assert "00:00:00,120 --> 00:00:01,340" in _segments_to_srt(segments)
    assert _segments_to_vtt(segments).startswith("WEBVTT")

