from audio_diarized_transcription_runner.pipeline import (
    _configure_native_timestamp_asr_decoding,
    _extract_native_timestamp_words,
    _normalize_segments,
    _normalize_words,
    _render_speaker_labeled_text,
    _segments_to_srt,
    _segments_to_vtt,
    _unwrap_transcribe_result,
    _uses_native_parakeet_timestamps,
)
from audio_diarized_transcription_runner.settings import RunnerSettings


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


def test_parakeet_tdt_models_use_native_timestamp_path():
    assert _uses_native_parakeet_timestamps("nvidia/parakeet-tdt-0.6b-v3")
    assert _uses_native_parakeet_timestamps("nvidia/parakeet_tdt_ctc-1.1b")
    assert _uses_native_parakeet_timestamps("nvidia/parakeet-rnnt-1.1b")
    assert not _uses_native_parakeet_timestamps("nvidia/parakeet-ctc-1.1b")
    assert not _uses_native_parakeet_timestamps("stt_en_conformer_ctc_large")


def test_extract_native_parakeet_timestamp_words_from_hypothesis_dict():
    words, timestamps = _extract_native_timestamp_words(
        {
            "text": "Well, I don't",
            "timestamp": {
                "word": [
                    {"word": "Well,", "start": 0.321, "end": 0.559},
                    {"word": "I", "start": 0.64, "end": 0.8},
                    {"word": "don't", "start": 0.8, "end": 1.04},
                ]
            },
        }
    )

    assert words == ["Well,", "I", "don't"]
    assert timestamps == [[0.321, 0.559], [0.64, 0.8], [0.8, 1.04]]


class FakeHypothesis:
    text = "hello world"
    timestep = {
        "word": [
            {"start_time": 0.0, "end_time": 0.25},
            {"start_time": 0.25, "end_time": 0.5},
        ]
    }


def test_extract_native_parakeet_timestamp_words_falls_back_to_hypothesis_text():
    words, timestamps = _extract_native_timestamp_words(FakeHypothesis())

    assert words == ["hello", "world"]
    assert timestamps == [[0.0, 0.25], [0.25, 0.5]]


def test_extract_native_parakeet_timestamp_words_accepts_blank_hypothesis():
    words, timestamps = _extract_native_timestamp_words(
        {"text": "", "timestamp": {"word": [], "segment": [], "char": []}}
    )

    assert words == []
    assert timestamps == []


def test_extract_native_parakeet_timestamp_words_approximates_from_segments():
    words, timestamps = _extract_native_timestamp_words(
        {
            "text": "hello world",
            "timestamp": {
                "word": [],
                "segment": [{"start": 1.0, "end": 3.0}],
            },
        }
    )

    assert words == ["hello", "world"]
    assert timestamps == [[1.0, 2.0], [2.0, 3.0]]


def test_unwrap_transcribe_result_accepts_nemo_tuple_shape():
    assert _unwrap_transcribe_result((["hypothesis"], None)) == ["hypothesis"]
    assert _unwrap_transcribe_result("single") == ["single"]


def test_configure_native_timestamp_asr_decoding_sets_optional_strategy_and_beam():
    from omegaconf import OmegaConf

    class FakeModel:
        def __init__(self):
            self.cfg = OmegaConf.create({"decoding": {"strategy": "greedy", "beam": {}}})
            self.changed = False

        def change_decoding_strategy(self, cfg):
            self.changed = True
            self.changed_cfg = cfg

    model = FakeModel()
    settings = RunnerSettings(
        live_final_asr_decoding_strategy="beam",
        live_final_asr_beam_size=4,
    )

    _configure_native_timestamp_asr_decoding(model, settings)

    assert model.changed is True
    assert model.cfg.decoding.strategy == "beam"
    assert model.cfg.decoding.beam.beam_size == 4
    assert model.cfg.decoding.compute_timestamps is True
    assert model.cfg.decoding.preserve_alignments is True
