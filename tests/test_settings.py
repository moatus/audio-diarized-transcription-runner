from audio_diarized_transcription_runner.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LIVE_FINAL_ASR_MODEL,
    DEFAULT_LIVE_PROVISIONAL_ASR_MODEL,
    DEFAULT_TRUE_STREAMING_ASR_MODEL,
    DEFAULT_TRUE_STREAMING_DIAR_MODEL,
    RunnerSettings,
)


def test_default_config_is_packaged_with_runner():
    assert DEFAULT_CONFIG_PATH.exists()
    assert "diarizer:" in DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def test_default_settings_are_standalone():
    settings = RunnerSettings()

    assert settings.capability_name == "audio:diarized-transcription@v0"
    assert settings.nemo_config_path == DEFAULT_CONFIG_PATH
    assert settings.live_provisional_asr_model == DEFAULT_LIVE_PROVISIONAL_ASR_MODEL
    assert settings.live_provisional_asr_model == "stt_en_conformer_ctc_large"
    assert settings.live_final_asr_model == DEFAULT_LIVE_FINAL_ASR_MODEL
    assert settings.live_final_asr_model == "nvidia/parakeet-tdt-0.6b-v3"
    assert settings.live_final_asr_batch_size == 4
    assert settings.live_final_asr_decoding_strategy == ""
    assert settings.live_final_asr_beam_size is None
    assert settings.true_streaming_asr_model == DEFAULT_TRUE_STREAMING_ASR_MODEL
    assert settings.true_streaming_asr_model == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert settings.true_streaming_diar_model == DEFAULT_TRUE_STREAMING_DIAR_MODEL
    assert settings.true_streaming_diar_model == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert settings.true_streaming_asr_greedy_max_symbols == 16
    assert settings.true_streaming_asr_stabilize_partials is True
    assert settings.true_streaming_asr_emit_min_words == 6
    assert settings.true_streaming_asr_emit_min_chars == 28
    assert settings.true_streaming_asr_emit_max_hold_seconds == 1.2
    assert "livepeer-roboflow" not in str(settings.nemo_config_path)


def test_live_final_asr_model_can_be_overridden():
    settings = RunnerSettings(
        live_provisional_asr_model="stt_en_fastconformer_ctc_large",
        live_final_asr_model="stt_en_conformer_ctc_large",
    )

    assert settings.default_asr_model == "stt_en_conformer_ctc_large"
    assert settings.live_provisional_asr_model == "stt_en_fastconformer_ctc_large"
    assert settings.live_final_asr_model == "stt_en_conformer_ctc_large"
