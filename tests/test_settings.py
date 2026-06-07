from audio_diarized_transcription_runner.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LIVE_FINAL_ASR_MODEL,
    RunnerSettings,
)


def test_default_config_is_packaged_with_runner():
    assert DEFAULT_CONFIG_PATH.exists()
    assert "diarizer:" in DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def test_default_settings_are_standalone():
    settings = RunnerSettings()

    assert settings.capability_name == "audio:diarized-transcription@v0"
    assert settings.nemo_config_path == DEFAULT_CONFIG_PATH
    assert settings.live_final_asr_model == DEFAULT_LIVE_FINAL_ASR_MODEL
    assert "livepeer-roboflow" not in str(settings.nemo_config_path)


def test_live_final_asr_model_can_be_overridden():
    settings = RunnerSettings(live_final_asr_model="stt_en_conformer_ctc_large")

    assert settings.default_asr_model == "stt_en_conformer_ctc_large"
    assert settings.live_final_asr_model == "stt_en_conformer_ctc_large"
