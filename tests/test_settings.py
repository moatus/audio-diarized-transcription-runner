from audio_diarized_transcription_runner.settings import DEFAULT_CONFIG_PATH, RunnerSettings


def test_default_config_is_packaged_with_runner():
    assert DEFAULT_CONFIG_PATH.exists()
    assert "diarizer:" in DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def test_default_settings_are_standalone():
    settings = RunnerSettings()

    assert settings.capability_name == "audio:diarized-transcription@v0"
    assert settings.nemo_config_path == DEFAULT_CONFIG_PATH
    assert "livepeer-roboflow" not in str(settings.nemo_config_path)

