"""Environment-backed settings for the diarized transcription runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CAPABILITY_NAME = "audio:diarized-transcription@v0"
DEFAULT_MODEL_NAME = "nemo-diarized-transcription-meeting-v0"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diar_infer_meeting.yaml"


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return Path(value)


@dataclass(frozen=True)
class RunnerSettings:
    runner_port: int = _int_env("RUNNER_PORT", 8080)
    device: str = os.getenv("DEVICE", "cuda")
    capability_name: str = os.getenv("CAPABILITY_NAME", DEFAULT_CAPABILITY_NAME)
    metrics_enabled: bool = _bool_env("METRICS_ENABLED", False)
    max_queue_size: int = _int_env("MAX_QUEUE_SIZE", 1)
    max_audio_mb: int = _int_env("MAX_AUDIO_MB", 100)
    model_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", "/models"))
    work_dir: Path = Path(os.getenv("WORK_DIR", "/tmp/audio-diarized-transcription-runner"))
    default_vad_model: str = os.getenv("DEFAULT_VAD_MODEL", "vad_multilingual_marblenet")
    default_speaker_model: str = os.getenv("DEFAULT_SPEAKER_MODEL", "titanet_large")
    default_asr_model: str = os.getenv("DEFAULT_ASR_MODEL", "stt_en_conformer_ctc_large")
    default_max_speakers: int = _int_env("DEFAULT_MAX_SPEAKERS", 8)
    default_preset: str = os.getenv("DEFAULT_PRESET", "meeting")
    default_model: str = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL_NAME)
    nemo_config_path: Path = _path_env("NEMO_DIARIZER_CONFIG", DEFAULT_CONFIG_PATH)


def configure_cache_environment(settings: RunnerSettings) -> None:
    cache_dir = str(settings.model_cache_dir)
    os.environ.setdefault("NEMO_HOME", cache_dir)
    os.environ.setdefault("HF_HOME", str(settings.model_cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(settings.model_cache_dir / "torch"))
