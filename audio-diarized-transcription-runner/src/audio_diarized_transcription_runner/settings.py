"""Environment-backed settings for the diarized transcription runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CAPABILITY_NAME = "audio:diarized-transcription@v0"
DEFAULT_MODEL_NAME = "nemo-diarized-transcription-meeting-v0"
DEFAULT_LIVE_PROVISIONAL_ASR_MODEL = "stt_en_conformer_ctc_large"
DEFAULT_LIVE_FINAL_ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_TRUE_STREAMING_ASR_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_TRUE_STREAMING_PARAKEET_ASR_MODEL = "nvidia/parakeet_realtime_eou_120m-v1"
DEFAULT_TRUE_STREAMING_DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
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


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _int_list_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _str_list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


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
    live_provisional_asr_model: str = os.getenv(
        "LIVE_PROVISIONAL_ASR_MODEL",
        DEFAULT_LIVE_PROVISIONAL_ASR_MODEL,
    )
    live_final_asr_model: str = os.getenv(
        "LIVE_FINAL_ASR_MODEL",
        DEFAULT_LIVE_FINAL_ASR_MODEL,
    )
    live_final_asr_batch_size: int = _int_env("LIVE_FINAL_ASR_BATCH_SIZE", 4)
    live_final_asr_decoding_strategy: str = os.getenv("LIVE_FINAL_ASR_DECODING_STRATEGY", "")
    live_final_asr_beam_size: int | None = _optional_int_env("LIVE_FINAL_ASR_BEAM_SIZE")
    default_max_speakers: int = _int_env("DEFAULT_MAX_SPEAKERS", 8)
    default_preset: str = os.getenv("DEFAULT_PRESET", "meeting")
    default_model: str = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL_NAME)
    nemo_config_path: Path = _path_env("NEMO_DIARIZER_CONFIG", DEFAULT_CONFIG_PATH)
    live_sample_rate: int = _int_env("LIVE_SAMPLE_RATE", 16000)
    live_history_buffer_size: int = _int_env("LIVE_HISTORY_BUFFER_SIZE", 256)
    live_current_buffer_size: int = _int_env("LIVE_CURRENT_BUFFER_SIZE", 256)
    live_session_ttl_seconds: int = _int_env("LIVE_SESSION_TTL_SECONDS", 3600)
    live_closed_session_ttl_seconds: int = _int_env("LIVE_CLOSED_SESSION_TTL_SECONDS", 300)
    true_streaming_enabled: bool = _bool_env("TRUE_STREAMING_ENABLED", True)
    true_streaming_engine: str = os.getenv("TRUE_STREAMING_ENGINE", "nemo")
    true_streaming_asr_model: str = os.getenv(
        "TRUE_STREAMING_ASR_MODEL",
        DEFAULT_TRUE_STREAMING_ASR_MODEL,
    )
    true_streaming_diar_model: str = os.getenv(
        "TRUE_STREAMING_DIAR_MODEL",
        DEFAULT_TRUE_STREAMING_DIAR_MODEL,
    )
    true_streaming_sample_rate: int = _int_env("TRUE_STREAMING_SAMPLE_RATE", 16000)
    true_streaming_asr_att_context_size: tuple[int, ...] = _int_list_env(
        "TRUE_STREAMING_ASR_ATT_CONTEXT_SIZE",
        (70, 1),
    )
    true_streaming_asr_chunk_seconds: float = _float_env(
        "TRUE_STREAMING_ASR_CHUNK_SECONDS",
        0.08,
    )
    true_streaming_asr_greedy_max_symbols: int = _int_env(
        "TRUE_STREAMING_ASR_GREEDY_MAX_SYMBOLS",
        16,
    )
    true_streaming_asr_eou_tokens: tuple[str, ...] = _str_list_env(
        "TRUE_STREAMING_ASR_EOU_TOKENS",
        ("<EOU>", "<EOB>"),
    )
    true_streaming_asr_stabilize_partials: bool = _bool_env(
        "TRUE_STREAMING_ASR_STABILIZE_PARTIALS",
        True,
    )
    true_streaming_asr_emit_min_words: int = _int_env(
        "TRUE_STREAMING_ASR_EMIT_MIN_WORDS",
        6,
    )
    true_streaming_asr_emit_min_chars: int = _int_env(
        "TRUE_STREAMING_ASR_EMIT_MIN_CHARS",
        28,
    )
    true_streaming_asr_emit_max_hold_seconds: float = _float_env(
        "TRUE_STREAMING_ASR_EMIT_MAX_HOLD_SECONDS",
        1.2,
    )
    true_streaming_diar_frame_seconds: float = _float_env(
        "TRUE_STREAMING_DIAR_FRAME_SECONDS",
        0.08,
    )
    true_streaming_diar_threshold: float = _float_env("TRUE_STREAMING_DIAR_THRESHOLD", 0.4)
    true_streaming_max_session_seconds: int = _int_env(
        "TRUE_STREAMING_MAX_SESSION_SECONDS",
        3600,
    )


def configure_cache_environment(settings: RunnerSettings) -> None:
    cache_dir = str(settings.model_cache_dir)
    os.environ.setdefault("NEMO_HOME", cache_dir)
    os.environ.setdefault("HF_HOME", str(settings.model_cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(settings.model_cache_dir / "torch"))
