"""NeMo offline diarization-with-ASR pipeline wrapper."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .settings import RunnerSettings, configure_cache_environment


SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "vtt"}


class RunnerInputError(ValueError):
    status_code = 400
    error_type = "invalid_request_error"


class UnsupportedParameterError(RunnerInputError):
    status_code = 422


class CudaOutOfMemoryError(RuntimeError):
    status_code = 507
    error_type = "cuda_out_of_memory"


@dataclass(frozen=True)
class DiarizationRequest:
    audio_path: Path
    filename: str
    content_type: Optional[str]
    model: str
    language: str
    preset: str
    num_speakers: Optional[int]
    max_speakers: int
    response_format: str
    include_words: bool
    include_artifacts: bool


def assert_runtime_ready(settings: RunnerSettings) -> Dict[str, Any]:
    """Fail fast for unsupported CUDA startup and return health device facts."""
    configure_cache_environment(settings)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as error:
        if settings.device == "cuda":
            raise RuntimeError("CUDA requested but PyTorch is not installed") from error
        return {"cuda_available": False, "torch_version": None}

    cuda_available = bool(torch.cuda.is_available())
    if settings.device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but not available inside container")

    return {
        "cuda_available": cuda_available,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def validate_request(request: DiarizationRequest) -> None:
    if request.language != "en":
        raise UnsupportedParameterError("Only language=en is supported in phase 1")
    if request.preset != "meeting":
        raise UnsupportedParameterError("Only preset=meeting is supported in phase 1")
    if request.response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise UnsupportedParameterError(
            f"response_format must be one of {sorted(SUPPORTED_RESPONSE_FORMATS)}"
        )
    if request.num_speakers is not None and request.num_speakers <= 0:
        raise UnsupportedParameterError("num_speakers must be positive when provided")
    if request.max_speakers <= 0:
        raise UnsupportedParameterError("max_speakers must be positive")
    if request.num_speakers is not None and request.num_speakers > request.max_speakers:
        raise UnsupportedParameterError("num_speakers cannot exceed max_speakers")


def run_diarized_transcription(
    request: DiarizationRequest,
    settings: RunnerSettings,
) -> Dict[str, Any]:
    validate_request(request)
    configure_cache_environment(settings)

    request_id = f"dtx_local_{uuid.uuid4().hex[:12]}"
    job_dir = settings.work_dir / "jobs" / request_id
    job_dir.mkdir(parents=True, exist_ok=False)

    normalized_audio = job_dir / f"{_safe_stem(request.filename)}.wav"
    manifest_path = job_dir / "manifest.json"
    duration_seconds = _normalize_audio(request.audio_path, normalized_audio)
    _write_manifest(manifest_path, normalized_audio, request.num_speakers)

    try:
        session = _run_nemo_pipeline(
            manifest_path=manifest_path,
            output_dir=job_dir,
            settings=settings,
            num_speakers=request.num_speakers,
            max_speakers=request.max_speakers,
        )
    except RuntimeError as error:
        if _is_cuda_oom(error):
            raise CudaOutOfMemoryError(str(error)) from error
        raise

    stem = normalized_audio.stem
    artifacts = _collect_artifacts(job_dir, stem)
    segments = _normalize_segments(session.get("sentences", []))
    words = _normalize_words(session.get("words", [])) if request.include_words else []
    text = _render_speaker_labeled_text(segments)
    if not text:
        text = str(session.get("transcription", "")).strip()

    srt_path = job_dir / f"{stem}.srt"
    vtt_path = job_dir / f"{stem}.vtt"
    srt_path.write_text(_segments_to_srt(segments), encoding="utf-8")
    vtt_path.write_text(_segments_to_vtt(segments), encoding="utf-8")
    artifacts["srt_path"] = str(srt_path)
    artifacts["vtt_path"] = str(vtt_path)

    speakers = _speaker_summaries(segments, words)
    work_units = max(1, math.ceil(duration_seconds))
    public_artifacts = (
        artifacts if request.include_artifacts or request.response_format in {"srt", "vtt"} else {}
    )
    response = {
        "id": request_id,
        "status": "success",
        "capability": settings.capability_name,
        "mode": "local-direct",
        "source": {
            "filename": request.filename,
            "content_type": request.content_type,
        },
        "models": {
            "vad": settings.default_vad_model,
            "speaker_embeddings": settings.default_speaker_model,
            "asr": settings.default_asr_model,
        },
        "duration_seconds": duration_seconds,
        "text": text,
        "speaker_count": int(session.get("speaker_count") or len(speakers)),
        "speakers": speakers,
        "segments": segments,
        "words": words,
        "artifacts": public_artifacts,
        "usage": {
            "audio_seconds": work_units,
            "work_units": work_units,
        },
    }
    return response


def _run_nemo_pipeline(
    manifest_path: Path,
    output_dir: Path,
    settings: RunnerSettings,
    num_speakers: Optional[int],
    max_speakers: int,
) -> Dict[str, Any]:
    from omegaconf import OmegaConf
    from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
    from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR

    cfg = OmegaConf.load(settings.nemo_config_path)
    cfg.device = settings.device
    cfg.diarizer.manifest_filepath = str(manifest_path)
    cfg.diarizer.out_dir = str(output_dir)
    cfg.diarizer.vad.model_path = settings.default_vad_model
    cfg.diarizer.speaker_embeddings.model_path = settings.default_speaker_model
    cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False
    cfg.diarizer.asr.model_path = settings.default_asr_model
    cfg.diarizer.clustering.parameters.max_num_speakers = int(max_speakers)
    cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None

    asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()
    word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
    diar_hyp, _ = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(
        diar_hyp, word_hyp, word_ts_hyp
    )
    if not trans_info_dict:
        raise RuntimeError("NeMo produced no diarized transcription output")
    return next(iter(trans_info_dict.values()))


def _safe_stem(filename: str) -> str:
    stem = Path(filename or "audio").stem or "audio"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "audio"


def _normalize_audio(input_path: Path, output_path: Path) -> float:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RunnerInputError("ffmpeg is required in the runner container") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RunnerInputError(f"Could not decode uploaded audio: {detail}") from error
    return _probe_duration(output_path)


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
    except Exception as error:
        raise RunnerInputError(f"Could not determine audio duration for {path}") from error
    if duration <= 0:
        raise RunnerInputError("Uploaded audio duration must be positive")
    return duration


def _write_manifest(path: Path, audio_path: Path, num_speakers: Optional[int]) -> None:
    record = {
        "audio_filepath": str(audio_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _normalize_segments(sentences: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for sentence in sentences:
        text = str(sentence.get("sentence") or sentence.get("text") or "").strip()
        speaker = str(sentence.get("speaker") or "speaker_0")
        start = _round_seconds(sentence.get("start_time", sentence.get("start", 0)))
        end = _round_seconds(sentence.get("end_time", sentence.get("end", start)))
        segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return segments


def _normalize_words(words: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for word in words:
        normalized.append(
            {
                "speaker": str(word.get("speaker") or "speaker_0"),
                "start": _round_seconds(word.get("start_time", word.get("start", 0))),
                "end": _round_seconds(word.get("end_time", word.get("end", 0))),
                "word": str(word.get("word") or ""),
            }
        )
    return normalized


def _round_seconds(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _render_speaker_labeled_text(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(f"{segment['speaker']}: {text}")
    return "\n".join(lines)


def _speaker_summaries(
    segments: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    totals: Dict[str, float] = {}
    basis = segments or words
    for item in basis:
        speaker = str(item.get("speaker") or "speaker_0")
        totals[speaker] = totals.get(speaker, 0.0) + max(
            0.0, float(item.get("end", 0.0)) - float(item.get("start", 0.0))
        )
    return [
        {"id": speaker, "talk_seconds": round(seconds, 3)}
        for speaker, seconds in sorted(totals.items())
    ]


def _collect_artifacts(job_dir: Path, stem: str) -> Dict[str, str]:
    pred_dir = job_dir / "pred_rttms"
    artifact_names = {
        "json_path": pred_dir / f"{stem}.json",
        "rttm_path": pred_dir / f"{stem}.rttm",
        "ctm_path": pred_dir / f"{stem}.ctm",
        "txt_path": pred_dir / f"{stem}.txt",
        "gecko_path": pred_dir / f"{stem}_gecko.json",
    }
    return {name: str(path) for name, path in artifact_names.items() if path.exists()}


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_srt_time(segment['start'])} --> {_format_srt_time(segment['end'])}",
                    f"{segment['speaker']}: {segment['text']}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    body = []
    for segment in segments:
        body.append(
            "\n".join(
                [
                    f"{_format_vtt_time(segment['start'])} --> {_format_vtt_time(segment['end'])}",
                    f"{segment['speaker']}: {segment['text']}",
                ]
            )
        )
    return "WEBVTT\n\n" + "\n\n".join(body) + ("\n" if body else "")


def _format_srt_time(seconds: float) -> str:
    return _format_subtitle_time(seconds, decimal=",")


def _format_vtt_time(seconds: float) -> str:
    return _format_subtitle_time(seconds, decimal=".")


def _format_subtitle_time(seconds: float, decimal: str) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{decimal}{millis:03}"


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cuda" in message and ("out of memory" in message or "cublas" in message)


def copy_upload_to_path(source_path: Path, target_path: Path) -> None:
    with source_path.open("rb") as source, target_path.open("wb") as target:
        shutil.copyfileobj(source, target)
