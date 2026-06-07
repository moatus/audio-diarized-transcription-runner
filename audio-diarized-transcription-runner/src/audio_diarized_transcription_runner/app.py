"""HTTP surface for the local diarized transcription runner."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .live import (
    LiveAudioIngestRequest,
    LiveDiarizationSessionManager,
    LiveSessionConfig,
    LiveSessionNotFoundError,
)
from .pipeline import (
    CudaOutOfMemoryError,
    DiarizationRequest,
    RunnerInputError,
    UnsupportedParameterError,
    assert_runtime_ready,
    run_diarized_transcription,
)
from .settings import RunnerSettings


settings = RunnerSettings()
live_sessions = LiveDiarizationSessionManager(settings=settings)
startup_error: Optional[str] = None
runtime_facts = {"cuda_available": False}
metrics = {
    "requests_total": 0,
    "requests_failed_total": 0,
    "audio_seconds_total": 0,
    "work_units_total": 0,
    "live_sessions_total": 0,
    "live_audio_chunks_total": 0,
    "live_sessions_finished_total": 0,
}
_queue_lock = asyncio.Lock()
_inflight = 0


app = FastAPI(title="Livepeer Audio Diarized Transcription Runner")


class LiveSessionCreateBody(BaseModel):
    session_id: Optional[str] = Field(default=None)
    language: str = Field(default="en")
    preset: str = Field(default="meeting")
    num_speakers: Optional[int] = Field(default=None)
    max_speakers: int = Field(default=8)
    vad_strategy: str = Field(default="energy")
    rolling_window_seconds: float = Field(default=60.0)
    min_speech_duration_seconds: float = Field(default=0.2)
    min_silence_duration_seconds: float = Field(default=0.15)
    energy_threshold: float = Field(default=0.012)
    include_partial_segments: bool = Field(default=True)


class LiveSessionFinishBody(BaseModel):
    run_final_transcription: bool = Field(default=False)
    include_words: bool = Field(default=True)
    include_artifacts: bool = Field(default=True)


@app.on_event("startup")
async def startup() -> None:
    global startup_error, runtime_facts
    try:
        runtime_facts = assert_runtime_ready(settings)
        startup_error = None
    except Exception as error:
        startup_error = str(error)
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return _error_response(str(detail), exc.status_code, "request_error")


@app.get("/healthz")
def healthz() -> JSONResponse:
    if startup_error:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "device": settings.device,
                "cuda_available": runtime_facts.get("cuda_available", False),
                "error": startup_error,
            },
        )
    return JSONResponse(
        {
            "status": "ok",
            "device": settings.device,
            "cuda_available": runtime_facts.get("cuda_available", False),
            "torch_version": runtime_facts.get("torch_version"),
            "cuda_device_count": runtime_facts.get("cuda_device_count", 0),
            "cuda_device_name": runtime_facts.get("cuda_device_name"),
        }
    )


@app.get("/audio:diarized-transcription@v0/options")
def options() -> dict:
    return {
        "capability": settings.capability_name,
        "modes": ["local-direct", "http-multipart@v0"],
        "live_modes": ["sessioned-online-diarization@v0"],
        "models": [settings.default_model],
        "languages": ["en"],
        "presets": [settings.default_preset],
        "response_formats": ["json", "text", "srt", "vtt"],
        "defaults": {
            "model": settings.default_model,
            "language": "en",
            "preset": settings.default_preset,
            "include_words": True,
            "include_artifacts": True,
            "max_speakers": settings.default_max_speakers,
        },
        "limits": {
            "max_audio_mb": settings.max_audio_mb,
            "max_queue_size": settings.max_queue_size,
            "live_session_ttl_seconds": settings.live_session_ttl_seconds,
            "live_closed_session_ttl_seconds": settings.live_closed_session_ttl_seconds,
        },
        "models_detail": {
            "vad": settings.default_vad_model,
            "speaker_embeddings": settings.default_speaker_model,
            "asr": settings.default_asr_model,
            "live_final_asr": settings.live_final_asr_model,
        },
        "live": {
            "endpoints": {
                "create": "POST /v1/audio/diarized-transcriptions/live/sessions",
                "ingest": "POST /v1/audio/diarized-transcriptions/live/sessions/{session_id}/audio",
                "snapshot": "GET /v1/audio/diarized-transcriptions/live/sessions/{session_id}",
                "finish": "POST /v1/audio/diarized-transcriptions/live/sessions/{session_id}/finish",
                "delete": "DELETE /v1/audio/diarized-transcriptions/live/sessions/{session_id}",
            },
            "vad_strategies": ["energy", "provided"],
            "sample_rate": settings.live_sample_rate,
            "asr_strategy": "final-offline-transcription-on-finish",
        },
    }


@app.get("/metrics")
def metrics_endpoint() -> Response:
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics disabled")
    lines = [
        f"livepeer_audio_diarized_transcription_requests_total {metrics['requests_total']}",
        (
            "livepeer_audio_diarized_transcription_requests_failed_total "
            f"{metrics['requests_failed_total']}"
        ),
        (
            "livepeer_audio_diarized_transcription_audio_seconds_total "
            f"{metrics['audio_seconds_total']}"
        ),
        (
            "livepeer_audio_diarized_transcription_work_units_total "
            f"{metrics['work_units_total']}"
        ),
        (
            "livepeer_audio_diarized_transcription_live_sessions_total "
            f"{metrics['live_sessions_total']}"
        ),
        (
            "livepeer_audio_diarized_transcription_live_audio_chunks_total "
            f"{metrics['live_audio_chunks_total']}"
        ),
        (
            "livepeer_audio_diarized_transcription_live_sessions_finished_total "
            f"{metrics['live_sessions_finished_total']}"
        ),
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/audio/diarized-transcriptions")
async def diarized_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(settings.default_model),
    language: str = Form("en"),
    preset: str = Form(settings.default_preset),
    num_speakers: Optional[int] = Form(None),
    max_speakers: int = Form(settings.default_max_speakers),
    response_format: str = Form("json"),
    include_words: bool = Form(True),
    include_artifacts: bool = Form(True),
) -> Response:
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": startup_error, "type": "server_error"}},
        )

    acquired = await _try_acquire_slot()
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail={"error": {"message": "runner queue full", "type": "queue_full"}},
        )

    uploaded_path: Optional[Path] = None
    try:
        uploaded_path = await _persist_upload(file)
        request = DiarizationRequest(
            audio_path=uploaded_path,
            filename=file.filename or "audio",
            content_type=file.content_type,
            model=model,
            language=language,
            preset=preset,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
            response_format=response_format,
            include_words=include_words,
            include_artifacts=include_artifacts,
        )
        started = time.monotonic()
        result = await run_in_threadpool(run_diarized_transcription, request, settings)
        elapsed = time.monotonic() - started
        work_units = int(result["usage"]["work_units"])
        _record_success(result["duration_seconds"], work_units)
        headers = {
            "X-Livepeer-Work-Units": str(work_units),
            "X-Livepeer-Runner-Elapsed-Seconds": f"{elapsed:.3f}",
        }
        if response_format == "text":
            return PlainTextResponse(result["text"], headers=headers)
        if response_format in {"srt", "vtt"}:
            artifact_key = f"{response_format}_path"
            path = Path(result["artifacts"].get(artifact_key, ""))
            return PlainTextResponse(path.read_text(encoding="utf-8"), headers=headers)
        return JSONResponse(result, headers=headers)
    except CudaOutOfMemoryError as error:
        _record_failure()
        return _error_response(str(error), 507, "cuda_out_of_memory")
    except UnsupportedParameterError as error:
        _record_failure()
        return _error_response(str(error), error.status_code, error.error_type)
    except RunnerInputError as error:
        _record_failure()
        return _error_response(str(error), error.status_code, error.error_type)
    except Exception as error:
        _record_failure()
        return _error_response(str(error), 500, "server_error")
    finally:
        await _release_slot()
        if uploaded_path and uploaded_path.exists():
            uploaded_path.unlink(missing_ok=True)


@app.post("/v1/audio/diarized-transcriptions/live/sessions")
async def create_live_session(
    body: Optional[LiveSessionCreateBody] = Body(default=None),
) -> JSONResponse:
    body = body or LiveSessionCreateBody()
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": startup_error, "type": "server_error"}},
        )
    config = LiveSessionConfig(
        session_id=body.session_id or LiveSessionConfig().session_id,
        language=body.language,
        preset=body.preset,
        num_speakers=body.num_speakers,
        max_speakers=body.max_speakers,
        vad_strategy=body.vad_strategy,
        rolling_window_seconds=body.rolling_window_seconds,
        min_speech_duration_seconds=body.min_speech_duration_seconds,
        min_silence_duration_seconds=body.min_silence_duration_seconds,
        energy_threshold=body.energy_threshold,
        include_partial_segments=body.include_partial_segments,
    )
    try:
        result = await run_in_threadpool(live_sessions.create_session, config)
        metrics["live_sessions_total"] += 1
        return JSONResponse(result)
    except UnsupportedParameterError as error:
        return _error_response(str(error), error.status_code, error.error_type)
    except Exception as error:
        return _error_response(str(error), 500, "server_error")


@app.get("/v1/audio/diarized-transcriptions/live/sessions/{session_id}")
def get_live_session(session_id: str) -> JSONResponse:
    try:
        return JSONResponse(live_sessions.get_session(session_id).snapshot())
    except LiveSessionNotFoundError:
        return _error_response(f"live session {session_id} not found", 404, "session_not_found")


@app.post("/v1/audio/diarized-transcriptions/live/sessions/{session_id}/audio")
async def ingest_live_audio(
    session_id: str,
    file: UploadFile = File(...),
    sequence_index: Optional[int] = Form(None),
    vad_segments_json: Optional[str] = Form(None),
) -> JSONResponse:
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": startup_error, "type": "server_error"}},
        )
    uploaded_path: Optional[Path] = None
    try:
        uploaded_path = await _persist_upload(file)
        request = LiveAudioIngestRequest(
            audio_path=uploaded_path,
            filename=file.filename or "audio",
            content_type=file.content_type,
            sequence_index=sequence_index,
            vad_segments=_parse_vad_segments(vad_segments_json),
        )
        result = await run_in_threadpool(live_sessions.ingest_audio, session_id, request)
        metrics["live_audio_chunks_total"] += 1
        return JSONResponse(result)
    except LiveSessionNotFoundError:
        return _error_response(f"live session {session_id} not found", 404, "session_not_found")
    except UnsupportedParameterError as error:
        return _error_response(str(error), error.status_code, error.error_type)
    except RunnerInputError as error:
        return _error_response(str(error), error.status_code, error.error_type)
    except Exception as error:
        return _error_response(str(error), 500, "server_error")
    finally:
        if uploaded_path and uploaded_path.exists():
            uploaded_path.unlink(missing_ok=True)


@app.post("/v1/audio/diarized-transcriptions/live/sessions/{session_id}/finish")
async def finish_live_session(
    session_id: str,
    body: Optional[LiveSessionFinishBody] = Body(default=None),
) -> JSONResponse:
    body = body or LiveSessionFinishBody()
    try:
        result = await run_in_threadpool(
            live_sessions.finish_session,
            session_id,
            run_final_transcription_path=body.run_final_transcription,
            include_words=body.include_words,
            include_artifacts=body.include_artifacts,
        )
        metrics["live_sessions_finished_total"] += 1
        return JSONResponse(result)
    except LiveSessionNotFoundError:
        return _error_response(f"live session {session_id} not found", 404, "session_not_found")
    except CudaOutOfMemoryError as error:
        return _error_response(str(error), 507, "cuda_out_of_memory")
    except RunnerInputError as error:
        return _error_response(str(error), error.status_code, error.error_type)
    except Exception as error:
        return _error_response(str(error), 500, "server_error")


@app.delete("/v1/audio/diarized-transcriptions/live/sessions/{session_id}")
async def delete_live_session(session_id: str) -> JSONResponse:
    try:
        result = await run_in_threadpool(live_sessions.delete_session, session_id)
        return JSONResponse(result)
    except LiveSessionNotFoundError:
        return _error_response(f"live session {session_id} not found", 404, "session_not_found")


async def _try_acquire_slot() -> bool:
    global _inflight
    async with _queue_lock:
        if _inflight >= settings.max_queue_size:
            return False
        _inflight += 1
        return True


async def _release_slot() -> None:
    global _inflight
    async with _queue_lock:
        _inflight = max(0, _inflight - 1)


async def _persist_upload(file: UploadFile) -> Path:
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = settings.work_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{time.time_ns()}_{Path(file.filename or 'audio').name}"
    max_bytes = settings.max_audio_mb * 1024 * 1024
    bytes_seen = 0
    with target.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            bytes_seen += len(chunk)
            if bytes_seen > max_bytes:
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": {
                            "message": f"uploaded audio exceeds MAX_AUDIO_MB={settings.max_audio_mb}",
                            "type": "request_too_large",
                        }
                    },
                )
            output.write(chunk)
    if bytes_seen == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "empty upload", "type": "invalid_request_error"}},
        )
    return target


def _record_success(audio_seconds: float, work_units: int) -> None:
    metrics["requests_total"] += 1
    metrics["audio_seconds_total"] += int(audio_seconds)
    metrics["work_units_total"] += work_units


def _record_failure() -> None:
    metrics["requests_total"] += 1
    metrics["requests_failed_total"] += 1


def _parse_vad_segments(value: Optional[str]) -> Optional[List[Dict[str, float]]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise RunnerInputError("vad_segments_json must be valid JSON") from error
    if not isinstance(parsed, list):
        raise RunnerInputError("vad_segments_json must be a list")
    segments: List[Dict[str, float]] = []
    for item in parsed:
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise RunnerInputError("each VAD segment must contain start and end")
        segments.append({"start": float(item["start"]), "end": float(item["end"])})
    return segments


def _error_response(message: str, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )
