"""HTTP surface for the local diarized transcription runner."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.websockets import WebSocketDisconnect
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
from .streaming import (
    TrueStreamingSessionConfig,
    TrueStreamingSessionManager,
)


settings = RunnerSettings()
live_sessions = LiveDiarizationSessionManager(settings=settings)
true_streaming_sessions = TrueStreamingSessionManager(settings=settings)
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
    "true_streaming_sessions_total": 0,
    "true_streaming_audio_frames_total": 0,
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
        "modes": ["openai-audio-transcriptions@v1", "local-direct"],
        "live_modes": ["sessioned-online-diarization@v0"],
        "streaming_modes": ["websocket-pcm16-true-streaming@v0"],
        "endpoints": {
            "bounded_transcriptions": "POST /v1/audio/transcriptions",
            "openai_compatible": "POST /v1/audio/transcriptions",
            "options": "GET /audio:diarized-transcription@v0/options",
            "live_sessions": "POST /v1/audio/diarized-transcriptions/live/sessions",
            "true_streaming": "WS /v1/audio/transcriptions/stream",
        },
        "models": [settings.default_model],
        "languages": ["en"],
        "presets": [settings.default_preset],
        "response_formats": ["json", "verbose_json", "text", "srt", "vtt"],
        "openai_compatible": {
            "endpoint": "POST /v1/audio/transcriptions",
            "native_capability": settings.capability_name,
            "additive": True,
            "response_formats": ["json", "verbose_json", "text", "srt", "vtt"],
            "timestamp_granularities": ["segment", "word"],
            "feature_flags": ["diarization", "include_diarization"],
            "extension_fields": [
                "diarization",
                "speaker_labeled_text",
                "usage",
                "models",
                "artifacts",
                "speaker",
            ],
        },
        "defaults": {
            "model": settings.default_model,
            "language": "en",
            "preset": settings.default_preset,
            "include_words": False,
            "include_artifacts": False,
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
            "live_provisional_asr": settings.live_provisional_asr_model,
            "live_final_asr": settings.live_final_asr_model,
            "true_streaming_asr": settings.true_streaming_asr_model,
            "true_streaming_diarization": settings.true_streaming_diar_model,
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
            "asr_strategy": "provisional-chunk-asr-during-ingest-with-final-offline-transcription-on-finish",
        },
        "true_streaming": {
            "endpoint": "WS /v1/audio/transcriptions/stream",
            "transport": "persistent WebSocket",
            "input_audio": "binary frames containing little-endian 16 kHz mono int16 PCM",
            "control_messages": [
                {"type": "finish"},
                {"type": "ping"},
            ],
            "engine": settings.true_streaming_engine,
            "enabled": settings.true_streaming_enabled,
            "sample_rate": settings.true_streaming_sample_rate,
            "asr_model": settings.true_streaming_asr_model,
            "diarization_model": settings.true_streaming_diar_model,
            "asr_chunk_seconds": settings.true_streaming_asr_chunk_seconds,
            "asr_greedy_max_symbols": settings.true_streaming_asr_greedy_max_symbols,
            "asr_eou_tokens": list(settings.true_streaming_asr_eou_tokens),
            "asr_stabilize_partials": settings.true_streaming_asr_stabilize_partials,
            "asr_emit_min_words": settings.true_streaming_asr_emit_min_words,
            "asr_emit_min_chars": settings.true_streaming_asr_emit_min_chars,
            "asr_emit_max_hold_seconds": settings.true_streaming_asr_emit_max_hold_seconds,
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
        (
            "livepeer_audio_diarized_transcription_true_streaming_sessions_total "
            f"{metrics['true_streaming_sessions_total']}"
        ),
        (
            "livepeer_audio_diarized_transcription_true_streaming_audio_frames_total "
            f"{metrics['true_streaming_audio_frames_total']}"
        ),
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.websocket("/v1/audio/transcriptions/stream")
async def true_streaming_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    if startup_error:
        await websocket.send_json(
            {"error": {"message": startup_error, "type": "server_error"}}
        )
        await websocket.close(code=1011)
        return

    session = None
    try:
        config = _streaming_config_from_query(websocket)
        session = await run_in_threadpool(true_streaming_sessions.create_session, config)
        metrics["true_streaming_sessions_total"] += 1
        await websocket.send_json(session.snapshot())
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                events = await run_in_threadpool(session.process_audio, message["bytes"])
                metrics["true_streaming_audio_frames_total"] += 1
                for event in events:
                    await websocket.send_json(event)
                continue
            if "text" in message and message["text"] is not None:
                should_close = await _handle_streaming_control_message(
                    websocket,
                    session,
                    message["text"],
                )
                if should_close:
                    return
    except WebSocketDisconnect:
        pass
    except UnsupportedParameterError as error:
        await websocket.send_json({"error": {"message": str(error), "type": error.error_type}})
        await websocket.close(code=1008)
    except RunnerInputError as error:
        await websocket.send_json({"error": {"message": str(error), "type": error.error_type}})
        await websocket.close(code=1003)
    except Exception as error:
        await websocket.send_json({"error": {"message": str(error), "type": "server_error"}})
        await websocket.close(code=1011)
    finally:
        if session is not None:
            try:
                if not session.closed:
                    await run_in_threadpool(session.finish)
            except Exception:
                pass
            finally:
                await run_in_threadpool(true_streaming_sessions.release_session, session)


@app.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(settings.default_model),
    language: str = Form("en"),
    response_format: str = Form("json"),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    preset: str = Form(settings.default_preset),
    num_speakers: Optional[int] = Form(None),
    max_speakers: int = Form(settings.default_max_speakers),
    include_words: bool = Form(False),
    include_artifacts: bool = Form(False),
    diarization: Optional[bool] = Form(None),
    include_diarization: Optional[bool] = Form(None),
) -> Response:
    del prompt, temperature
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": startup_error, "type": "server_error"}},
        )
    try:
        timestamp_granularities = await _openai_timestamp_granularities(request)
    except UnsupportedParameterError as error:
        return _error_response(str(error), error.status_code, error.error_type)
    if response_format == "verbose_json" and "word" in timestamp_granularities:
        include_words = True
    diarization_enabled = bool(
        diarization if diarization is not None else include_diarization
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
        request_model = DiarizationRequest(
            audio_path=uploaded_path,
            filename=file.filename or "audio",
            content_type=file.content_type,
            model=model,
            language=language,
            preset=preset,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
            response_format="json" if response_format == "verbose_json" else response_format,
            include_words=include_words,
            include_artifacts=include_artifacts,
        )
        started = time.monotonic()
        result = await run_in_threadpool(run_diarized_transcription, request_model, settings)
        elapsed = time.monotonic() - started
        work_units = int(result["usage"]["work_units"])
        _record_success(result["duration_seconds"], work_units)
        headers = {
            "X-Livepeer-Work-Units": str(work_units),
            "X-Livepeer-Runner-Elapsed-Seconds": f"{elapsed:.3f}",
        }
        return _openai_transcription_response(
            result=result,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            include_diarization=diarization_enabled,
            headers=headers,
        )
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


def _streaming_config_from_query(websocket: WebSocket) -> TrueStreamingSessionConfig:
    query = websocket.query_params
    max_speakers = _streaming_int_query_param(query, "max_speakers", 4)
    return TrueStreamingSessionConfig(
        session_id=query.get("session_id") or TrueStreamingSessionConfig().session_id,
        language=query.get("language", "en"),
        preset=query.get("preset", settings.default_preset),
        max_speakers=max_speakers,
        sample_rate=_streaming_int_query_param(
            query,
            "sample_rate",
            settings.true_streaming_sample_rate,
        ),
    )


def _streaming_int_query_param(query: Any, name: str, default: int) -> int:
    raw_value = query.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise UnsupportedParameterError(f"{name} must be an integer") from error


async def _handle_streaming_control_message(
    websocket: WebSocket,
    session: Any,
    message: str,
) -> bool:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        await websocket.send_json(
            {"error": {"message": "stream control messages must be JSON", "type": "invalid_request_error"}}
        )
        return False
    message_type = str(payload.get("type") or payload.get("event_type") or "").strip()
    if message_type == "ping":
        await websocket.send_json({"event_type": "pong", "session_id": session.session_id})
        return False
    if message_type in {"finish", "session.finish", "transcript.session.finish"}:
        finish_events = await run_in_threadpool(session.finish_events)
        for event in finish_events:
            await websocket.send_json(event)
        await websocket.close(code=1000)
        return True
    await websocket.send_json(
        {
            "error": {
                "message": f"unsupported stream control message type: {message_type or '<empty>'}",
                "type": "invalid_request_error",
            }
        }
    )
    return False


async def _openai_timestamp_granularities(request: Request) -> List[str]:
    form = await request.form()
    values: List[str] = []
    for key in ("timestamp_granularities[]", "timestamp_granularities"):
        for value in form.getlist(key):
            if value is None:
                continue
            for item in str(value).split(","):
                normalized = item.strip().lower()
                if normalized:
                    values.append(normalized)
    allowed = {"segment", "word"}
    unsupported = sorted(set(values) - allowed)
    if unsupported:
        raise UnsupportedParameterError(
            f"timestamp_granularities must contain only {sorted(allowed)}"
        )
    if not values:
        return ["segment"]
    deduped: List[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _openai_transcription_response(
    *,
    result: Dict[str, Any],
    response_format: str,
    timestamp_granularities: List[str],
    include_diarization: bool,
    headers: Dict[str, str],
) -> Response:
    if response_format == "text":
        return PlainTextResponse(_openai_plain_text(result), headers=headers)
    if response_format in {"srt", "vtt"}:
        artifact_key = f"{response_format}_path"
        path = Path(result.get("artifacts", {}).get(artifact_key, ""))
        return PlainTextResponse(path.read_text(encoding="utf-8"), headers=headers)
    if response_format == "json":
        return JSONResponse({"text": _openai_plain_text(result)}, headers=headers)
    if response_format == "verbose_json":
        return JSONResponse(
            _openai_verbose_json(
                result,
                timestamp_granularities=timestamp_granularities,
                include_diarization=include_diarization,
            ),
            headers=headers,
        )
    raise UnsupportedParameterError(
        "response_format must be one of ['json', 'text', 'srt', 'vtt', 'verbose_json']"
    )


def _openai_verbose_json(
    result: Dict[str, Any],
    *,
    timestamp_granularities: List[str],
    include_diarization: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task": "transcribe",
        "language": "english",
        "duration": result.get("duration_seconds", 0.0),
        "text": _openai_plain_text(result),
    }
    if "segment" in timestamp_granularities:
        payload["segments"] = _openai_segments(
            result.get("segments") or [],
            include_speaker=include_diarization,
        )
    if "word" in timestamp_granularities:
        payload["words"] = _openai_words(
            result.get("words") or [],
            include_speaker=include_diarization,
        )
    if include_diarization:
        payload.update(_openai_generic_extensions(result))
    return payload


def _openai_plain_text(result: Dict[str, Any]) -> str:
    segments = result.get("segments") or []
    if isinstance(segments, list):
        pieces = [
            str(segment.get("text") or "").strip()
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("text") or "").strip()
        ]
        if pieces:
            return " ".join(pieces)
    return str(result.get("text") or "").strip()


def _openai_segments(
    segments: List[Dict[str, Any]],
    *,
    include_speaker: bool,
) -> List[Dict[str, Any]]:
    openai_segments: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        payload: Dict[str, Any] = {
            "id": index,
            "seek": 0,
            "start": _openai_seconds(segment.get("start", 0.0)),
            "end": _openai_seconds(segment.get("end", 0.0)),
            "text": str(segment.get("text") or ""),
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0,
        }
        if include_speaker:
            payload["speaker"] = segment.get("speaker")
        openai_segments.append(payload)
    return openai_segments


def _openai_words(
    words: List[Dict[str, Any]],
    *,
    include_speaker: bool,
) -> List[Dict[str, Any]]:
    openai_words: List[Dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        payload: Dict[str, Any] = {
            "word": str(word.get("word") or ""),
            "start": _openai_seconds(word.get("start", 0.0)),
            "end": _openai_seconds(word.get("end", 0.0)),
        }
        if include_speaker:
            payload["speaker"] = word.get("speaker")
        openai_words.append(payload)
    return openai_words


def _openai_generic_extensions(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "transcription_id": result.get("id"),
        "capability": result.get("capability"),
        "mode": result.get("mode"),
        "models": result.get("models") or {},
        "usage": result.get("usage") or {},
        "speaker_labeled_text": str(result.get("text") or ""),
        "diarization": {
            "speaker_count": int(result.get("speaker_count") or 0),
            "speakers": result.get("speakers") or [],
            "segments": result.get("segments") or [],
            "words": result.get("words") or [],
        },
        "artifacts": result.get("artifacts") or {},
    }


def _openai_seconds(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _error_response(message: str, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )
