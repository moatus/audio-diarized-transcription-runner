"""Stateful live diarization runner built around NeMo OnlineClusteringDiarizer."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
import wave
from array import array
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .pipeline import (
    DiarizationRequest,
    RunnerInputError,
    UnsupportedParameterError,
    _hypothesis_field,
    _normalize_audio,
    _round_seconds,
    _safe_stem,
    _speaker_summaries,
    _timestamp_entry_get,
    _timestamp_entry_seconds,
    _unwrap_transcribe_result,
    run_diarized_transcription,
)
from .settings import RunnerSettings, configure_cache_environment


SUPPORTED_LIVE_VAD_STRATEGIES = {"energy", "provided"}
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
TRANSCRIPT_EVENT_SCHEMA_VERSION = "livepeer.diarized_transcript_event.v1"


class LiveSessionNotFoundError(KeyError):
    status_code = 404
    error_type = "session_not_found"


@dataclass(frozen=True)
class LiveSessionConfig:
    session_id: str = field(default_factory=lambda: f"live_{uuid.uuid4().hex[:12]}")
    language: str = "en"
    preset: str = "meeting"
    num_speakers: Optional[int] = None
    max_speakers: int = 8
    vad_strategy: str = "energy"
    rolling_window_seconds: float = 60.0
    min_speech_duration_seconds: float = 0.2
    min_silence_duration_seconds: float = 0.15
    energy_threshold: float = 0.012
    include_partial_segments: bool = True


@dataclass(frozen=True)
class LiveAudioIngestRequest:
    audio_path: Path
    filename: str
    content_type: Optional[str] = None
    sequence_index: Optional[int] = None
    vad_segments: Optional[List[Dict[str, float]]] = None


@dataclass(frozen=True)
class ProvisionalASRResult:
    text: str
    words: List[Dict[str, Any]]
    model: str
    text_status: str
    unavailable_reason: str = ""


class LiveDiarizationSession:
    """One online diarizer instance plus rolling audio/VAD state for a live session."""

    def __init__(
        self,
        *,
        settings: RunnerSettings,
        config: LiveSessionConfig,
        diarizer_factory: Optional[Callable[[Any], Any]] = None,
        provisional_asr_factory: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        _validate_live_config(config)
        self.settings = settings
        self.config = config
        self.session_id = config.session_id
        self.session_dir = settings.work_dir / "live_sessions" / self.session_id
        self.chunk_dir = self.session_dir / "chunks"
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.closed = False
        self._lock = threading.RLock()
        self._diarizer_factory = diarizer_factory
        self._provisional_asr_factory = provisional_asr_factory
        self._diarizer: Any = None
        self._provisional_asr_model: Any = None
        self._samples: List[float] = []
        self._cumulative_vad: List[List[float]] = []
        self._diarizer_vad: List[List[float]] = []
        self._segments: List[Dict[str, Any]] = []
        self._last_emitted_end = 0.0
        self._chunk_count = 0
        self.transcript_events_path = self.session_dir / "transcript-events.jsonl"
        self._transcript_events: List[Dict[str, Any]] = []
        self._next_transcript_event_index = 0

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self._diarizer is None:
                configure_cache_environment(self.settings)
                self.chunk_dir.mkdir(parents=True, exist_ok=True)
                self._write_bootstrap_manifest()
                self._diarizer = self._build_diarizer()
                self._append_transcript_event(
                    {
                        "event_type": "transcript.session.started",
                        "status": "active",
                        "is_provisional": True,
                        "is_final": False,
                        "authority": "online_diarization",
                        "text_status": "not_applicable",
                        "text": "",
                    }
                )
            return self.snapshot("session.started")

    def ingest_audio(self, request: LiveAudioIngestRequest) -> Dict[str, Any]:
        with self._lock:
            if self.closed:
                raise RunnerInputError(f"live session {self.session_id} is closed")
            if self._diarizer is None:
                self.start()

            sequence_index = request.sequence_index
            if sequence_index is None:
                sequence_index = self._chunk_count
            normalized_path = self.chunk_dir / f"chunk_{sequence_index:06d}_{_safe_stem(request.filename)}.wav"
            samples, chunk_duration = _decode_audio_to_samples(request.audio_path, normalized_path)
            if not samples:
                raise RunnerInputError("live audio chunk decoded to no samples")

            chunk_start = self.duration_seconds
            new_samples = self._samples + list(samples)
            chunk_end = len(new_samples) / float(self.settings.live_sample_rate)
            chunk_vad = self._vad_for_chunk(
                samples=samples,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                provided=request.vad_segments,
            )
            diarizer_chunk_vad = _clamp_vad_for_online_frame_boundaries(
                chunk_vad,
                chunk_end=chunk_end,
            )
            cumulative_vad = _merge_intervals(
                self._cumulative_vad + chunk_vad,
                min_gap=self.config.min_silence_duration_seconds,
            )
            diarizer_vad = _merge_intervals(
                self._diarizer_vad + diarizer_chunk_vad,
                min_gap=self.config.min_silence_duration_seconds,
            )

            diar_hyp = self._run_online_step(
                samples=new_samples,
                vad_timestamps=diarizer_vad,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_count=self._chunk_count,
            )
            segments = _normalize_online_diarization(diar_hyp)
            new_segments = [
                segment
                for segment in segments
                if float(segment.get("end", 0.0)) > self._last_emitted_end
            ]
            last_emitted_end = self._last_emitted_end
            if new_segments:
                last_emitted_end = max(float(segment["end"]) for segment in new_segments)

            provisional_asr = self._transcribe_provisional_chunk(
                normalized_path=normalized_path,
                chunk_start=chunk_start,
            )
            enriched_new_segments = _attach_provisional_text_to_segments(
                segments=new_segments,
                asr_result=provisional_asr,
            )
            enriched_segments = _merge_segment_text(
                current_segments=segments,
                previous_segments=self._segments,
                new_segments=enriched_new_segments,
            )

            self._samples = new_samples
            self._cumulative_vad = cumulative_vad
            self._diarizer_vad = diarizer_vad
            self._segments = enriched_segments
            self._last_emitted_end = last_emitted_end
            self._chunk_count += 1
            self.updated_at = time.time()
            chunk_payload = {
                "sequence_index": sequence_index,
                "filename": request.filename,
                "content_type": request.content_type,
                "duration_seconds": _round_seconds(chunk_duration),
                "start": _round_seconds(chunk_start),
                "end": _round_seconds(chunk_end),
                "vad_segments": _segments_from_intervals(chunk_vad),
                "normalized_path": str(normalized_path),
                "text": provisional_asr.text,
                "text_status": provisional_asr.text_status,
                "asr_model": provisional_asr.model,
            }
            if provisional_asr.unavailable_reason:
                chunk_payload["text_unavailable_reason"] = provisional_asr.unavailable_reason
            transcript_events = self._append_ingest_transcript_events(
                chunk=chunk_payload,
                new_segments=enriched_new_segments,
                provisional_asr=provisional_asr,
            )
            return {
                **self.snapshot("audio.ingested"),
                "chunk": chunk_payload,
                "new_segments": enriched_new_segments,
                "transcript_events": transcript_events,
            }

    def finish(
        self,
        *,
        run_final_transcription_path: bool = False,
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            self.closed = True
            self.updated_at = time.time()
            final_audio_path = self.session_dir / "session.wav"
            _write_samples_to_wav(final_audio_path, self._samples)
            response = {
                **self.snapshot("session.finished"),
                "final_audio_path": str(final_audio_path),
                "final_transcription": None,
                "transcript_events": [],
            }
            if run_final_transcription_path and self.duration_seconds > 0:
                transcription_settings = dataclass_replace(
                    self.settings,
                    default_asr_model=self.settings.live_final_asr_model,
                )
                request = DiarizationRequest(
                    audio_path=final_audio_path,
                    filename=final_audio_path.name,
                    content_type="audio/wav",
                    model=transcription_settings.default_model,
                    language=self.config.language,
                    preset=self.config.preset,
                    num_speakers=self.config.num_speakers,
                    max_speakers=self.config.max_speakers,
                    response_format="json",
                    include_words=include_words,
                    include_artifacts=include_artifacts,
                )
                response["final_transcription"] = run_diarized_transcription(
                    request,
                    transcription_settings,
                )
                response["transcript_events"] = self._append_final_transcript_events(
                    response["final_transcription"]
                )
            response["transcript_events"].append(
                self._append_transcript_event(
                    {
                        "event_type": "transcript.session.finished",
                        "status": "closed",
                        "is_provisional": False,
                        "is_final": True,
                        "authority": "session_lifecycle",
                        "text_status": "not_applicable",
                        "text": "",
                    }
                )
            )
            response["transcript_event_count"] = len(self._transcript_events)
            response["transcript_jsonl_path"] = str(self.transcript_events_path)
            return response

    @property
    def duration_seconds(self) -> float:
        return len(self._samples) / float(self.settings.live_sample_rate)

    def snapshot(self, event_type: str = "session.snapshot") -> Dict[str, Any]:
        speakers = _speaker_summaries(self._segments, [])
        return {
            "event_type": event_type,
            "session_id": self.session_id,
            "status": "closed" if self.closed else "active",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_seconds": _round_seconds(self.duration_seconds),
            "chunk_count": self._chunk_count,
            "config": _live_config_payload(self.config),
            "speaker_count": len(speakers),
            "speakers": speakers,
            "segments": self._segments,
            "vad_segments": _segments_from_intervals(self._cumulative_vad),
            "transcript_event_count": len(self._transcript_events),
            "transcript_jsonl_path": str(self.transcript_events_path),
            "models": {
                "online_diarizer": "OnlineClusteringDiarizer",
                "vad": self.config.vad_strategy,
                "speaker_embeddings": self.settings.default_speaker_model,
                "asr": self.settings.live_provisional_asr_model or None,
                "live_final_asr": self.settings.live_final_asr_model,
            },
        }

    def _transcribe_provisional_chunk(
        self,
        *,
        normalized_path: Path,
        chunk_start: float,
    ) -> ProvisionalASRResult:
        model_name = str(self.settings.live_provisional_asr_model or "").strip()
        if not model_name:
            return ProvisionalASRResult(
                text="",
                words=[],
                model="",
                text_status="not_available_online",
                unavailable_reason="LIVE_PROVISIONAL_ASR_MODEL is empty.",
            )
        try:
            model = self._get_provisional_asr_model(model_name)
            hypothesis = _transcribe_one_audio_file(model, normalized_path)
            text = _hypothesis_text(hypothesis)
            words = _extract_provisional_words(hypothesis, chunk_start=chunk_start)
            return ProvisionalASRResult(
                text=text,
                words=words,
                model=model_name,
                text_status="available" if text else "empty",
            )
        except Exception as error:
            return ProvisionalASRResult(
                text="",
                words=[],
                model=model_name,
                text_status="not_available_online",
                unavailable_reason=f"provisional ASR unavailable: {error}",
            )

    def _get_provisional_asr_model(self, model_name: str) -> Any:
        if self._provisional_asr_model is not None:
            return self._provisional_asr_model
        if self._provisional_asr_factory is not None:
            self._provisional_asr_model = self._provisional_asr_factory(
                model_name,
                self.settings.device,
            )
            return self._provisional_asr_model

        from nemo.collections.asr.models import ASRModel

        if model_name.endswith(".nemo"):
            asr_model = ASRModel.restore_from(restore_path=model_name)
        else:
            asr_model = ASRModel.from_pretrained(model_name)
        if str(self.settings.device).startswith("cuda"):
            asr_model = asr_model.cuda()
        elif hasattr(asr_model, "to"):
            asr_model = asr_model.to(self.settings.device)
        asr_model.eval()
        self._provisional_asr_model = asr_model
        return self._provisional_asr_model

    def _build_diarizer(self) -> Any:
        if self._diarizer_factory is not None:
            try:
                cfg = _online_diarizer_config(self.settings, self.config, self.session_dir)
            except ModuleNotFoundError:
                cfg = {"session_id": self.session_id, "session_dir": str(self.session_dir)}
            return self._diarizer_factory(cfg)
        cfg = _online_diarizer_config(self.settings, self.config, self.session_dir)
        from nemo.collections.asr.models.online_diarizer import OnlineClusteringDiarizer

        return OnlineClusteringDiarizer(cfg)

    def _run_online_step(
        self,
        *,
        samples: Sequence[float],
        vad_timestamps: Sequence[Sequence[float]],
        chunk_start: float,
        chunk_end: float,
        chunk_count: int,
    ) -> Any:
        rolling_start = max(0.0, chunk_end - self.config.rolling_window_seconds)
        start_sample = int(rolling_start * self.settings.live_sample_rate)
        previous_timing = {
            "frame_index": getattr(self._diarizer, "frame_index", None),
            "frame_start": getattr(self._diarizer, "frame_start", None),
            "buffer_start": getattr(self._diarizer, "buffer_start", None),
            "buffer_end": getattr(self._diarizer, "buffer_end", None),
            "total_buffer_in_secs": getattr(self._diarizer, "total_buffer_in_secs", None),
        }
        try:
            import torch

            audio_buffer: Any = torch.tensor(samples[start_sample:], dtype=torch.float32)
            diarizer_vad: Any = torch.tensor(vad_timestamps, dtype=torch.float32)
        except ModuleNotFoundError:
            audio_buffer = list(samples[start_sample:])
            diarizer_vad = [list(interval) for interval in vad_timestamps]

        self._diarizer.frame_index = chunk_count
        self._diarizer.frame_start = chunk_start
        self._diarizer.buffer_start = rolling_start
        self._diarizer.buffer_end = chunk_end
        self._diarizer.total_buffer_in_secs = chunk_end - rolling_start
        try:
            return self._diarizer.diarize_step(audio_buffer, diarizer_vad)
        except Exception:
            for name, value in previous_timing.items():
                if value is not None:
                    setattr(self._diarizer, name, value)
            raise

    def _vad_for_chunk(
        self,
        *,
        samples: Sequence[float],
        chunk_start: float,
        chunk_end: float,
        provided: Optional[List[Dict[str, float]]],
    ) -> List[List[float]]:
        if self.config.vad_strategy == "provided":
            if provided is None:
                raise RunnerInputError("vad_segments are required when vad_strategy=provided")
            intervals = []
            for segment in provided:
                start = chunk_start + float(segment["start"])
                end = chunk_start + float(segment["end"])
                intervals.append([max(chunk_start, start), min(chunk_end, end)])
            return _merge_intervals(intervals, min_gap=self.config.min_silence_duration_seconds)
        return _energy_vad(
            samples=samples,
            sample_rate=self.settings.live_sample_rate,
            base_offset=chunk_start,
            threshold=self.config.energy_threshold,
            min_speech_duration=self.config.min_speech_duration_seconds,
            min_silence_duration=self.config.min_silence_duration_seconds,
        )

    def _write_bootstrap_manifest(self) -> None:
        bootstrap_audio = self.session_dir / "bootstrap.wav"
        _write_samples_to_wav(bootstrap_audio, [0.0] * int(self.settings.live_sample_rate * 0.1))
        manifest_path = self.session_dir / "manifest.json"
        record = {
            "audio_filepath": str(bootstrap_audio),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": self.config.num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
            "uniq_id": self.session_id,
        }
        manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def _append_ingest_transcript_events(
        self,
        *,
        chunk: Dict[str, Any],
        new_segments: Sequence[Dict[str, Any]],
        provisional_asr: ProvisionalASRResult,
    ) -> List[Dict[str, Any]]:
        chunk_text = str(chunk.get("text") or "")
        chunk_text_status = str(chunk.get("text_status") or "empty")
        events = [
            self._append_transcript_event(
                {
                    "event_type": "transcript.chunk_ingested",
                    "status": "active",
                    "chunk_sequence_index": chunk["sequence_index"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "duration_seconds": chunk["duration_seconds"],
                    "speaker": None,
                    "text": chunk_text,
                    "is_provisional": True,
                    "is_final": False,
                    "authority": "online_diarization_provisional_asr",
                    "text_status": chunk_text_status,
                    "asr_model": provisional_asr.model,
                    **(
                        {"text_unavailable_reason": provisional_asr.unavailable_reason}
                        if provisional_asr.unavailable_reason
                        else {}
                    ),
                }
            )
        ]
        for segment in new_segments:
            text = str(segment.get("text") or "")
            text_status = (
                "available"
                if text.strip()
                else "empty"
                if provisional_asr.text
                else provisional_asr.text_status
            )
            events.append(
                self._append_transcript_event(
                    {
                        "event_type": "transcript.segment",
                        "status": "active",
                        "chunk_sequence_index": chunk["sequence_index"],
                        "segment_id": _segment_event_id(
                            self.session_id,
                            "provisional",
                            segment,
                        ),
                        "start": _round_seconds(segment.get("start", 0.0)),
                        "end": _round_seconds(segment.get("end", 0.0)),
                        "speaker": segment.get("speaker"),
                        "text": text,
                        "is_provisional": True,
                        "is_final": False,
                        "authority": "online_diarization_provisional_asr",
                        "text_status": text_status,
                        "asr_model": provisional_asr.model,
                        **(
                            {"text_unavailable_reason": provisional_asr.unavailable_reason}
                            if not text.strip() and provisional_asr.unavailable_reason
                            else {}
                        ),
                    }
                )
            )
        return events

    def _append_final_transcript_events(
        self,
        final_transcription: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        segments = final_transcription.get("segments") or []
        models = final_transcription.get("models") or {}
        asr_model = str(models.get("asr") or self.settings.live_final_asr_model)
        events: List[Dict[str, Any]] = []
        if segments:
            for index, segment in enumerate(segments):
                text = str(segment.get("text") or "").strip()
                events.append(
                    self._append_transcript_event(
                        {
                            "event_type": "transcript.segment",
                            "status": "closed",
                            "segment_id": _segment_event_id(
                                self.session_id,
                                f"final:{index:06d}",
                                segment,
                            ),
                            "start": _round_seconds(segment.get("start", 0.0)),
                            "end": _round_seconds(segment.get("end", 0.0)),
                            "speaker": segment.get("speaker"),
                            "text": text,
                            "is_provisional": False,
                            "is_final": True,
                            "authority": "final_offline_diarized_transcription",
                            "text_status": "available" if text else "empty",
                            "asr_model": asr_model,
                        }
                    )
                )
        else:
            text = str(final_transcription.get("text") or "").strip()
            if text:
                events.append(
                    self._append_transcript_event(
                        {
                            "event_type": "transcript.segment",
                            "status": "closed",
                            "segment_id": f"{self.session_id}:final:text",
                            "start": 0.0,
                            "end": _round_seconds(self.duration_seconds),
                            "speaker": None,
                            "text": text,
                            "is_provisional": False,
                            "is_final": True,
                            "authority": "final_offline_diarized_transcription",
                            "text_status": "available",
                            "asr_model": asr_model,
                        }
                    )
                )
        return events

    def _append_transcript_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event_index = self._next_transcript_event_index
        self._next_transcript_event_index += 1
        event_with_defaults = {
            "schema_version": TRANSCRIPT_EVENT_SCHEMA_VERSION,
            "event_id": f"{self.session_id}:transcript:{event_index:06d}",
            "transcript_event_index": event_index,
            "session_id": self.session_id,
            "emitted_at_epoch": time.time(),
            **event,
        }
        self.transcript_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript_events_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event_with_defaults, sort_keys=True) + "\n")
        self._transcript_events.append(event_with_defaults)
        return event_with_defaults


class LiveDiarizationSessionManager:
    def __init__(
        self,
        *,
        settings: RunnerSettings,
        diarizer_factory: Optional[Callable[[Any], Any]] = None,
        provisional_asr_factory: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        self.settings = settings
        self._diarizer_factory = diarizer_factory
        self._provisional_asr_factory = provisional_asr_factory
        self._sessions: Dict[str, LiveDiarizationSession] = {}
        self._lock = threading.Lock()

    def create_session(self, config: LiveSessionConfig) -> Dict[str, Any]:
        self.cleanup_expired()
        with self._lock:
            if config.session_id in self._sessions:
                raise UnsupportedParameterError(f"live session {config.session_id} already exists")
            session = LiveDiarizationSession(
                settings=self.settings,
                config=config,
                diarizer_factory=self._diarizer_factory,
                provisional_asr_factory=self._provisional_asr_factory,
            )
            self._sessions[config.session_id] = session
        return session.start()

    def get_session(self, session_id: str) -> LiveDiarizationSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise LiveSessionNotFoundError(session_id)
        return session

    def ingest_audio(self, session_id: str, request: LiveAudioIngestRequest) -> Dict[str, Any]:
        self.cleanup_expired()
        return self.get_session(session_id).ingest_audio(request)

    def finish_session(
        self,
        session_id: str,
        *,
        run_final_transcription_path: bool = False,
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        response = session.finish(
            run_final_transcription_path=run_final_transcription_path,
            include_words=include_words,
            include_artifacts=include_artifacts,
        )
        return response

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise LiveSessionNotFoundError(session_id)
        return session.finish(run_final_transcription_path=False)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired: List[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                age = now - session.updated_at
                if session.closed and age > self.settings.live_closed_session_ttl_seconds:
                    expired.append(session_id)
                elif age > self.settings.live_session_ttl_seconds:
                    expired.append(session_id)
            for session_id in expired:
                self._sessions.pop(session_id, None)
        return len(expired)


def _validate_live_config(config: LiveSessionConfig) -> None:
    if not isinstance(config.session_id, str) or not SESSION_ID_PATTERN.fullmatch(
        config.session_id
    ):
        raise UnsupportedParameterError(
            "session_id must be 1-128 characters and contain only ASCII letters, "
            "digits, underscore, hyphen, or dot; it must start with a letter or digit"
        )
    if config.language != "en":
        raise UnsupportedParameterError("Only language=en is supported in phase 1")
    if config.preset != "meeting":
        raise UnsupportedParameterError("Only preset=meeting is supported in phase 1")
    if config.vad_strategy not in SUPPORTED_LIVE_VAD_STRATEGIES:
        raise UnsupportedParameterError(
            f"vad_strategy must be one of {sorted(SUPPORTED_LIVE_VAD_STRATEGIES)}"
        )
    if config.max_speakers <= 0:
        raise UnsupportedParameterError("max_speakers must be positive")
    if config.num_speakers is not None and config.num_speakers > config.max_speakers:
        raise UnsupportedParameterError("num_speakers cannot exceed max_speakers")
    if config.rolling_window_seconds <= 0:
        raise UnsupportedParameterError("rolling_window_seconds must be positive")
    if config.energy_threshold <= 0:
        raise UnsupportedParameterError("energy_threshold must be positive")


def _online_diarizer_config(
    settings: RunnerSettings,
    config: LiveSessionConfig,
    session_dir: Path,
) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(settings.nemo_config_path)
    cfg.device = settings.device
    cfg.sample_rate = settings.live_sample_rate
    cfg.diarizer.manifest_filepath = str(session_dir / "manifest.json")
    cfg.diarizer.out_dir = str(session_dir / "nemo")
    cfg.diarizer.uniq_id = config.session_id
    cfg.diarizer.vad.model_path = settings.default_vad_model
    cfg.diarizer.speaker_embeddings.model_path = settings.default_speaker_model
    cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False
    cfg.diarizer.clustering.parameters.max_num_speakers = int(config.max_speakers)
    cfg.diarizer.clustering.parameters.oracle_num_speakers = config.num_speakers is not None
    cfg.diarizer.clustering.parameters.history_buffer_size = int(
        settings.live_history_buffer_size
    )
    cfg.diarizer.clustering.parameters.current_buffer_size = int(
        settings.live_current_buffer_size
    )
    cfg.diarizer.clustering.parameters.use_temporal_label_major_vote = True
    cfg.diarizer.clustering.parameters.temporal_label_major_vote_buffer_size = 3
    return cfg


def _decode_audio_to_samples(input_path: Path, normalized_path: Path) -> tuple[List[float], float]:
    duration = _normalize_audio(input_path, normalized_path)
    with wave.open(str(normalized_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise RunnerInputError("normalized live audio must be 16 kHz mono int16 WAV")
    pcm = array("h")
    pcm.frombytes(frames)
    if pcm.itemsize != 2:
        pcm.byteswap()
    return [max(-1.0, min(1.0, sample / 32768.0)) for sample in pcm], duration


def _write_samples_to_wav(path: Path, samples: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array("h", [int(max(-1.0, min(1.0, sample)) * 32767) for sample in samples])
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())


def _energy_vad(
    *,
    samples: Sequence[float],
    sample_rate: int,
    base_offset: float,
    threshold: float,
    min_speech_duration: float,
    min_silence_duration: float,
) -> List[List[float]]:
    frame_size = max(1, int(sample_rate * 0.03))
    speech_frames: List[tuple[float, float]] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        if rms >= threshold:
            frame_start = base_offset + start / sample_rate
            frame_end = base_offset + min(len(samples), start + frame_size) / sample_rate
            speech_frames.append((frame_start, frame_end))
    merged = _merge_intervals(
        [[start, end] for start, end in speech_frames],
        min_gap=min_silence_duration,
    )
    return [
        [start, end]
        for start, end in merged
        if end - start >= min_speech_duration
    ]


def _clamp_vad_for_online_frame_boundaries(
    intervals: Iterable[Sequence[float]],
    *,
    chunk_end: float,
) -> List[List[float]]:
    clamped: List[List[float]] = []
    boundary_epsilon = 0.001
    for interval in intervals:
        start = float(interval[0])
        end = float(interval[1])
        if math.isclose(end, chunk_end, abs_tol=boundary_epsilon / 2):
            end = chunk_end - boundary_epsilon
        if end > start:
            clamped.append([start, end])
    return clamped


def _merge_intervals(intervals: Iterable[Sequence[float]], *, min_gap: float) -> List[List[float]]:
    cleaned = sorted(
        [list(map(float, interval)) for interval in intervals if float(interval[1]) > float(interval[0])],
        key=lambda item: item[0],
    )
    merged: List[List[float]] = []
    for start, end in cleaned:
        if not merged or start - merged[-1][1] > min_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [[_round_seconds(start), _round_seconds(end)] for start, end in merged]


def _normalize_online_diarization(diar_hyp: Any) -> List[Dict[str, Any]]:
    if diar_hyp is None:
        return []
    if hasattr(diar_hyp, "tolist"):
        diar_hyp = diar_hyp.tolist()
    segments: List[Dict[str, Any]] = []
    for item in diar_hyp:
        if isinstance(item, dict):
            speaker = str(item.get("speaker") or item.get("label") or "speaker_0")
            start = _round_seconds(item.get("start", item.get("start_time", 0)))
            end = _round_seconds(item.get("end", item.get("end_time", start)))
        else:
            parts = str(item).strip().split()
            if len(parts) < 3:
                continue
            start = _round_seconds(parts[0])
            end = _round_seconds(parts[1])
            speaker = _normalize_speaker_label(parts[2])
        if end > start:
            segments.append({"speaker": speaker, "start": start, "end": end, "text": ""})
    return segments


def _transcribe_one_audio_file(asr_model: Any, audio_path: Path) -> Any:
    audio_paths = [str(audio_path)]
    try:
        result = asr_model.transcribe(audio_paths, batch_size=1, timestamps=True)
    except TypeError:
        result = asr_model.transcribe(audio_paths, batch_size=1)
    hypotheses = _unwrap_transcribe_result(result)
    if not hypotheses:
        return ""
    return hypotheses[0]


def _hypothesis_text(hypothesis: Any) -> str:
    if hypothesis is None:
        return ""
    if isinstance(hypothesis, str):
        return hypothesis.strip()
    text = _hypothesis_field(hypothesis, "text")
    if text is None:
        text = _hypothesis_field(hypothesis, "transcription")
    return str(text or "").strip()


def _extract_provisional_words(
    hypothesis: Any,
    *,
    chunk_start: float,
) -> List[Dict[str, Any]]:
    timestamp = _hypothesis_field(hypothesis, "timestamp")
    if not isinstance(timestamp, dict):
        timestamp = _hypothesis_field(hypothesis, "timestep")
    if not isinstance(timestamp, dict):
        return []

    word_entries = timestamp.get("word") or []
    words: List[Dict[str, Any]] = []
    fallback_words = _hypothesis_text(hypothesis).split()
    for index, entry in enumerate(word_entries):
        word = _timestamp_entry_get(entry, "word")
        if word is None:
            word = _timestamp_entry_get(entry, "text")
        if word is None and index < len(fallback_words):
            word = fallback_words[index]
        start = _timestamp_entry_seconds(entry, ("start", "start_time"))
        end = _timestamp_entry_seconds(entry, ("end", "end_time"))
        word_text = str(word or "").strip()
        if not word_text or start is None or end is None:
            continue
        absolute_start = float(chunk_start) + float(start)
        absolute_end = float(chunk_start) + float(end)
        words.append(
            {
                "word": word_text,
                "start": _round_seconds(absolute_start),
                "end": _round_seconds(absolute_end),
            }
        )
    return words


def _attach_provisional_text_to_segments(
    *,
    segments: Sequence[Dict[str, Any]],
    asr_result: ProvisionalASRResult,
) -> List[Dict[str, Any]]:
    if not segments:
        return []
    if asr_result.words:
        return [
            {
                **segment,
                "text": _words_for_segment(segment, asr_result.words),
            }
            for segment in segments
        ]

    text = asr_result.text.strip()
    if not text:
        return [dict(segment) for segment in segments]
    if len(segments) == 1:
        return [{**segments[0], "text": text}]

    split_text = _split_text_across_segments(text, segments)
    return [
        {
            **segment,
            "text": split_text[index],
        }
        for index, segment in enumerate(segments)
    ]


def _words_for_segment(segment: Dict[str, Any], words: Sequence[Dict[str, Any]]) -> str:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    selected = []
    for word in words:
        word_start = float(word.get("start", 0.0))
        word_end = float(word.get("end", word_start))
        midpoint = (word_start + word_end) / 2.0
        if start <= midpoint <= end:
            selected.append(str(word.get("word") or ""))
    return " ".join(word for word in selected if word).strip()


def _split_text_across_segments(
    text: str,
    segments: Sequence[Dict[str, Any]],
) -> List[str]:
    tokens = text.split()
    if not tokens:
        return ["" for _ in segments]
    durations = [
        max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
        for segment in segments
    ]
    total_duration = sum(durations)
    if total_duration <= 0:
        durations = [1.0 for _ in segments]
        total_duration = float(len(segments))

    output: List[str] = []
    token_index = 0
    for index, duration in enumerate(durations):
        remaining_segments = len(segments) - index
        remaining_tokens = len(tokens) - token_index
        if index == len(durations) - 1:
            count = remaining_tokens
        else:
            proportional_count = round(len(tokens) * (duration / total_duration))
            count = max(1, min(proportional_count, remaining_tokens - (remaining_segments - 1)))
        output.append(" ".join(tokens[token_index : token_index + count]).strip())
        token_index += count
    return output


def _merge_segment_text(
    *,
    current_segments: Sequence[Dict[str, Any]],
    previous_segments: Sequence[Dict[str, Any]],
    new_segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    text_by_key = {
        _segment_key(segment): str(segment.get("text") or "")
        for segment in previous_segments
        if str(segment.get("text") or "").strip()
    }
    text_by_key.update(
        {
            _segment_key(segment): str(segment.get("text") or "")
            for segment in new_segments
            if str(segment.get("text") or "").strip()
        }
    )
    return [
        {
            **segment,
            "text": text_by_key.get(_segment_key(segment), str(segment.get("text") or "")),
        }
        for segment in current_segments
    ]


def _segment_key(segment: Dict[str, Any]) -> tuple[str, float, float]:
    return (
        str(segment.get("speaker") or ""),
        _round_seconds(segment.get("start", 0.0)),
        _round_seconds(segment.get("end", 0.0)),
    )


def _normalize_speaker_label(value: str) -> str:
    value = value.strip()
    if value.startswith("speaker_"):
        return value
    match = re.search(r"(\d+)$", value)
    return f"speaker_{match.group(1)}" if match else value


def _segments_from_intervals(intervals: Iterable[Sequence[float]]) -> List[Dict[str, float]]:
    return [{"start": _round_seconds(start), "end": _round_seconds(end)} for start, end in intervals]


def _segment_event_id(session_id: str, phase: str, segment: Dict[str, Any]) -> str:
    speaker = str(segment.get("speaker") or "unknown")
    start = _round_seconds(segment.get("start", 0.0))
    end = _round_seconds(segment.get("end", 0.0))
    return f"{session_id}:{phase}:{speaker}:{start:.3f}-{end:.3f}"


def _live_config_payload(config: LiveSessionConfig) -> Dict[str, Any]:
    return {
        "session_id": config.session_id,
        "language": config.language,
        "preset": config.preset,
        "num_speakers": config.num_speakers,
        "max_speakers": config.max_speakers,
        "vad_strategy": config.vad_strategy,
        "rolling_window_seconds": config.rolling_window_seconds,
        "min_speech_duration_seconds": config.min_speech_duration_seconds,
        "min_silence_duration_seconds": config.min_silence_duration_seconds,
        "energy_threshold": config.energy_threshold,
        "include_partial_segments": config.include_partial_segments,
    }
