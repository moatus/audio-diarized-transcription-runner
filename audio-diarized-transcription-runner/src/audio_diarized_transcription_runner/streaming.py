"""Persistent WebSocket streaming ASR plus diarization session support."""

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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .pipeline import RunnerInputError, UnsupportedParameterError, _round_seconds
from .settings import RunnerSettings, configure_cache_environment


STREAMING_EVENT_SCHEMA_VERSION = "livepeer.true_streaming_transcript_event.v1"
SUPPORTED_STREAMING_ENGINES = {"nemo", "fake"}
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class StreamingASREngine(Protocol):
    model_name: str

    def transcribe(self, audio: bytes, stream_id: str) -> Any:
        ...

    def reset_state(self, stream_id: str) -> None:
        ...


class StreamingDiarizationEngine(Protocol):
    model_name: str

    def diarize(self, audio: bytes, stream_id: str) -> Any:
        ...

    def reset_state(self, stream_id: str) -> None:
        ...


@dataclass(frozen=True)
class TrueStreamingSessionConfig:
    session_id: str = field(default_factory=lambda: f"stream_{uuid.uuid4().hex[:12]}")
    language: str = "en"
    preset: str = "meeting"
    max_speakers: int = 4
    sample_rate: int = 16000
    run_final_transcription_on_finish: bool = False


@dataclass(frozen=True)
class StreamingEnginePair:
    asr: StreamingASREngine
    diarization: StreamingDiarizationEngine


class TrueStreamingSession:
    """One persistent transport session with stateful ASR and diarization engines."""

    def __init__(
        self,
        *,
        settings: RunnerSettings,
        config: TrueStreamingSessionConfig,
        engines: StreamingEnginePair,
        owner_id: Optional[str] = None,
        on_close: Optional[Callable[["TrueStreamingSession"], None]] = None,
    ) -> None:
        _validate_streaming_config(config)
        self.settings = settings
        self.config = config
        self.session_id = config.session_id
        self.owner_id = owner_id or uuid.uuid4().hex
        self._on_close = on_close
        self.engines = engines
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.closed = False
        self.session_dir = (
            settings.work_dir / "true_streaming_sessions" / self.session_id / self.owner_id
        )
        self.events_path = self.session_dir / "stream-events.jsonl"
        self.audio_path = self.session_dir / "session.wav"
        self._event_index = 0
        self._sample_count = 0
        self._pcm_chunks: List[bytes] = []
        self._last_text = ""
        self._last_speaker: Optional[str] = None
        self._transcript_stabilizer = (
            _StreamingTranscriptStabilizer(
                min_words=settings.true_streaming_asr_emit_min_words,
                min_chars=settings.true_streaming_asr_emit_min_chars,
                max_hold_seconds=settings.true_streaming_asr_emit_max_hold_seconds,
            )
            if settings.true_streaming_asr_stabilize_partials
            and settings.true_streaming_engine.strip().lower() == "nemo"
            else None
        )
        self._events: List[Dict[str, Any]] = []
        self._finished_event: Optional[Dict[str, Any]] = None

    def start(self) -> Dict[str, Any]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self._append_event(
            {
                "event_type": "transcript.session.started",
                "status": "active",
                "is_provisional": True,
                "is_final": False,
                "authority": "true_streaming_session",
                "text": "",
                "text_status": "not_applicable",
                "models": self.models_payload,
            }
        )

    def process_audio(self, audio: bytes) -> List[Dict[str, Any]]:
        if self.closed:
            raise RunnerInputError(f"streaming session {self.session_id} is closed")
        if not audio:
            return []
        if len(audio) % 2:
            raise RunnerInputError("streaming audio frames must be little-endian int16 PCM")

        started = time.monotonic()
        chunk_sample_count = len(audio) // 2
        chunk_start = self._sample_count / float(self.config.sample_rate)
        chunk_end = (self._sample_count + chunk_sample_count) / float(self.config.sample_rate)
        if chunk_end > self.settings.true_streaming_max_session_seconds:
            raise RunnerInputError(
                "streaming session exceeded TRUE_STREAMING_MAX_SESSION_SECONDS="
                f"{self.settings.true_streaming_max_session_seconds}"
            )

        self._pcm_chunks.append(audio)
        self._sample_count += chunk_sample_count
        self.updated_at = time.time()

        diarization = self.engines.diarization.diarize(audio, self.session_id)
        speaker_payload = _speaker_payload_from_diarization(
            diarization,
            chunk_start=chunk_start,
            frame_seconds=self.settings.true_streaming_diar_frame_seconds,
            threshold=self.settings.true_streaming_diar_threshold,
        )
        if speaker_payload.get("speaker"):
            self._last_speaker = str(speaker_payload["speaker"])

        asr_result = self.engines.asr.transcribe(audio, self.session_id)
        text = _asr_text(asr_result)
        token_pieces = _asr_token_pieces(asr_result)
        is_final = _asr_is_final(asr_result)
        transcript_start = chunk_start
        transcript_text = text.strip()
        if self._transcript_stabilizer is not None:
            stable = (
                self._transcript_stabilizer.update_token_pieces(
                    token_pieces,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                if token_pieces is not None
                else self._transcript_stabilizer.update(
                    text,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
            )
            transcript_text = stable.text if stable is not None else ""
            transcript_start = stable.start if stable is not None else chunk_start
            if is_final and not transcript_text:
                stable = self._transcript_stabilizer.flush(chunk_end=chunk_end)
                transcript_text = stable.text if stable is not None else ""
                transcript_start = stable.start if stable is not None else chunk_start
        text_changed = bool(transcript_text) and transcript_text != self._last_text
        if text_changed:
            self._last_text = transcript_text

        events = [
            self._append_event(
                {
                    "event_type": "audio.frame.received",
                    "status": "active",
                    "start": _round_seconds(chunk_start),
                    "end": _round_seconds(chunk_end),
                    "duration_seconds": _round_seconds(chunk_end - chunk_start),
                    "audio_bytes": len(audio),
                    "is_provisional": True,
                    "is_final": False,
                    "authority": "websocket_pcm16_transport",
                    "text": "",
                    "text_status": "not_applicable",
                }
            )
        ]
        if speaker_payload:
            events.append(
                self._append_event(
                    {
                        "event_type": "speaker.update",
                        "status": "active",
                        "start": speaker_payload["start"],
                        "end": speaker_payload["end"],
                        "speaker": speaker_payload.get("speaker"),
                        "speaker_confidence": speaker_payload.get("speaker_confidence"),
                        "speaker_probabilities": speaker_payload.get("speaker_probabilities"),
                        "is_provisional": True,
                        "is_final": False,
                        "authority": "nemo_streaming_sortformer",
                        "text": "",
                        "text_status": "not_applicable",
                        "diarization_model": self.engines.diarization.model_name,
                    }
                )
            )
        if text_changed or is_final:
            events.append(
                self._append_event(
                    {
                        "event_type": "transcript.segment",
                        "status": "active",
                        "start": _round_seconds(transcript_start),
                        "end": _round_seconds(chunk_end),
                        "speaker": self._last_speaker,
                        "text": transcript_text,
                        "is_provisional": not is_final,
                        "is_final": is_final,
                        "authority": "nemo_cache_aware_streaming_asr",
                        "text_status": "available" if transcript_text else "empty",
                        "asr_model": self.engines.asr.model_name,
                        "processing_seconds": _round_seconds(time.monotonic() - started),
                    }
                )
            )
        return events

    def finish(self) -> Dict[str, Any]:
        if self._finished_event is not None:
            return self._finished_event

        if self.closed:
            self._finished_event = self._finished_payload()
            return self._finished_event

        self.closed = True
        self.updated_at = time.time()
        try:
            self._flush_transcript_stabilizer()
            self._write_audio()
            self.engines.asr.reset_state(self.session_id)
            self.engines.diarization.reset_state(self.session_id)
        finally:
            try:
                self._finished_event = self._append_event(self._finished_payload())
            finally:
                self._release_owner()
        return self._finished_event

    def finish_events(self) -> List[Dict[str, Any]]:
        if self._finished_event is not None:
            return [self._finished_event]
        event_count = len(self._events)
        self.finish()
        return self._events[event_count:]

    @property
    def duration_seconds(self) -> float:
        return self._sample_count / float(self.config.sample_rate)

    @property
    def models_payload(self) -> Dict[str, str]:
        return {
            "streaming_asr": self.engines.asr.model_name,
            "streaming_diarization": self.engines.diarization.model_name,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "event_type": "session.snapshot",
            "session_id": self.session_id,
            "status": "closed" if self.closed else "active",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_seconds": _round_seconds(self.duration_seconds),
            "sample_rate": self.config.sample_rate,
            "transcript_event_count": len(self._events),
            "transcript_jsonl_path": str(self.events_path),
            "models": self.models_payload,
        }

    def _append_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event_index = self._event_index
        self._event_index += 1
        event_with_defaults = {
            "schema_version": STREAMING_EVENT_SCHEMA_VERSION,
            "event_id": f"{self.session_id}:stream:{event_index:06d}",
            "transcript_event_index": event_index,
            "session_id": self.session_id,
            "emitted_at_epoch": time.time(),
            **event,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event_with_defaults, sort_keys=True) + "\n")
        self._events.append(event_with_defaults)
        return event_with_defaults

    def _write_audio(self) -> None:
        self.audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(b"".join(self._pcm_chunks))

    def _finished_payload(self) -> Dict[str, Any]:
        return {
            "event_type": "transcript.session.finished",
            "status": "closed",
            "duration_seconds": _round_seconds(self.duration_seconds),
            "audio_path": str(self.audio_path),
            "transcript_jsonl_path": str(self.events_path),
            "transcript_event_count": len(self._events) + 1,
            "is_provisional": False,
            "is_final": True,
            "authority": "true_streaming_session",
            "text": "",
            "text_status": "not_applicable",
        }

    def _flush_transcript_stabilizer(self) -> None:
        if self._transcript_stabilizer is None:
            return
        stable = self._transcript_stabilizer.flush(chunk_end=self.duration_seconds)
        if stable is None or not stable.text or stable.text == self._last_text:
            return
        self._last_text = stable.text
        self._append_event(
            {
                "event_type": "transcript.segment",
                "status": "closed",
                "start": _round_seconds(stable.start),
                "end": _round_seconds(self.duration_seconds),
                "speaker": self._last_speaker,
                "text": stable.text,
                "is_provisional": False,
                "is_final": True,
                "authority": "nemo_cache_aware_streaming_asr",
                "text_status": "available",
                "asr_model": self.engines.asr.model_name,
                "processing_seconds": 0.0,
            }
        )

    def _release_owner(self) -> None:
        if self._on_close is not None:
            self._on_close(self)


class TrueStreamingSessionManager:
    def __init__(
        self,
        *,
        settings: RunnerSettings,
        engine_factory: Optional[Callable[[RunnerSettings], StreamingEnginePair]] = None,
    ) -> None:
        self.settings = settings
        self._engine_factory = engine_factory or build_streaming_engines
        self._sessions: Dict[str, Optional[TrueStreamingSession]] = {}
        self._lock = threading.Lock()

    def create_session(self, config: TrueStreamingSessionConfig) -> TrueStreamingSession:
        if not self.settings.true_streaming_enabled:
            raise UnsupportedParameterError("true streaming WebSocket path is disabled")
        _validate_streaming_config(config)
        with self._lock:
            if config.session_id in self._sessions:
                raise UnsupportedParameterError(
                    f"true streaming session {config.session_id} already active"
                )
            self._sessions[config.session_id] = None
        try:
            engines = self._engine_factory(self.settings)
            session = TrueStreamingSession(
                settings=self.settings,
                config=config,
                engines=engines,
                owner_id=uuid.uuid4().hex,
                on_close=self.release_session,
            )
            session.start()
        except Exception:
            with self._lock:
                if self._sessions.get(config.session_id) is None:
                    self._sessions.pop(config.session_id, None)
            raise
        with self._lock:
            self._sessions[config.session_id] = session
        return session

    def release_session(self, session: TrueStreamingSession) -> bool:
        with self._lock:
            if self._sessions.get(session.session_id) is not session:
                return False
            self._sessions.pop(session.session_id, None)
            return True


def build_streaming_engines(settings: RunnerSettings) -> StreamingEnginePair:
    engine = settings.true_streaming_engine.strip().lower()
    if engine not in SUPPORTED_STREAMING_ENGINES:
        raise UnsupportedParameterError(
            f"TRUE_STREAMING_ENGINE must be one of {sorted(SUPPORTED_STREAMING_ENGINES)}"
        )
    if engine == "fake":
        return StreamingEnginePair(
            asr=FakeStreamingASREngine(settings.true_streaming_asr_model),
            diarization=FakeStreamingDiarizationEngine(settings.true_streaming_diar_model),
        )
    configure_cache_environment(settings)
    return StreamingEnginePair(
        asr=NemoCacheAwareStreamingASREngine(settings),
        diarization=NemoStreamingSortformerDiarizationEngine(settings),
    )


class FakeStreamingASREngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._frames = 0

    def transcribe(self, audio: bytes, stream_id: str) -> Dict[str, Any]:
        self._frames += 1
        return {"text": f"fake streaming transcript {self._frames}", "is_final": False}

    def reset_state(self, stream_id: str) -> None:
        self._frames = 0


class FakeStreamingDiarizationEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._frames = 0

    def diarize(self, audio: bytes, stream_id: str) -> List[List[float]]:
        self._frames += 1
        speaker = (self._frames - 1) % 2
        return [[0.92 if index == speaker else 0.02 for index in range(4)]]

    def reset_state(self, stream_id: str) -> None:
        self._frames = 0


class NemoCacheAwareStreamingASREngine:
    """Minimal adapter around NeMo cache-aware streaming ASR internals."""

    def __init__(self, settings: RunnerSettings) -> None:
        import torch

        self.model_name = settings.true_streaming_asr_model
        self.device = settings.device
        self.sample_rate = settings.true_streaming_sample_rate
        self.att_context_size = list(settings.true_streaming_asr_att_context_size)
        self.chunk_size_in_secs = settings.true_streaming_asr_chunk_seconds
        self.greedy_max_symbols = settings.true_streaming_asr_greedy_max_symbols
        self.eou_tokens = set(settings.true_streaming_asr_eou_tokens)
        self._torch = torch
        self._load_model()
        self._audio_buffer = _CacheFeatureBufferer(
            sample_rate=self.sample_rate,
            buffer_size_in_secs=self._buffer_size_seconds(),
            chunk_size_in_secs=self.chunk_size_in_secs,
            preprocessor_cfg=self.asr_model.cfg.preprocessor,
            device=self.device,
        )
        self._reset_cache()
        self._previous_hypotheses = self._blank_hypothesis()

    def _load_model(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch
        from omegaconf import open_dict

        _install_rnnt_prompt_model_compat()
        if self.model_name.endswith(".nemo"):
            asr_model = nemo_asr.models.ASRModel.restore_from(
                self.model_name,
                map_location=torch.device(self.device),
            )
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name,
                map_location=torch.device(self.device),
            )
        self.decoder_type = "rnnt" if hasattr(asr_model, "joint") else "ctc"
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = True
            if hasattr(asr_model, "joint"):
                decoding_cfg.greedy.max_symbols = int(self.greedy_max_symbols)
                decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)
        asr_model.eval()
        self.asr_model = asr_model
        self.tokenizer = asr_model.tokenizer
        self.blank_id = len(self.tokenizer.vocab)

    def _buffer_size_seconds(self) -> float:
        window_stride = float(self.asr_model.cfg.preprocessor.window_stride)
        model_stride = int(self.asr_model.cfg.encoder.subsampling_factor)
        model_chunk_size = self.asr_model.encoder.streaming_cfg.chunk_size
        if isinstance(model_chunk_size, list):
            model_chunk_size = model_chunk_size[1]
        pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, list):
            pre_encode_cache_size = pre_encode_cache_size[1]
        tokens_per_frame = math.ceil(math.trunc(self.chunk_size_in_secs / window_stride) / model_stride)
        self.asr_model.encoder.setup_streaming_params(
            chunk_size=int(model_chunk_size) // model_stride,
            shift_size=tokens_per_frame,
        )
        return float(pre_encode_cache_size) * window_stride + float(model_chunk_size) * window_stride

    def _reset_cache(self) -> None:
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = self.asr_model.encoder.get_initial_cache_state(1)

    def _blank_hypothesis(self) -> List[Any]:
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        return [Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)]

    def transcribe(self, audio: bytes, stream_id: str) -> Dict[str, Any]:
        import numpy as np
        import torch

        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.update(audio_array)
        features = self._audio_buffer.get_feature_buffer()
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)
        features = features.unsqueeze(0)
        with torch.no_grad():
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.asr_model.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded,
            )
        if self.decoder_type == "ctc":
            best_hyp = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
            )
        else:
            best_hyp = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
                partial_hypotheses=self._previous_hypotheses,
            )
        self._previous_hypotheses = best_hyp
        self._cache_last_channel = cache_last_channel
        self._cache_last_time = cache_last_time
        self._cache_last_channel_len = cache_last_channel_len
        token_ids = self._token_ids_from_alignments(best_hyp[0].alignments)
        token_pieces = self.tokenizer.ids_to_tokens(token_ids) if token_ids else []
        visible_token_pieces, is_final = _strip_eou_token_pieces(
            token_pieces,
            self.eou_tokens,
        )
        if is_final:
            self.reset_state(stream_id)
        return {
            "text": self._text_from_token_pieces(visible_token_pieces),
            "token_pieces": visible_token_pieces,
            "is_final": is_final,
        }

    def reset_state(self, stream_id: str) -> None:
        self._audio_buffer.reset()
        self._reset_cache()
        self._previous_hypotheses = self._blank_hypothesis()

    def _text_from_alignments(self, alignments: Any) -> str:
        return self._text_from_token_pieces(
            self.tokenizer.ids_to_tokens(self._token_ids_from_alignments(alignments))
        )

    def _token_ids_from_alignments(self, alignments: Any) -> List[int]:
        token_ids: List[int] = []
        if self.decoder_type == "ctc":
            for token_id in alignments[1]:
                token = int(token_id)
                if token != self.blank_id:
                    token_ids.append(token)
        else:
            for timestep in alignments:
                for _, token_id in timestep:
                    token = int(token_id)
                    if token != self.blank_id:
                        token_ids.append(token)
        return token_ids

    def _text_from_token_pieces(self, pieces: Sequence[str]) -> str:
        if not pieces:
            return ""
        separator = "\u2581"
        return "".join(piece.replace(separator, " ") if piece.startswith(separator) else piece for piece in pieces)


def _install_rnnt_prompt_model_compat() -> None:
    """Register the RNNT prompt class expected by newer Nemotron checkpoints.

    NeMo 2.6.2 ships the hybrid RNNT/CTC prompt model, but not the RNNT-only
    prompt model referenced by nvidia/nemotron-3.5-asr-streaming-0.6b. The true
    streaming path uses the encoder, joint, tokenizer, and RNNT decoding objects
    directly; this shim supplies the missing prompt projection module so the
    checkpoint can restore cleanly without changing the default model.
    """
    import sys
    import types

    module_name = "nemo.collections.asr.models.rnnt_bpe_models_prompt"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name, fromlist=["EncDecRNNTBPEModelWithPrompt"])
        return
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    import torch
    from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
    from pytorch_lightning import Trainer

    from nemo.collections.asr.metrics.wer import WER
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
    from nemo.collections.asr.parts.mixins import ASRTranscriptionMixin
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
    from nemo.utils import logging, model_utils

    class EncDecRNNTBPEModelWithPrompt(EncDecRNNTBPEModel, ASRTranscriptionMixin):
        def __init__(self, cfg: DictConfig, trainer: Trainer = None):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            cfg = model_utils.maybe_update_config_version(cfg)
            if "tokenizer" not in cfg:
                raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
            if not isinstance(cfg, DictConfig):
                cfg = OmegaConf.create(cfg)

            self._setup_tokenizer(cfg.tokenizer)
            vocabulary = self.tokenizer.tokenizer.get_vocab()
            with open_dict(cfg):
                cfg.labels = ListConfig(list(vocabulary))
                cfg.num_prompts = cfg.model_defaults.get("num_prompts", 128)
                if "prompt_dictionary" not in cfg.model_defaults:
                    logging.warning("No prompt_dictionary in config; using empty dict.")
                    cfg.model_defaults.prompt_dictionary = {}
                self.subsampling_factor = cfg.get("subsampling_factor", 8)
            with open_dict(cfg.decoder):
                cfg.decoder.vocab_size = len(vocabulary)
            with open_dict(cfg.joint):
                cfg.joint.num_classes = len(vocabulary)
                cfg.joint.vocabulary = ListConfig(list(vocabulary))
                cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
                cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

            super().__init__(cfg=cfg, trainer=trainer)
            self.concat = False
            if self.cfg.model_defaults.get("initialize_prompt_feature", False):
                self.initialize_prompt_feature()

        def initialize_prompt_feature(self) -> None:
            logging.info("Model with prompt feature has been initialized (RNNT-only compat)")
            self.concat = True
            self.num_prompts = self.cfg.get("num_prompts", 128)
            proj_in_size = self.num_prompts + self._cfg.model_defaults.enc_hidden
            proj_out_size = self._cfg.model_defaults.enc_hidden
            self.prompt_kernel = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size * 2, proj_out_size),
            )
            self.decoding = RNNTBPEDecoding(
                decoding_cfg=self.cfg.decoding,
                decoder=self.decoder,
                joint=self.joint,
                tokenizer=self.tokenizer,
            )
            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=0,
                use_cer=self.cfg.get("use_cer", False),
                log_prediction=self.cfg.get("log_prediction", True),
                dist_sync_on_step=True,
            )
            if self.joint.fuse_loss_wer:
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

    module = types.ModuleType(module_name)
    module.EncDecRNNTBPEModelWithPrompt = EncDecRNNTBPEModelWithPrompt
    sys.modules[module_name] = module


class NemoStreamingSortformerDiarizationEngine:
    """Minimal adapter around NeMo streaming Sortformer diarization internals."""

    def __init__(self, settings: RunnerSettings) -> None:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model_name = settings.true_streaming_diar_model
        self.device = settings.device
        self.sample_rate = settings.true_streaming_sample_rate
        self.frame_seconds = settings.true_streaming_diar_frame_seconds
        self.max_speakers = 4
        if self.model_name.endswith(".nemo"):
            model = SortformerEncLabelModel.restore_from(self.model_name, map_location=self.device)
        else:
            model = SortformerEncLabelModel.from_pretrained(self.model_name, map_location=self.device)
        model.sortformer_modules.chunk_len = 6
        model.sortformer_modules.spkcache_len = 188
        model.sortformer_modules.chunk_left_context = 1
        model.sortformer_modules.chunk_right_context = 7
        model.sortformer_modules.fifo_len = 188
        model.sortformer_modules.log = False
        if hasattr(model.sortformer_modules, "spkcache_refresh_rate"):
            model.sortformer_modules.spkcache_refresh_rate = 144
        elif hasattr(model.sortformer_modules, "spkcache_update_period"):
            model.sortformer_modules.spkcache_update_period = 300
        model.eval()
        self.model = model
        self.chunk_size = int(model.sortformer_modules.chunk_len)
        self.feature_bufferer = _CacheFeatureBufferer(
            sample_rate=self.sample_rate,
            buffer_size_in_secs=self.chunk_size * self.frame_seconds + 0.16,
            chunk_size_in_secs=self.chunk_size * self.frame_seconds,
            preprocessor_cfg=model.cfg.preprocessor,
            device=self.device,
        )
        self.streaming_state = self.model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=self.model.async_streaming,
            device=self.device,
        )
        self.total_preds = torch.zeros((1, 0, self.max_speakers), device=self.model.device)

    def diarize(self, audio: bytes, stream_id: str) -> Any:
        import numpy as np
        import torch

        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self.feature_bufferer.update(audio_array)
        features = self.feature_bufferer.get_feature_buffer()
        feature_buffers = features.unsqueeze(0).transpose(1, 2)
        feature_buffer_lens = torch.tensor([feature_buffers.shape[1]], device=self.device)
        with torch.inference_mode(), torch.no_grad():
            self.streaming_state, diar_pred_out_stream = self.model.forward_streaming_step(
                processed_signal=feature_buffers,
                processed_signal_length=feature_buffer_lens,
                streaming_state=self.streaming_state,
                total_preds=self.total_preds,
                left_offset=8,
                right_offset=8,
            )
        self.total_preds = diar_pred_out_stream
        return diar_pred_out_stream[:, -self.chunk_size :, :].clone().cpu().numpy()[0]

    def reset_state(self, stream_id: str) -> None:
        import torch

        self.feature_bufferer.reset()
        self.streaming_state = self.model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=self.model.async_streaming,
            device=self.device,
        )
        self.total_preds = torch.zeros((1, 0, self.max_speakers), device=self.model.device)


class _AudioBufferer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float) -> None:
        import torch

        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)

    def reset(self) -> None:
        self.sample_buffer.zero_()

    def update(self, audio: Any) -> None:
        import torch

        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        if audio.shape[0] > self.buffer_size:
            raise ValueError(f"Frame size ({audio.shape[0]}) exceeds buffer size ({self.buffer_size})")
        self.sample_buffer[:- audio.shape[0]] = self.sample_buffer[audio.shape[0] :].clone()
        self.sample_buffer[-audio.shape[0] :] = audio.clone()


class _CacheFeatureBufferer:
    def __init__(
        self,
        *,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg: Any,
        device: str,
    ) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.device = device
        self.zero_level = -16.635 if getattr(preprocessor_cfg, "log", False) else 0.0
        self.timestep_duration = float(preprocessor_cfg.window_stride)
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.sample_buffer = _AudioBufferer(sample_rate, buffer_size_in_secs)
        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [int(preprocessor_cfg.features), self.feature_buffer_len],
            self.zero_level,
            dtype=torch.float32,
            device=self.device,
        )
        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)

    def reset(self) -> None:
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.zero_level)

    def update(self, audio: Any) -> None:
        self.sample_buffer.update(audio)
        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            samples = self.sample_buffer.sample_buffer.clone()
        else:
            samples = self.sample_buffer.sample_buffer[-(self.n_chunk_look_back + self.chunk_size) :]
        features = self._preprocess(samples)
        diff = features.shape[1] - self.feature_chunk_len - 1
        if diff > 0:
            features = features[:, :-diff]
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[
            :,
            self.feature_chunk_len :,
        ].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = features[:, -self.feature_chunk_len :].clone()

    def get_feature_buffer(self) -> Any:
        return self.feature_buffer.clone()

    def _preprocess(self, audio_signal: Any) -> Any:
        import torch

        audio_signal = audio_signal.unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(input_signal=audio_signal, length=audio_signal_len)
        return features.squeeze()


@dataclass(frozen=True)
class _StableTranscriptSegment:
    start: float
    text: str


class _StreamingTranscriptStabilizer:
    """Coalesce NeMo streaming token pieces into larger provisional text deltas."""

    _PUNCTUATION = {".", ",", "?", "!", ":", ";", "%", ")", "]", "}"}
    _OPENING_PUNCTUATION = {"(", "[", "{"}
    _WORD_BOUNDARY = "\u2581"

    def __init__(
        self,
        *,
        min_words: int,
        min_chars: int,
        max_hold_seconds: float,
    ) -> None:
        self.min_words = max(1, int(min_words))
        self.min_chars = max(1, int(min_chars))
        self.max_hold_seconds = max(0.0, float(max_hold_seconds))
        self._pending_word = ""
        self._parts: List[str] = []
        self._segment_start: Optional[float] = None

    def update(
        self,
        text: str,
        *,
        chunk_start: float,
        chunk_end: float,
    ) -> Optional[_StableTranscriptSegment]:
        if not text:
            return self._maybe_emit(chunk_end=chunk_end, force=False)
        if self._segment_start is None:
            self._segment_start = chunk_start

        starts_new_word = text[:1].isspace()
        tokens = text.strip().split()
        if not tokens:
            return self._maybe_emit(chunk_end=chunk_end, force=False)

        for index, token in enumerate(tokens):
            if index == 0 and not starts_new_word and self._pending_word:
                self._pending_word += token
                continue
            if index == 0 and not starts_new_word and not self._pending_word:
                self._pending_word = token
                continue
            self._complete_pending_word()
            self._accept_new_token(token)
        return self._maybe_emit(chunk_end=chunk_end, force=False)

    def update_token_pieces(
        self,
        token_pieces: Sequence[str],
        *,
        chunk_start: float,
        chunk_end: float,
    ) -> Optional[_StableTranscriptSegment]:
        if self._segment_start is None:
            self._segment_start = chunk_start

        accepted_piece = False
        for piece in token_pieces:
            accepted_piece = self._accept_token_piece(piece) or accepted_piece
        if not accepted_piece:
            return self._maybe_emit(chunk_end=chunk_end, force=False)
        return self._maybe_emit(chunk_end=chunk_end, force=False)

    def flush(self, *, chunk_end: float = 0.0) -> Optional[_StableTranscriptSegment]:
        self._complete_pending_word()
        return self._maybe_emit(chunk_end=chunk_end, force=True)

    def _accept_token_piece(self, piece: str) -> bool:
        if not piece:
            return False
        starts_new_word = piece.startswith(self._WORD_BOUNDARY) or piece[:1].isspace()
        token = piece.replace(self._WORD_BOUNDARY, " ").strip()
        if not token:
            return False
        if token in self._PUNCTUATION:
            self._complete_pending_word()
            self._accept_new_token(token)
            return True
        if token in self._OPENING_PUNCTUATION:
            self._complete_pending_word()
            self._accept_new_token(token)
            return True
        if starts_new_word:
            self._complete_pending_word()
            self._accept_new_token(token)
            return True
        if self._pending_word:
            self._pending_word += token
            return True
        self._pending_word = token
        return True

    def _accept_new_token(self, token: str) -> None:
        if not token:
            return
        if token in self._PUNCTUATION:
            if self._parts:
                self._parts[-1] = f"{self._parts[-1]}{token}"
            else:
                self._parts.append(token)
            return
        if token in self._OPENING_PUNCTUATION:
            self._parts.append(token)
            return
        self._pending_word = token

    def _complete_pending_word(self) -> None:
        if not self._pending_word:
            return
        self._parts.append(self._pending_word)
        self._pending_word = ""

    def _maybe_emit(
        self,
        *,
        chunk_end: float,
        force: bool,
    ) -> Optional[_StableTranscriptSegment]:
        if not self._parts:
            return None
        text = _polish_streaming_text(" ".join(self._parts))
        if not text:
            self._parts = []
            return None
        word_count = len(re.findall(r"[A-Za-z0-9]+", text))
        held_seconds = 0.0
        if self._segment_start is not None:
            held_seconds = max(0.0, float(chunk_end) - float(self._segment_start))
        should_emit = (
            force
            or word_count >= self.min_words
            or len(text) >= self.min_chars
            or held_seconds >= self.max_hold_seconds
        )
        if not should_emit:
            return None
        segment = _StableTranscriptSegment(start=self._segment_start or 0.0, text=text)
        self._parts = []
        self._segment_start = None if not self._pending_word else chunk_end
        return segment


def _polish_streaming_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b([A-Za-z]+)\s+'(s|re|ve|ll|d|m|t)\b", r"\1'\2", text)
    text = re.sub(
        r"\b(?:[A-Z]\s+){1,}[A-Z]\b",
        lambda match: match.group(0).replace(" ", ""),
        text,
    )
    text = re.sub(r"\s+([.,?!:;%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text


def _validate_streaming_config(config: TrueStreamingSessionConfig) -> None:
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
    if config.max_speakers <= 0 or config.max_speakers > 4:
        raise UnsupportedParameterError("true streaming Sortformer supports max_speakers from 1 to 4")
    if config.sample_rate != 16000:
        raise UnsupportedParameterError("true streaming WebSocket audio must be 16 kHz PCM")


def _asr_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("text") or "")
    return str(getattr(result, "text", "") or "")


def _asr_token_pieces(result: Any) -> Optional[List[str]]:
    if not isinstance(result, dict):
        return None
    pieces = result.get("token_pieces")
    if pieces is None:
        return None
    if not isinstance(pieces, list):
        return None
    return [str(piece) for piece in pieces]


def _strip_eou_token_pieces(
    token_pieces: Sequence[str],
    eou_tokens: Sequence[str],
) -> tuple[List[str], bool]:
    eou_token_set = set(eou_tokens)
    if not eou_token_set:
        return [str(piece) for piece in token_pieces], False
    visible = [str(piece) for piece in token_pieces if str(piece) not in eou_token_set]
    is_final = len(visible) != len(token_pieces)
    return visible, is_final


def _asr_is_final(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("is_final", False))
    return bool(getattr(result, "is_final", False))


def _speaker_payload_from_diarization(
    diarization: Any,
    *,
    chunk_start: float,
    frame_seconds: float,
    threshold: float,
) -> Dict[str, Any]:
    frames = _diarization_frames(diarization)
    if not frames:
        return {}
    scores = [0.0 for _ in range(max(len(frame) for frame in frames))]
    for frame in frames:
        for index, value in enumerate(frame):
            scores[index] += float(value)
    frame_count = float(len(frames))
    averages = [score / frame_count for score in scores]
    best_index = max(range(len(averages)), key=lambda index: averages[index])
    best_score = averages[best_index]
    return {
        "start": _round_seconds(chunk_start),
        "end": _round_seconds(chunk_start + len(frames) * frame_seconds),
        "speaker": f"speaker_{best_index}" if best_score >= threshold else None,
        "speaker_confidence": _round_seconds(best_score),
        "speaker_probabilities": [_round_seconds(value) for value in averages],
    }


def _diarization_frames(value: Any) -> List[List[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    frames: List[List[float]] = []
    for frame in value:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        if isinstance(frame, (list, tuple)):
            frames.append([float(item) for item in frame])
    return frames


def pcm16_sine_wave(*, sample_rate: int, seconds: float, frequency: float = 440.0) -> bytes:
    sample_count = int(sample_rate * seconds)
    samples = array(
        "h",
        [
            int(0.25 * 32767 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            for index in range(sample_count)
        ],
    )
    return samples.tobytes()
