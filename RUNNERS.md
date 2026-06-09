# Runner Reference

## `audio-diarized-transcription-runner`

Serves the Livepeer `audio:diarized-transcription@v0` capability over HTTP.
The runner accepts one multipart audio upload, decodes it with `ffmpeg`, converts
it to 16 kHz mono WAV, and runs NVIDIA NeMo offline diarization-with-ASR.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Runtime readiness and CUDA facts |
| `GET` | `/audio:diarized-transcription@v0/options` | Capability defaults and limits |
| `POST` | `/v1/audio/transcriptions` | Main OpenAI-compatible bounded transcription request backed by the native diarized transcription capability |
| `POST` | `/v1/audio/diarized-transcriptions/live/sessions` | Create a stateful online diarization session |
| `POST` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}/audio` | Ingest one audio chunk into a live session |
| `GET` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}` | Read the latest live diarization snapshot |
| `POST` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}/finish` | Close a live session, optionally running final offline ASR+diarization |
| `DELETE` | `/v1/audio/diarized-transcriptions/live/sessions/{session_id}` | Close a live session without final transcription |
| `WS` | `/v1/audio/transcriptions/stream` | True streaming PCM16 ASR+diarization over one persistent transport |
| `GET` | `/metrics` | Prometheus text metrics when `METRICS_ENABLED=true` |

### Request

`POST /v1/audio/transcriptions` requires multipart form data:

| Field | Required | Default | Notes |
|---|---:|---|---|
| `file` | yes |  | Audio file. Anything decodable by `ffmpeg` is accepted. |
| `model` | no | `nemo-diarized-transcription-meeting-v0` | Logical model id exposed to Livepeer. |
| `language` | no | `en` | Only English is supported in this phase. |
| `preset` | no | `meeting` | Only the meeting preset is supported in this phase. |
| `num_speakers` | no |  | Exact speaker count, if known. |
| `max_speakers` | no | `8` | Upper bound for clustering. |
| `response_format` | no | `json` | `json`, `verbose_json`, `text`, `srt`, or `vtt`. |
| `timestamp_granularities[]` | no | `segment` | `segment`, `word`, or both; also accepts comma-separated `timestamp_granularities`. |
| `diarization` / `include_diarization` | no | `false` | Include speaker labels and the generic top-level diarization extension in `verbose_json`. |
| `include_words` | no | `false` | Include normalized word timestamps in verbose JSON. |
| `include_artifacts` | no | `false` | Include local artifact paths in verbose JSON. |

Unsupported language, preset, or response format returns a structured `422`.
Oversized uploads return `413`. A full local queue returns `429`. CUDA OOM
returns `507`.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `RUNNER_PORT` | `8080` | HTTP listen port |
| `DEVICE` | `cuda` | `cuda` or `cpu`; CUDA is expected for production |
| `CAPABILITY_NAME` | `audio:diarized-transcription@v0` | Capability advertised in responses |
| `METRICS_ENABLED` | `false` | Expose `/metrics` |
| `MAX_QUEUE_SIZE` | `1` | Concurrent in-process request slots |
| `MAX_AUDIO_MB` | `100` | Upload limit |
| `MODEL_CACHE_DIR` | `/models` | NeMo, Hugging Face, and torch cache root |
| `WORK_DIR` | `/tmp/audio-diarized-transcription-runner` | Per-request uploads and artifacts |
| `DEFAULT_VAD_MODEL` | `vad_multilingual_marblenet` | NeMo VAD model |
| `DEFAULT_SPEAKER_MODEL` | `titanet_large` | NeMo speaker embedding model |
| `DEFAULT_ASR_MODEL` | `stt_en_conformer_ctc_large` | NeMo English ASR model |
| `LIVE_PROVISIONAL_ASR_MODEL` | `stt_en_conformer_ctc_large` | NeMo English ASR model used for live-session provisional chunk text during ingest |
| `LIVE_FINAL_ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo English ASR model used only by live-session final transcription |
| `LIVE_FINAL_ASR_BATCH_SIZE` | `4` | Batch size passed to final native timestamp ASR |
| `LIVE_FINAL_ASR_DECODING_STRATEGY` |  | Optional NeMo RNNT/TDT decoding strategy override for final ASR |
| `LIVE_FINAL_ASR_BEAM_SIZE` |  | Optional beam size when the final decoding strategy supports beam search |
| `TRUE_STREAMING_ENGINE` | `nemo` | `nemo` for real streaming models, `fake` for transport smoke tests |
| `TRUE_STREAMING_ASR_MODEL` | `nvidia/nemotron-speech-streaming-en-0.6b` | NeMo cache-aware streaming ASR model for the WebSocket path |
| `TRUE_STREAMING_DIAR_MODEL` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | NeMo streaming Sortformer diarization model for the WebSocket path |
| `TRUE_STREAMING_SAMPLE_RATE` | `16000` | Required WebSocket PCM sample rate |
| `TRUE_STREAMING_ASR_ATT_CONTEXT_SIZE` | `70,1` | Cache-aware ASR attention context |
| `TRUE_STREAMING_ASR_CHUNK_SECONDS` | `0.08` | ASR frame size for cache-aware streaming |
| `TRUE_STREAMING_ASR_GREEDY_MAX_SYMBOLS` | `16` | NeMo RNNT greedy max symbols per step for streaming decode |
| `TRUE_STREAMING_ASR_EOU_TOKENS` | `<EOU>,<EOB>` | ASR marker tokens that finalize a turn and are stripped from visible text |
| `TRUE_STREAMING_ASR_STABILIZE_PARTIALS` | `true` | Hold NeMo partial token pieces until phrase-sized provisional text is stable enough to emit |
| `TRUE_STREAMING_ASR_EMIT_MIN_WORDS` | `6` | Minimum completed words before emitting a stabilized streaming transcript segment |
| `TRUE_STREAMING_ASR_EMIT_MIN_CHARS` | `28` | Minimum completed characters before emitting a stabilized streaming transcript segment |
| `TRUE_STREAMING_ASR_EMIT_MAX_HOLD_SECONDS` | `1.2` | Maximum audio-timeline hold before emitting a short stabilized segment |
| `TRUE_STREAMING_DIAR_FRAME_SECONDS` | `0.08` | Sortformer frame duration |
| `TRUE_STREAMING_DIAR_THRESHOLD` | `0.4` | Minimum average Sortformer score for speaker assignment |
| `TRUE_STREAMING_MAX_SESSION_SECONDS` | `3600` | Maximum accepted WebSocket session duration |
| `DEFAULT_MAX_SPEAKERS` | `8` | Default clustering speaker upper bound |
| `DEFAULT_PRESET` | `meeting` | Default preset |
| `NEMO_DIARIZER_CONFIG` | packaged config | Optional override for the NeMo diarization YAML |

### Response Shape

JSON responses include:

- `id`, `status`, `capability`, and `mode`
- `source` filename/content type
- `models` used for VAD, speaker embeddings, and ASR
- `duration_seconds`, `text`, and `speaker_count`
- `speakers`, `segments`, and optional `words`
- optional artifact paths for NeMo JSON/RTTM/CTM/TXT/Gecko plus generated SRT/VTT
- `usage.audio_seconds` and `usage.work_units`

`text`, `srt`, and `vtt` response formats return plain text.

### OpenAI-Compatible Transcriptions

`POST /v1/audio/transcriptions` is the intended bounded request route for
clients that already use the OpenAI audio transcription API. It does not change
the runner capability id; requests are still served by
`audio:diarized-transcription@v0` and the same NeMo diarization stack.

The route accepts multipart form data:

| Field | Required | Default | Notes |
|---|---:|---|---|
| `file` | yes |  | Audio file. Anything decodable by `ffmpeg` is accepted. |
| `model` | no | `nemo-diarized-transcription-meeting-v0` | Logical model id exposed to Livepeer. |
| `language` | no | `en` | Only English is supported in this phase. |
| `response_format` | no | `json` | `json`, `verbose_json`, `text`, `srt`, or `vtt`. |
| `timestamp_granularities[]` | no | `segment` | `segment`, `word`, or both; also accepts comma-separated `timestamp_granularities`. |
| `prompt` | no |  | Accepted for OpenAI request-shape compatibility and ignored. |
| `temperature` | no |  | Accepted for OpenAI request-shape compatibility and ignored. |
| `preset` | no | `meeting` | Livepeer extension field for the native diarization preset. |
| `num_speakers` | no |  | Livepeer extension field for exact speaker count, if known. |
| `max_speakers` | no | `8` | Livepeer extension field for clustering upper bound. |
| `diarization` / `include_diarization` | no | `false` | Include speaker labels and the generic top-level diarization extension in `verbose_json`. |
| `include_words` | no | `false` | Include native word timestamps when building verbose responses; `timestamp_granularities[]=word` also enables this. |
| `include_artifacts` | no | `false` | Include local artifact paths under the Livepeer extension object. |

`json`, `text`, `srt`, and `vtt` follow OpenAI-style response formats.
`verbose_json` includes OpenAI `segments` and `words` when requested. When
`diarization=true` or `include_diarization=true`, it also adds generic extension
fields:

- `diarization`
- `speaker_labeled_text`
- `usage`
- `models`
- `artifacts`
- `speaker` on emitted segment and word entries

The bounded route remains the only public file/request transcription contract.

### Live Session Path

The live path is additive and keeps one NeMo `OnlineClusteringDiarizer` instance
per session. Audio chunks are normalized to 16 kHz mono WAV, appended to the
session timeline, diarized with a rolling audio buffer, and returned as
cumulative speaker turn events. The default live VAD strategy is an energy VAD
that feeds absolute speech intervals into NeMo; callers that already have VAD
can create the session with `vad_strategy=provided` and pass
`vad_segments_json` on each chunk upload.

Create a session:

```bash
curl -s http://localhost:8080/v1/audio/diarized-transcriptions/live/sessions \
  -H 'content-type: application/json' \
  -d '{"session_id":"live_demo","max_speakers":4,"vad_strategy":"energy"}' | jq
```

Ingest one chunk:

```bash
curl -s http://localhost:8080/v1/audio/diarized-transcriptions/live/sessions/live_demo/audio \
  -F file=@chunk-000.wav \
  -F sequence_index=0 | jq
```

Finish without ASR:

```bash
curl -s http://localhost:8080/v1/audio/diarized-transcriptions/live/sessions/live_demo/finish \
  -H 'content-type: application/json' \
  -d '{"run_final_transcription":false}' | jq
```

Near-live responses include provisional chunk ASR text and speaker turn events
with `is_provisional=true`. The authoritative transcript path remains
`run_final_transcription=true` on finish, which reuses the existing verified
offline NeMo ASR+diarization path over the accumulated session WAV.
Live-session provisional text uses `LIVE_PROVISIONAL_ASR_MODEL`, and final
transcription uses `LIVE_FINAL_ASR_MODEL`, which defaults to
`nvidia/parakeet-tdt-0.6b-v3` while preserving `DEFAULT_ASR_MODEL` for existing
non-live requests.

The final-ASR path requires the NeMo 2.6.x runtime in this image. NeMo 2.0.0
loads `nvidia/parakeet-tdt-0.6b-v3`, but its RNN-T/TDT `transcribe()` method
does not accept `timestamps=True`, and the older `ASRDecoderTimeStamps` helper
only covers CTC-style timestamp flows. This runner uses native Parakeet
Hypothesis word timestamps for TDT/RNN-T model names and then feeds the
resulting `word_hyp`/`word_ts_hyp` dictionaries into NeMo `OfflineDiarWithASR`
for clustering, speaker labeling, and artifact writing. Non-live direct
requests continue to use the existing `ASRDecoderTimeStamps` integration.

### True Streaming WebSocket Path

The true streaming path keeps one WebSocket connection open for the whole live
session. Binary messages must contain little-endian 16 kHz mono int16 PCM. JSON
control messages currently support `{"type":"ping"}` and `{"type":"finish"}`.
The default ASR is `nvidia/nemotron-speech-streaming-en-0.6b`. The best local
Parakeet true-streaming candidate is `nvidia/parakeet_realtime_eou_120m-v1`;
select it with `TRUE_STREAMING_ASR_MODEL=nvidia/parakeet_realtime_eou_120m-v1`.
Its `<EOU>`/`<EOB>` markers are controlled by `TRUE_STREAMING_ASR_EOU_TOKENS`,
cause the segment to be marked final, and are stripped from emitted transcript
text.
The optional ASR partial stabilizer is a churn reducer, not a complete streaming
assembler. In the 2026-06-08 30s real-stream VDO benchmark, the same current
image and input produced 64 transcript segments with stabilization disabled and
18 with stabilization enabled. One-word-or-less transcript segments dropped
from 61 to 1, but the stabilized run still emitted 8 segments with two words or
fewer.
Treat this as evidence that NeMo/model-side tuning materially improves this
fixture but does not fully solve streaming segmentation fragmentation.

Incremental responses are emitted before finish:

- `audio.frame.received`
- `speaker.update`
- `transcript.segment`
- `transcript.session.finished`

Example smoke flow with fake engines:

```bash
TRUE_STREAMING_ENGINE=fake DEVICE=cpu docker compose \
  -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up
python scripts/audio_diarized_transcription_streaming_smoke.py --require-transcript
```

Example NeMo flow on a CUDA host:

```bash
docker compose -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up
python scripts/audio_diarized_transcription_streaming_smoke.py ./sample.wav --require-transcript
```
