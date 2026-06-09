# Livepeer NeMo True Streaming Remote GPU Spec

## Summary

This is the canonical PoC spec for the greenfield NeMo diarized transcription
runtime: local Livepeer/Roboflow control planes own session intent, routing,
events, billing, and artifacts, while a remote GPU executor owns the long-lived
NeMo streaming inference process.

This spec synthesizes the agreed greenfield lanes:

- Codex `00fa89a6-8643-4dd4-9ff5-e867f741acb0`
- Claude `c8272754-b303-4fd4-83cd-732a887f2097`
- Grok `f357f28e-4c05-4e85-8622-da30643c0768`
- cursor/default `73963bf0-087e-4914-8374-084980f070a5`

The implementation target for the PoC container is
`audio-diarized-transcription-runner`. `livepeer-roboflow` should only carry
additive docs or integration references until this runner surface is promoted
into a broader workflow runtime.

## Current Compatibility Contract

The existing paths remain intact:

- `POST /v1/audio/transcriptions` is the intended bounded
  OpenAI-compatible request path backed by the same native
  `audio:diarized-transcription@v0` capability.
- `POST /v1/audio/diarized-transcriptions` remains as the legacy/native
  multipart request path.
- `/v1/audio/diarized-transcriptions/live/sessions/*` remains the additive
  REST session path using online diarization with chunk ingest and optional
  final offline transcription.
- The true streaming path is adjacent to the bounded API:
  `WS /v1/audio/transcriptions/stream`.
- `WS /v1/audio/diarized-transcriptions/stream` remains as a compatibility
  alias for existing PoC clients.

No existing request shape, response shape, capability id, or model default is
removed for the current PoC.

## Architecture

### Control Plane

Local control remains authoritative for:

- session creation and termination
- capability/model selection
- admission control and future Livepeer payment checks
- incremental event persistence
- final artifact location
- user-visible status and error envelopes

The control plane should be able to run on a non-GPU host. It connects to a
GPU-capable executor through a persistent stream transport.

### Remote GPU Executor

The GPU executor runs one long-lived NeMo session per user stream. It must keep
model state in memory across frames:

- cache-aware streaming ASR state
- streaming Sortformer speaker cache/state
- per-session audio timeline counters
- incremental transcript event counters

The executor must not treat each incoming chunk as an independent transcription
job. Audio frames are input to one persistent session until the client sends a
finish control message or disconnects.

### Transport

The PoC transport is WebSocket:

- endpoint: `WS /v1/audio/transcriptions/stream`
- binary messages: little-endian 16 kHz mono int16 PCM frames
- JSON control messages:
  - `{"type":"ping"}`
  - `{"type":"finish"}`

The transport is intentionally simple and maps cleanly to a future remote GPU
executor connection. A future brokered version can wrap the same binary frames
and JSON event envelope in Livepeer-specific authorization/payment metadata.

## NeMo Stack

The preferred true streaming stack is based on the checked-in NeMo references:

- `livepeer-roboflow/references/NeMo/examples/voice_agent/README.md`
- `livepeer-roboflow/references/NeMo/examples/asr/asr_streaming_inference/README.md`
- `livepeer-roboflow/references/NeMo/examples/speaker_tasks/diarization/README.md`

Default PoC models:

- ASR: `nvidia/nemotron-speech-streaming-en-0.6b`
- diarization: `nvidia/diar_streaming_sortformer_4spk-v2.1`

The ASR path uses NeMo cache-aware streaming FastConformer-style internals
(`encoder.cache_aware_stream_step`) rather than whole-file `transcribe()` calls.
The diarization path uses streaming Sortformer (`forward_streaming_step`) with
speaker cache state preserved for the WebSocket session.

## Event Envelope

Every emitted event is JSON and includes:

- `schema_version`
- `event_id`
- `transcript_event_index`
- `session_id`
- `event_type`
- `status`
- `emitted_at_epoch`
- `is_provisional`
- `is_final`
- `authority`

PoC event types:

- `session.snapshot`
- `audio.frame.received`
- `speaker.update`
- `transcript.segment`
- `transcript.session.finished`

`transcript.segment` events are emitted before session finish when streaming
ASR returns non-empty text. `speaker.update` events are emitted from in-stream
Sortformer predictions.

## Settings

Additive environment variables:

- `TRUE_STREAMING_ENABLED=true`
- `TRUE_STREAMING_ENGINE=nemo`
- `TRUE_STREAMING_ASR_MODEL=nvidia/nemotron-speech-streaming-en-0.6b`
- `TRUE_STREAMING_DIAR_MODEL=nvidia/diar_streaming_sortformer_4spk-v2.1`
- `TRUE_STREAMING_SAMPLE_RATE=16000`
- `TRUE_STREAMING_ASR_ATT_CONTEXT_SIZE=70,1`
- `TRUE_STREAMING_ASR_CHUNK_SECONDS=0.08`
- `TRUE_STREAMING_ASR_STABILIZE_PARTIALS=true`
- `TRUE_STREAMING_ASR_EMIT_MIN_WORDS=6`
- `TRUE_STREAMING_ASR_EMIT_MIN_CHARS=28`
- `TRUE_STREAMING_ASR_EMIT_MAX_HOLD_SECONDS=1.2`
- `TRUE_STREAMING_DIAR_FRAME_SECONDS=0.08`
- `TRUE_STREAMING_DIAR_THRESHOLD=0.4`
- `TRUE_STREAMING_MAX_SESSION_SECONDS=3600`

`TRUE_STREAMING_ENGINE=fake` is only for local smoke tests and CI. It must not
be advertised as production inference.

The partial stabilizer only batches visible ASR token pieces into larger
provisional text deltas. It is not a semantic merge layer. The 2026-06-08 real
30s benchmark reduced transcript segments from 64 to 18 with stabilization
enabled, but the stabilized output still had 8 transcript segments with two
words or fewer. Runtime benchmark artifacts should remain local and out of git.

## Acceptance Criteria

The PoC is acceptable when:

- one WebSocket connection owns one persistent session
- binary audio frames are processed incrementally over that connection
- at least one `transcript.segment` event can be emitted before finish
- at least one `speaker.update` event can be emitted before finish
- the existing offline and REST live endpoints still pass their tests
- Docker/Compose can build and run the GPU-capable image
- docs and smoke scripts explain both fake-engine and NeMo-engine validation

## Non-Goals

This PoC does not:

- remove the existing chunk-based live REST session path
- remove the final offline transcription path
- implement Livepeer payment enforcement inside the runner
- implement distributed executor discovery
- guarantee production diarization quality for overlapping speakers

## Rollout

1. Ship the WebSocket PoC in the runner image behind additive settings.
2. Validate fake-engine transport smoke locally.
3. Validate NeMo model loading and incremental events on a CUDA host.
4. Add Livepeer/Roboflow integration docs pointing to the WebSocket surface.
5. Promote the same event envelope into the remote GPU executor protocol.
