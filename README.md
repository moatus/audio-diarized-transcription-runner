# audio-diarized-transcription-runner

Workload binary that serves the Livepeer `audio:diarized-transcription@v0`
capability over HTTP and WebSocket transports. It exposes:

- a bounded OpenAI-compatible transcription route at `POST /v1/audio/transcriptions`
- a native bounded diarized route at `POST /v1/audio/diarized-transcriptions`
- an adjacent true streaming route at `WS /v1/audio/transcriptions/stream`

The bounded routes are backed by NVIDIA NeMo diarization-with-ASR. The
streaming route uses NeMo true-streaming ASR plus streaming diarization. One
Docker image per capability; one process per broker-dispatched container.

> **For agents:** start at [`AGENTS.md`](./AGENTS.md).

## What this repo ships

| Image | Language | Capability |
|---|---|---|
| `audio-diarized-transcription-runner` | Python | `audio:diarized-transcription@v0` with additive OpenAI-compatible transcription route |

The runner keeps the native Livepeer capability framing, but the intended
bounded integration path for general-purpose clients is now
`POST /v1/audio/transcriptions`. Richer timestamp, word, and diarization
behavior is additive on that route. Persistent live audio uses the adjacent true
streaming WebSocket path `/v1/audio/transcriptions/stream`; the older
`/v1/audio/diarized-transcriptions/stream` path remains as a compatibility
alias. It also exposes an additive stateful live-session API under
`/v1/audio/diarized-transcriptions/live/sessions` backed by NeMo
`OnlineClusteringDiarizer`. Audio is normalized with `ffmpeg`; models and
caches live under `MODEL_CACHE_DIR`.

The Docker image is based on `nvcr.io/nvidia/pytorch:25.09-py3` and installs
NeMo Toolkit ASR dependencies during build.

Runtime validation artifacts and local benchmark captures should stay out of
git. Treat [`README.md`](./README.md), [`RUNNERS.md`](./RUNNERS.md), and the
runner options endpoint as the current contract documentation.

## API surfaces

The runner exposes three additive API surfaces:

| Surface | Path | Use when | Transport |
|---|---|---|---|
| OpenAI-compatible bounded transcription | `POST /v1/audio/transcriptions` | Your app already knows how to call OpenAI audio transcription APIs, or you want one request per file/recording | multipart HTTP |
| Native bounded diarized transcription | `POST /v1/audio/diarized-transcriptions` | You want the full native diarized response shape directly | multipart HTTP |
| Adjacent true streaming transcription | `WS /v1/audio/transcriptions/stream` | You have live PCM audio and want one persistent session with low-latency events | WebSocket |

The key split is intentional:

- `POST /v1/audio/transcriptions` is the bounded file/request API.
- `WS /v1/audio/transcriptions/stream` is the live streaming API.
- The old WebSocket path `WS /v1/audio/diarized-transcriptions/stream` remains as a compatibility alias.

## OpenAI-compatible bounded transcription

`POST /v1/audio/transcriptions` is the intended main integration route for apps
that already speak the OpenAI audio transcription API.

### What it accepts

Required:

- `file`: multipart audio upload

Common fields:

- `model`: defaults to `nemo-diarized-transcription-meeting-v0`
- `language`: defaults to `en`
- `response_format`: `json`, `verbose_json`, `text`, `srt`, or `vtt`
- `timestamp_granularities[]`: `segment`, `word`, or both
- `prompt`: accepted for compatibility and ignored
- `temperature`: accepted for compatibility and ignored

Additive feature flags:

- `diarization=true` or `include_diarization=true`
- `preset=meeting`
- `num_speakers=<exact count if known>`
- `max_speakers=<clustering upper bound>`
- `include_words=true`
- `include_artifacts=true`

### Default response behavior

By default, this route behaves like a normal OpenAI-style bounded transcription API:

- `response_format=json` returns:

```json
{"text":"..."}
```

- `response_format=text` returns plain text
- `response_format=srt` returns subtitle text
- `response_format=vtt` returns subtitle text

### Verbose timestamps

When you want timestamped output, request `verbose_json` and ask for the
granularities you need:

```bash
curl -s http://localhost:8080/v1/audio/transcriptions \
  -F file=@./sample.wav \
  -F model=nemo-diarized-transcription-meeting-v0 \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' \
  -F 'timestamp_granularities[]=word' | jq
```

That returns the normal OpenAI-style bounded transcription shape:

- `text`
- `segments` when `segment` timestamps are requested
- `words` when `word` timestamps are requested

### Diarization through the OpenAI-compatible route

Diarization is additive. Turn it on explicitly:

```bash
curl -s http://localhost:8080/v1/audio/transcriptions \
  -F file=@./sample.wav \
  -F model=nemo-diarized-transcription-meeting-v0 \
  -F response_format=verbose_json \
  -F diarization=true \
  -F 'timestamp_granularities[]=segment' \
  -F 'timestamp_granularities[]=word' | jq
```

When diarization is enabled on `verbose_json`:

- each `segment` gains `speaker`
- each `word` gains `speaker`
- additive top-level fields are added:
  - `transcription_id`
  - `capability`
  - `mode`
  - `models`
  - `usage`
  - `speaker_labeled_text`
  - `diarization`
  - `artifacts`

The `diarization` object contains:

- `speaker_count`
- `speakers`
- `segments`
- `words`

This is the recommended route if you want OpenAI-style compatibility first and
speaker-aware/timestamp-aware extensions second.

## Native bounded diarized transcription

`POST /v1/audio/diarized-transcriptions` is the native route for callers that
want the full diarized payload without the OpenAI-compatible compatibility
layer.

It accepts:

- `file`
- `model`
- `language`
- `preset`
- `num_speakers`
- `max_speakers`
- `response_format`
- `include_words`
- `include_artifacts`

Use this route if you want the full native response shape directly:

- `speakers`
- `segments`
- `words`
- `artifacts`
- `usage`
- native metadata such as `capability`, `mode`, and model breakdown

## True streaming transcription

`WS /v1/audio/transcriptions/stream` is the persistent live-audio route.

Use it when:

- audio is already arriving live
- you want one long-lived session instead of repeated file uploads
- you want incremental speaker and transcript events before the session ends

### Transport contract

- binary WebSocket frames: little-endian 16 kHz mono int16 PCM
- JSON control messages:
  - `{"type":"ping"}`
  - `{"type":"finish"}`

The server sends a `session.snapshot` event immediately after connect, then
emits incremental events during the session.

### Streaming events

The main event types are:

- `audio.frame.received`
- `speaker.update`
- `transcript.segment`
- `transcript.session.finished`

`transcript.segment` events are provisional during the live session; the event
payload includes flags like `is_provisional` and timing metadata. The final
`transcript.session.finished` event closes the stream session and summarizes the
last transcript state.

### Streaming example

Start the runner:

```bash
docker compose -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up
```

Then use the included smoke client:

```bash
python scripts/audio_diarized_transcription_streaming_smoke.py ./sample.wav --require-transcript
```

That script opens `WS /v1/audio/transcriptions/stream`, converts the input into
PCM16 frames, streams them over one persistent WebSocket session, and verifies
that transcript events arrive before finish.

## Build

Every normal operator gesture is Docker-first:

```bash
./build-images.sh build                                      # build all images
./build-images.sh build audio-diarized-transcription-runner  # build the runner
./build-images.sh validate                                   # validate compose overlays
./build-images.sh push                                       # push to ${REGISTRY}
./build-images.sh clean                                      # remove locally-built images
./build-images.sh help                                       # show subcommands
```

No host Python is required for Docker build/run.

## Quick start

```bash
# 1. Build the runner.
./build-images.sh build

# 2. Run the runner. The first real request may download NeMo model weights.
docker compose -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up

# 3. Transcribe and diarize an audio file.
curl -s http://localhost:8080/v1/audio/diarized-transcriptions \
  -F file=@./sample.wav \
  -F model=nemo-diarized-transcription-meeting-v0 \
  -F language=en \
  -F preset=meeting \
  -F max_speakers=8 \
  -F response_format=json | jq
```

OpenAI-compatible clients can use the bounded compatibility route directly:

```bash
curl -s http://localhost:8080/v1/audio/transcriptions \
  -F file=@./sample.wav \
  -F model=nemo-diarized-transcription-meeting-v0 \
  -F response_format=verbose_json \
  -F diarization=true \
  -F 'timestamp_granularities[]=segment' \
  -F 'timestamp_granularities[]=word' | jq
```

Persistent live audio uses the adjacent streaming route:

```bash
python scripts/audio_diarized_transcription_streaming_smoke.py ./sample.wav --require-transcript
```

For offline / no-egress hosts, pre-stage NeMo, Hugging Face, and torch caches
into the `audio-diarized-transcription-models` volume before starting the
runner. See [`RUNNERS.md`](./RUNNERS.md) for environment variables and request
details.

## Repo layout

```text
.
├── AGENTS.md, README.md, RUNNERS.md
├── LICENSE, NOTICE
├── build-images.sh
├── infra/
│   ├── compose/        # docker-compose overlay
│   ├── dockerfiles/    # runner Dockerfile
│   ├── env/            # .env.example template
│   └── offerings/      # Livepeer offering manifest
├── audio-diarized-transcription-runner/
│   ├── pyproject.toml
│   └── src/audio_diarized_transcription_runner/
│       ├── app.py
│       ├── pipeline.py
│       ├── settings.py
│       └── configs/diar_infer_meeting.yaml
├── scripts/
│   └── audio_diarized_transcription_smoke.py
└── tests/
```

## Configuration

Common env vars:

- `RUNNER_PORT` default `8080`
- `DEVICE` default `cuda`
- `MODEL_CACHE_DIR` default `/models`
- `WORK_DIR` default `/tmp/audio-diarized-transcription-runner`
- `MAX_QUEUE_SIZE` default `1`
- `MAX_AUDIO_MB` default `100`
- `METRICS_ENABLED` default `false`
- `LIVE_HISTORY_BUFFER_SIZE` default `256`
- `LIVE_CURRENT_BUFFER_SIZE` default `256`
- `LIVE_SESSION_TTL_SECONDS` default `3600`
- `LIVE_CLOSED_SESSION_TTL_SECONDS` default `300`
- `TRUE_STREAMING_ENGINE` default `nemo`; set `fake` only for smoke tests
- `TRUE_STREAMING_ASR_MODEL` default `nvidia/nemotron-speech-streaming-en-0.6b`
- `TRUE_STREAMING_DIAR_MODEL` default `nvidia/diar_streaming_sortformer_4spk-v2.1`
- `TRUE_STREAMING_ASR_ATT_CONTEXT_SIZE` default `70,1`
- `TRUE_STREAMING_ASR_CHUNK_SECONDS` default `0.08`
- `TRUE_STREAMING_ASR_GREEDY_MAX_SYMBOLS` default `16`
- `TRUE_STREAMING_ASR_EOU_TOKENS` default `<EOU>,<EOB>`
- `TRUE_STREAMING_ASR_STABILIZE_PARTIALS` default `true`
- `TRUE_STREAMING_ASR_EMIT_MIN_WORDS` default `6`
- `TRUE_STREAMING_ASR_EMIT_MIN_CHARS` default `28`
- `TRUE_STREAMING_ASR_EMIT_MAX_HOLD_SECONDS` default `1.2`
- `TRUE_STREAMING_DIAR_THRESHOLD` default `0.4`
- `LIVE_FINAL_ASR_BATCH_SIZE` default `4`
- `LIVE_FINAL_ASR_DECODING_STRATEGY` optional NeMo RNNT/TDT decoding override
- `LIVE_FINAL_ASR_BEAM_SIZE` optional beam size for compatible final decoding strategies

The default direct-request model stack is `vad_multilingual_marblenet`,
`titanet_large`, and `stt_en_conformer_ctc_large`. Live-session ingest uses
`LIVE_PROVISIONAL_ASR_MODEL` for provisional chunk text, while live-session
final transcription uses `nvidia/parakeet-tdt-0.6b-v3` by default through
`LIVE_FINAL_ASR_MODEL`. The true streaming WebSocket path uses
`TRUE_STREAMING_ASR_MODEL` and `TRUE_STREAMING_DIAR_MODEL`; the Parakeet
realtime EOU candidate is `nvidia/parakeet_realtime_eou_120m-v1`, selected by
setting `TRUE_STREAMING_ASR_MODEL` to that value. `TRUE_STREAMING_ASR_EOU_TOKENS`
lists marker tokens that should finalize a turn and be omitted from visible
transcript text. Its optional partial
stabilizer can reduce visible token-piece churn, but it does not solve streaming
segmentation fragmentation by itself. On the 2026-06-08 30s real-stream VDO
benchmark, stabilization reduced `transcript.segment` events from 64 to 18 and
one-word-or-less segments from 61 to 1; the after run still had 8 segments with
two words or fewer. The final live pass uses NeMo native Parakeet word
timestamps and feeds them into the existing diarization labeling/output flow,
while preserving `DEFAULT_ASR_MODEL` for non-live requests. The packaged NeMo
meeting config can be overridden with `NEMO_DIARIZER_CONFIG`.

## Validation

Fast host checks that avoid model downloads:

```bash
PYTHONPATH=audio-diarized-transcription-runner/src pytest -q
./build-images.sh validate
```

Full runtime validation requires a CUDA-capable Docker host and enough disk for
NeMo model caches. The default image uses `nvcr.io/nvidia/pytorch:25.09-py3`
with `nemo_toolkit[asr]==2.6.2` so Parakeet TDT/RNNT models support
`timestamps=True`:

```bash
./build-images.sh build
docker compose -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up
python scripts/audio_diarized_transcription_smoke.py ./sample.wav --require-text
```

True streaming transport smoke with fake engines, useful for validating the
container and WebSocket event path without downloading NeMo weights:

```bash
TRUE_STREAMING_ENGINE=fake DEVICE=cpu docker compose \
  -f infra/compose/docker-compose.audio-diarized-transcription-runner.yml up
python scripts/audio_diarized_transcription_streaming_smoke.py --require-transcript
```

For NeMo streaming validation on a CUDA host, keep `TRUE_STREAMING_ENGINE=nemo`.
The first connection may download the selected ASR model and
`nvidia/diar_streaming_sortformer_4spk-v2.1`. To test the Parakeet realtime EOU
candidate, run the container with
`TRUE_STREAMING_ASR_MODEL=nvidia/parakeet_realtime_eou_120m-v1`.

## License

**Apache-2.0** — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

This repo's runner code is Apache-2.0. The image installs NVIDIA NeMo Toolkit,
PyTorch, and uses an NVIDIA NGC base image; review those upstream licenses and
NGC container terms before redistributing images.
