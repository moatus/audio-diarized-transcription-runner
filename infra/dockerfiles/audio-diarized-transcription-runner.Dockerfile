# audio-diarized-transcription-runner — NeMo diarized transcription on CUDA.
# Build context: repo root.

ARG TAG=v0.1.0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/moatus/audio-diarized-transcription-runner"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="audio-diarized-transcription-runner"
LABEL org.opencontainers.image.description="Livepeer audio diarized transcription runner with bounded OpenAI-compatible transcription and adjacent true streaming"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    RUNNER_PORT=8080 \
    DEVICE=cuda \
    CAPABILITY_NAME=openai:audio-transcriptions \
    METRICS_ENABLED=false \
    MAX_QUEUE_SIZE=1 \
    MAX_AUDIO_MB=100 \
    MODEL_CACHE_DIR=/models \
    WORK_DIR=/tmp/audio-diarized-transcription-runner \
    NEMO_HOME=/models \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    DEFAULT_VAD_MODEL=vad_multilingual_marblenet \
    DEFAULT_SPEAKER_MODEL=titanet_large \
    DEFAULT_ASR_MODEL=stt_en_conformer_ctc_large \
    LIVE_PROVISIONAL_ASR_MODEL=stt_en_conformer_ctc_large \
    LIVE_FINAL_ASR_MODEL=nvidia/parakeet-tdt-0.6b-v3 \
    LIVE_FINAL_ASR_BATCH_SIZE=4 \
    LIVE_FINAL_ASR_DECODING_STRATEGY= \
    LIVE_FINAL_ASR_BEAM_SIZE= \
    DEFAULT_MAX_SPEAKERS=8 \
    DEFAULT_PRESET=meeting \
    LIVE_SAMPLE_RATE=16000 \
    LIVE_HISTORY_BUFFER_SIZE=256 \
    LIVE_CURRENT_BUFFER_SIZE=256 \
    LIVE_SESSION_TTL_SECONDS=3600 \
    LIVE_CLOSED_SESSION_TTL_SECONDS=300 \
    TRUE_STREAMING_ENABLED=true \
    TRUE_STREAMING_ENGINE=nemo \
    TRUE_STREAMING_ASR_MODEL=nvidia/nemotron-speech-streaming-en-0.6b \
    TRUE_STREAMING_DIAR_MODEL=nvidia/diar_streaming_sortformer_4spk-v2.1 \
    TRUE_STREAMING_SAMPLE_RATE=16000 \
    TRUE_STREAMING_ASR_ATT_CONTEXT_SIZE=70,1 \
    TRUE_STREAMING_ASR_CHUNK_SECONDS=0.08 \
    TRUE_STREAMING_ASR_GREEDY_MAX_SYMBOLS=16 \
    TRUE_STREAMING_ASR_EOU_TOKENS='<EOU>,<EOB>' \
    TRUE_STREAMING_ASR_STABILIZE_PARTIALS=true \
    TRUE_STREAMING_ASR_EMIT_MIN_WORDS=6 \
    TRUE_STREAMING_ASR_EMIT_MIN_CHARS=28 \
    TRUE_STREAMING_ASR_EMIT_MAX_HOLD_SECONDS=1.2 \
    TRUE_STREAMING_DIAR_FRAME_SECONDS=0.08 \
    TRUE_STREAMING_DIAR_THRESHOLD=0.4 \
    TRUE_STREAMING_MAX_SESSION_SECONDS=3600 \
    SOURCE_URL=https://github.com/moatus/audio-diarized-transcription-runner

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 sox git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY audio-diarized-transcription-runner/pyproject.toml ./
COPY audio-diarized-transcription-runner/src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY scripts ./scripts

RUN mkdir -p /models /tmp/audio-diarized-transcription-runner

VOLUME /models

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["python", "-m", "audio_diarized_transcription_runner"]
