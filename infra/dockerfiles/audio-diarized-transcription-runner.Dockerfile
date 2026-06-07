# audio-diarized-transcription-runner — NeMo diarized transcription on CUDA.
# Build context: repo root.

ARG TAG=v0.1.0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/moatus/audio-diarized-transcription-runner"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="audio-diarized-transcription-runner"
LABEL org.opencontainers.image.description="Livepeer audio diarized transcription runner (POST /v1/audio/diarized-transcriptions)"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    RUNNER_PORT=8080 \
    DEVICE=cuda \
    CAPABILITY_NAME=audio:diarized-transcription@v0 \
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
    LIVE_FINAL_ASR_MODEL=stt_en_fastconformer_ctc_large \
    DEFAULT_MAX_SPEAKERS=8 \
    DEFAULT_PRESET=meeting \
    LIVE_SAMPLE_RATE=16000 \
    LIVE_HISTORY_BUFFER_SIZE=256 \
    LIVE_CURRENT_BUFFER_SIZE=256 \
    LIVE_SESSION_TTL_SECONDS=3600 \
    LIVE_CLOSED_SESSION_TTL_SECONDS=300 \
    SOURCE_URL=https://github.com/moatus/audio-diarized-transcription-runner

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 sox git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY audio-diarized-transcription-runner/pyproject.toml ./
COPY audio-diarized-transcription-runner/src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /models /tmp/audio-diarized-transcription-runner

VOLUME /models

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["python", "-m", "audio_diarized_transcription_runner"]
