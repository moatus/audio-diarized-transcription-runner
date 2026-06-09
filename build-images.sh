#!/usr/bin/env bash
# Single orchestrator for the runner image.
#
# Subcommands:
#   build [name ...]   Build all images, or a named subset.
#   push  [name ...]   Push built images to ${REGISTRY}. Requires docker login.
#   validate           Run `docker compose config` against every overlay.
#   clean              Remove locally-built images for ${REGISTRY}/${TAG}.
#   help               Show this help.
#
# Environment:
#   REGISTRY       Registry prefix for runner images (default: moatus).
#   TAG            Image tag (default: v0.1.0).
#   BASE_IMAGE     CUDA/PyTorch base image (default: nvcr.io/nvidia/pytorch:25.09-py3).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REGISTRY="${REGISTRY:-moatus}"
TAG="${TAG:-v0.1.0}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.09-py3}"

ALL_IMAGES=(
  audio-diarized-transcription-runner
)

img_tag() {
  echo "${REGISTRY}/$1:${TAG}"
}

build_runner() {
  local name="$1"
  local image
  image="$(img_tag "${name}")"
  echo "==> Building ${image}"
  docker build \
    --build-arg "TAG=${TAG}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${image}" \
    -f "infra/dockerfiles/${name}.Dockerfile" \
    .
}

build_one() {
  case "$1" in
    audio-diarized-transcription-runner) build_runner audio-diarized-transcription-runner ;;
    *) echo "unknown image: $1" >&2; exit 2 ;;
  esac
}

cmd_build() {
  if [ "$#" -eq 0 ]; then
    for name in "${ALL_IMAGES[@]}"; do build_one "${name}"; done
  else
    for name in "$@"; do build_one "${name}"; done
  fi
  echo "All requested images built successfully."
}

cmd_push() {
  local targets=("$@")
  if [ "${#targets[@]}" -eq 0 ]; then
    targets=("${ALL_IMAGES[@]}")
  fi
  for name in "${targets[@]}"; do
    local image
    image="$(img_tag "${name}")"
    echo "==> Pushing ${image}"
    docker push "${image}"
  done
}

cmd_validate() {
  echo "==> Validating compose snippets in infra/compose/"
  shopt -s nullglob
  local snippets=(infra/compose/docker-compose.*.yml)
  if [ "${#snippets[@]}" -eq 0 ]; then
    echo "no compose snippets found under infra/compose/" >&2
    exit 1
  fi
  for f in "${snippets[@]}"; do
    echo "  - $f"
    docker compose -f "$f" config >/dev/null
  done
  echo "All compose snippets valid (${#snippets[@]} file(s))."
}

cmd_clean() {
  for name in "${ALL_IMAGES[@]}"; do
    docker rmi "$(img_tag "${name}")" 2>/dev/null || true
  done
}

cmd_help() {
  sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | head -n -2
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  build)    cmd_build "$@" ;;
  push)     cmd_push "$@" ;;
  validate) cmd_validate ;;
  clean)    cmd_clean ;;
  help|-h|--help) cmd_help ;;
  *) echo "unknown subcommand: ${cmd}" >&2; cmd_help; exit 2 ;;
esac
