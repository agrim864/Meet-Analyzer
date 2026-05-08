#!/usr/bin/env bash
# Submit a batch-mode GPU job from the cluster login node.
# Public links are downloaded here first so compute nodes only read local files.

set -euo pipefail

WRITER="${BATCH_REPORT_WRITER:-}"
if [[ "${1:-}" == "--writer" ]]; then
  WRITER="${2:?Usage: $0 [--writer mistral|gemini|qwen|mock] <public-video-url-or-local-path> [output-video-name]}"
  shift 2
fi
SOURCE="${1:?Usage: $0 [--writer mistral|gemini|qwen|mock] <public-video-url-or-local-path> [output-video-name]}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"
CONDA_ENV="${BATCH_CLUSTER_CONDA_ENV:-torch_clean}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/batch_mode/tmp" "${PROJECT_DIR}/batch_mode/myproject/logs" "${PROJECT_DIR}/batch_mode/cluster_jobs"

case "${WRITER:-}" in
  ""|mistral)
    export BATCH_LLM_BACKEND="${BATCH_LLM_BACKEND:-hf-local}"
    export BATCH_LLM_MODEL="${BATCH_LLM_MODEL:-unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit}"
    export BATCH_HF_QUANTIZATION="${BATCH_HF_QUANTIZATION:-}"
    export BATCH_LLM_USE_PREWRITTEN="${BATCH_LLM_USE_PREWRITTEN:-0}"
    export BATCH_LLM_MAX_PROMPT_CHARS="${BATCH_LLM_MAX_PROMPT_CHARS:-60000}"
    export BATCH_LLM_MAX_INPUT_TOKENS="${BATCH_LLM_MAX_INPUT_TOKENS:-24576}"
    export BATCH_LLM_MAX_NEW_TOKENS="${BATCH_LLM_MAX_NEW_TOKENS:-3072}"
    export BATCH_LLM_SECTION_WORKERS="${BATCH_LLM_SECTION_WORKERS:-1}"
    ;;
  gemini)
    export BATCH_LLM_BACKEND="gemini"
    export BATCH_LLM_MODEL="${BATCH_LLM_MODEL:-gemini-2.5-flash}"
    export BATCH_LLM_USE_PREWRITTEN="${BATCH_LLM_USE_PREWRITTEN:-0}"
    export BATCH_LLM_MAX_PROMPT_CHARS="${BATCH_LLM_MAX_PROMPT_CHARS:-80000}"
    export BATCH_LLM_MAX_NEW_TOKENS="${BATCH_LLM_MAX_NEW_TOKENS:-8192}"
    export BATCH_LLM_TIMEOUT_SEC="${BATCH_LLM_TIMEOUT_SEC:-900}"
    export BATCH_LLM_SECTION_WORKERS="${BATCH_LLM_SECTION_WORKERS:-1}"
    ;;
  qwen)
    export BATCH_LLM_BACKEND="hf-local"
    export BATCH_LLM_MODEL="${BATCH_LLM_MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
    export BATCH_HF_QUANTIZATION="${BATCH_HF_QUANTIZATION:-}"
    ;;
  mock|preserve)
    export BATCH_LLM_BACKEND="mock"
    ;;
  *)
    echo "Unknown --writer '${WRITER}'. Use mistral, gemini, qwen, or mock." >&2
    exit 2
    ;;
esac

if [[ "${SOURCE}" =~ ^https?:// ]]; then
  name="${2:-$(basename "${SOURCE%%\?*}")}"
  if [[ -z "${name}" || "${name}" == "." || "${name}" == "/" ]]; then
    name="cluster_input.mp4"
  fi
  name="$(printf '%s' "${name}" | tr -c 'A-Za-z0-9._-' '_')"
  if [[ "${name}" != *.* ]]; then
    name="${name}.mp4"
  fi
  dest="${PROJECT_DIR}/batch_mode/tmp/${name}"
  echo "Downloading source link on login node:"
  echo "  ${SOURCE}"
  echo "  -> ${dest}"
  curl -L --fail --show-error --connect-timeout 30 -o "${dest}" "${SOURCE}"
  export BATCH_SOURCE="${dest}"
else
  export BATCH_SOURCE="${SOURCE}"
fi

request_id="cli_$(date +%Y%m%d_%H%M%S)_$$"
request_dir="${PROJECT_DIR}/batch_mode/cluster_jobs/${request_id}"
mkdir -p "${request_dir}"
export BATCH_REQUEST_JSON="${request_dir}/request.json"
export BATCH_RESULT_JSON="${request_dir}/result.json"

prepare_args=(
  python "${PROJECT_DIR}/batch_mode/myproject/prepare_cluster_request.py"
  --source "${BATCH_SOURCE}"
  --request-json "${BATCH_REQUEST_JSON}"
  --transcript-json "${request_dir}/transcript_segments.json"
  --title "${BATCH_TITLE:-$(basename "${BATCH_SOURCE%.*}")}"
)
if [[ "${BATCH_TRANSCRIBE_ON_SUBMIT:-1}" == "0" ]]; then
  prepare_args+=(--skip-transcript)
fi
if [[ "${BATCH_REQUIRE_TRANSCRIPT:-1}" != "0" ]]; then
  prepare_args+=(--require-transcript)
fi
if [[ "${BATCH_PERSIST_ARTIFACTS:-1}" == "0" ]]; then
  prepare_args+=(--no-persist-artifacts)
fi

echo "Preparing request payload on login node:"
echo "  ${BATCH_REQUEST_JSON}"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" "${prepare_args[@]}"

export PROJECT_DIR
export BATCH_DISABLE_SIGLIP="${BATCH_DISABLE_SIGLIP:-1}"
export BATCH_LLM_BACKEND="${BATCH_LLM_BACKEND:-hf-local}"
export PYTHONUNBUFFERED=1

dependency_args=()
if [[ -n "${BATCH_DEPENDENCY_AFTER_JOB_ID:-}" ]]; then
  dependency_args+=(--dependency="afterany:${BATCH_DEPENDENCY_AFTER_JOB_ID}")
fi

sbatch "${dependency_args[@]}" --export=ALL batch_mode/myproject/sbatch_script/script.sh
