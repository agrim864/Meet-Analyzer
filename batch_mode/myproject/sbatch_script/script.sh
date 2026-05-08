#!/bin/bash
#SBATCH --job-name=batch_mode
#SBATCH --account=gshroff
#SBATCH --partition=short
#SBATCH --output=batch_mode/myproject/logs/%x_%j.out
#SBATCH --error=batch_mode/myproject/logs/%x_%j.err
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
if [[ ! -d "${PROJECT_DIR}/batch_mode" ]]; then
  echo "Could not resolve project root. Set PROJECT_DIR to the meet-agent repo path." >&2
  echo "Resolved PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 2
fi
WORK_DIR="${PROJECT_DIR}/batch_mode/myproject"
LOG_DIR="${PROJECT_DIR}/batch_mode/myproject/logs"

mkdir -p "${LOG_DIR}"

source ~/.bashrc || true
CONDA_BIN="${HOME}/anaconda3/bin/conda"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export BATCH_DISABLE_SIGLIP="${BATCH_DISABLE_SIGLIP:-1}"
export BATCH_LLM_MODEL="${BATCH_LLM_MODEL:-unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit}"
export BATCH_HF_QUANTIZATION="${BATCH_HF_QUANTIZATION:-}"
export BATCH_LLM_BASE_URL="${BATCH_LLM_BASE_URL:-}"
export BATCH_LLM_API_KEY="${BATCH_LLM_API_KEY:-EMPTY}"
export BATCH_QWEN_MODEL_NAME="${BATCH_QWEN_MODEL_NAME:-Qwen/Qwen3-30B-A3B-FP8}"
export BATCH_EMBEDDING_MODEL="${BATCH_EMBEDDING_MODEL:-BAAI/bge-m3}"
export BATCH_LLM_BACKEND="${BATCH_LLM_BACKEND:-mock}"
export BATCH_LLM_SECTION_WORKERS="${BATCH_LLM_SECTION_WORKERS:-4}"
export BATCH_OUTPUT_DIR="${PROJECT_DIR}/batch_mode/outputs"
export BATCH_TEMP_DIR="${PROJECT_DIR}/batch_mode/tmp"
export BATCH_CLUSTER_JOBS_DIR="${PROJECT_DIR}/batch_mode/cluster_jobs"

cd "${PROJECT_DIR}"
echo "batch_mode job started on $(hostname) at $(date -Is)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "BATCH_SOURCE=${BATCH_SOURCE:-}"
echo "BATCH_REQUEST_JSON=${BATCH_REQUEST_JSON:-}"
echo "BATCH_LLM_BACKEND=${BATCH_LLM_BACKEND:-}"

if [[ "${BATCH_BOOTSTRAP:-0}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n torch_clean python -m pip install -r "${PROJECT_DIR}/batch_mode/requirements.txt"
  "${CONDA_BIN}" run --no-capture-output -n torch_clean python - <<'PY'
from huggingface_hub import snapshot_download
import os

hf_token = os.getenv("HF_API_KEY", "").strip() or None
snapshot_download(repo_id=os.getenv("BATCH_EMBEDDING_MODEL", "BAAI/bge-m3"), token=hf_token)
snapshot_download(repo_id=os.getenv("BATCH_LLM_MODEL", "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit"), token=hf_token)
PY
fi

if [[ -n "${BATCH_REQUEST_JSON:-}" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n torch_clean python "${WORK_DIR}/test.py" \
    --request-json "${BATCH_REQUEST_JSON}" \
    --output-json "${BATCH_RESULT_JSON:-${PROJECT_DIR}/batch_mode/cluster_jobs/result.json}"
else
  "${CONDA_BIN}" run --no-capture-output -n torch_clean python "${WORK_DIR}/test.py" --source "${BATCH_SOURCE:?Set BATCH_SOURCE to a local path or public mp4 URL}"
fi
