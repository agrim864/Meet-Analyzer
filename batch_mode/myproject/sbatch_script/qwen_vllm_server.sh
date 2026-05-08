#!/bin/bash
#SBATCH --job-name=qwen_vllm
#SBATCH --partition=short
#SBATCH --output=batch_mode/myproject/logs/%x_%j.out
#SBATCH --error=batch_mode/myproject/logs/%x_%j.err
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
LOG_DIR="${PROJECT_DIR}/batch_mode/myproject/logs"
mkdir -p "${LOG_DIR}"

MODEL="${BATCH_LLM_MODEL:-${BATCH_QWEN_MODEL_NAME:-Qwen/Qwen3-30B-A3B-FP8}}"
HOST="${BATCH_VLLM_HOST:-0.0.0.0}"
PORT="${BATCH_VLLM_PORT:-8000}"
CONDA_ENV="${BATCH_CLUSTER_CONDA_ENV:-torch_clean}"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${PROJECT_DIR}"
echo "qwen_vllm server starting on $(hostname) at $(date -Is)"
echo "MODEL=${MODEL}"
echo "BASE_URL=http://$(hostname -f):${PORT}/v1"
echo "Use this for batch jobs: export BATCH_LLM_BASE_URL=http://$(hostname -f):${PORT}/v1"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code
