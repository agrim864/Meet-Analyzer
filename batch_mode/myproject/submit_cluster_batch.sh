#!/usr/bin/env bash
# Submit multiple batch-mode GPU jobs sequentially.
# Sources can be passed as arguments, read from --file, or pasted on stdin.

set -euo pipefail

WRITER="${BATCH_REPORT_WRITER:-mistral}"
TITLE_PREFIX="${BATCH_TITLE_PREFIX:-batch}"
SOURCE_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --writer)
      WRITER="${2:?Missing value for --writer}"
      shift 2
      ;;
    --file)
      SOURCE_FILE="${2:?Missing value for --file}"
      shift 2
      ;;
    --title-prefix)
      TITLE_PREFIX="${2:?Missing value for --title-prefix}"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  submit_cluster_batch.sh [--writer mistral|gemini|qwen|mock] [--title-prefix NAME] URL_OR_PATH...
  submit_cluster_batch.sh [--writer mistral|gemini|qwen|mock] --file links.txt
  submit_cluster_batch.sh [--writer mistral|gemini|qwen|mock] < links.txt

Each non-empty, non-comment line in a file/stdin is treated as one video source.
Jobs are submitted with afterany dependencies, so they run one after another.
EOF
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

sources=()
if [[ -n "${SOURCE_FILE}" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(printf '%s' "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${line}" ]] && sources+=("${line}")
  done < "${SOURCE_FILE}"
elif [[ $# -gt 0 ]]; then
  sources=("$@")
else
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(printf '%s' "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${line}" ]] && sources+=("${line}")
  done
fi

if [[ ${#sources[@]} -eq 0 ]]; then
  echo "No video links/paths provided." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
previous_job_id=""
index=1

for source in "${sources[@]}"; do
  safe_title="$(printf '%s_%02d' "${TITLE_PREFIX}" "${index}")"
  output_name="${safe_title}.mp4"
  echo "Submitting ${safe_title}: ${source}"
  if [[ -n "${previous_job_id}" ]]; then
    export BATCH_DEPENDENCY_AFTER_JOB_ID="${previous_job_id}"
  else
    unset BATCH_DEPENDENCY_AFTER_JOB_ID || true
  fi
  export BATCH_TITLE="${safe_title}"
  submit_output="$("${script_dir}/submit_cluster_source.sh" --writer "${WRITER}" "${source}" "${output_name}")"
  printf '%s\n' "${submit_output}"
  job_id="$(printf '%s\n' "${submit_output}" | awk '/Submitted batch job/ {print $NF}' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "Could not parse Slurm job id for ${safe_title}." >&2
    exit 1
  fi
  previous_job_id="${job_id}"
  index=$((index + 1))
done

unset BATCH_DEPENDENCY_AFTER_JOB_ID || true
echo "Submitted ${#sources[@]} jobs sequentially. Last job id: ${previous_job_id}"
