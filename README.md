# Meet Analyzer Batch

I built this repo to turn long recorded meetings into a report that is actually usable: not just a transcript, and not just OCR noise, but a timeline of what was on screen, what changed, where the cursor was, what the speaker said, and what the final report should say.

The current batch pipeline supports public video links, local MP4 files, a local FastAPI service, and direct Slurm GPU-cluster jobs. It is optimized for evaluated presentations, screen-share demos, code walkthroughs, and slide-heavy meetings.

<p align="center">
  <img src="docs/assets/architecture_overview.png" alt="Meet Analyzer Batch architecture" width="960">
</p>

## What It Does

The analyzer takes a video and produces:

- `analysis.json`: structured events, transcript segments, cursor targets, macro layout, UI elements, topic units, timing metrics, and writer output.
- `report.md`: a readable Markdown report with narrative, demo/code/slide sections, topics, decisions, and transcript.
- Optional proof artifacts for validation: raw frames, annotated segment overlays, metrics dashboards, and report previews.

The key design choice is that screen understanding is evidence-first. The pipeline samples video frames, detects the visible page structure, tracks cursor activity, parses UI elements, aligns transcript, and only then asks an LLM writer to produce the final report.

## Visual Proof

This is the current validation run on the Bunny sample video. The left side is the original frame; the right side is the generated macro-layout output.

<p align="center">
  <img src="docs/assets/segment_validation_contact_sheet.jpg" alt="Raw frame vs generated segment overlays" width="960">
</p>

## Metrics From The Validation Run

<p align="center">
  <img src="docs/assets/metrics_dashboard.png" alt="Validation metrics dashboard" width="960">
</p>

The validation metrics are stored in [docs/assets/validation_metrics.json](docs/assets/validation_metrics.json).

| Metric | Value |
| --- | ---: |
| Sampled frames | `6` |
| Sample timestamps | `2s, 15s, 58s, 100s, 145s, 205s` |
| Mean grid coverage | `100.00%` |
| High-confidence regions | `13` |
| Low-confidence retained regions | `13` |
| Unit tests | `14 passed` |

Previous full Bunny cluster run:

| Stage | Time |
| --- | ---: |
| Visual analysis | `1032.555s` |
| Mistral report generation | `341.796s` |
| Total runtime | `1388.791s` |

## Class Gallery

Each panel below is a photo-vs-output proof for one macro segment class.

<p align="center">
  <img src="docs/assets/class_gallery.png" alt="Class by class photo vs output gallery" width="960">
</p>

Current behavior:

- `header`, `left_sidebar`, and `main_content` line up well on the ICAPP validation sample.
- `right_sidebar` and `footer` are often below the `0.75` high-confidence threshold, but the coverage logic keeps them when they explain otherwise-unclaimed screen space.
- If no region exists for an uncovered area, it is assigned to `main_content`.

## Report Preview

This preview was generated from a short interval using the sampled visual events plus transcript evidence.

<p align="center">
  <img src="docs/assets/report_preview.png" alt="Generated report preview" width="960">
</p>

The mini report reconstructs the ICAPP layout, title slide, Game Rules slide, Simulation Environment slide, cursor/navigation flow, and evaluator/presenter turns. The raw Markdown output is available at [docs/examples/mini_interval_report.md](docs/examples/mini_interval_report.md).

## Pipeline

```text
Video URL or MP4
  -> source resolver / downloader
  -> Deepgram transcript
  -> frame sampling
  -> screen-mode routing
  -> cursor detection
  -> yolo-ui macro layout
  -> OmniParser full-frame or crop parse
  -> OCR / slide / code extraction
  -> topic segmentation
  -> report section builder
  -> Mistral / Gemini / Qwen / mock writer
  -> analysis.json + report.md
```

## Models

I pushed the model files that are small enough for normal GitHub storage and useful enough to make the repo runnable quickly.

### Included In Git

| Model artifact | Path | Size | Purpose |
| --- | --- | ---: | --- |
| yolo-ui macro detector | `real_time_mode/vendor/uitag/yolo-ui.pt` | `18.31 MB` | Coarse screen segments |
| Cursor detector | `batch_mode/cursor_class_best.pt` | `6.16 MB` | Cursor box and hotspot detection |
| OmniParser icon detector | `batch_mode/vendor/OmniParser/weights/icon_detect/model.pt` | `38.74 MB` | UI element detection |
| OmniParser detector metadata | `batch_mode/vendor/OmniParser/weights/icon_detect/*.yaml` | small | Required by icon detector |
| OmniParser caption config | `batch_mode/vendor/OmniParser/weights/icon_caption/*.json` | small | Caption model config |

I did not use Git LFS here because every committed model artifact is below GitHub's normal per-file limit. I intentionally kept duplicate realtime and backup checkpoints out of Git so the repository remains practical to push and clone.

### Not Included In Git

The following file is too large for GitHub:

```text
batch_mode/vendor/OmniParser/weights/icon_caption/model.safetensors
```

It is about `1.03 GB`. Download it after cloning if you want local semantic captions from OmniParser. Without it, the pipeline still runs with icon detection and OCR-backed labels, but caption semantics are weaker.

Download command:

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_caption/model.safetensors",
    repo_type="model",
    token=os.environ.get("HF_API_KEY", ""),
    local_dir="batch_mode/vendor/OmniParser/weights",
    local_dir_use_symlinks=False,
)
print("Downloaded OmniParser caption model.")
PY
```

PowerShell:

```powershell
@'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_caption/model.safetensors",
    repo_type="model",
    token=os.environ.get("HF_API_KEY", ""),
    local_dir="batch_mode/vendor/OmniParser/weights",
    local_dir_use_symlinks=False,
)
print("Downloaded OmniParser caption model.")
'@ | python -
```

## Fresh Setup

### 1. Clone

```bash
git clone https://github.com/agrim864/Meet-Analyzer-Batch.git
cd Meet-Analyzer-Batch
```

### 2. Create Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r batch_mode\requirements.txt
```

Linux / cluster:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r batch_mode/requirements.txt
```

Existing cluster conda environment:

```bash
conda activate torch_clean
pip install -r batch_mode/requirements.txt
```

### 3. Configure Environment

Copy the example env file:

```powershell
Copy-Item .env.example batch_mode\.env
```

Linux:

```bash
cp .env.example batch_mode/.env
```

Then edit `batch_mode/.env` and fill in your keys. Do not commit real keys.

## Environment Example

The committed template is [.env.example](.env.example).

Minimum useful local setup:

```env
DEEPGRAM_API_KEY=your_deepgram_key
HF_API_KEY=your_huggingface_token
BATCH_LLM_BACKEND=mock
BATCH_LLM_MOCK_ENABLED=1
```

For GPU-cluster Mistral report generation:

```env
BATCH_LLM_BACKEND=hf-local
BATCH_LLM_MODEL=unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit
BATCH_LLM_USE_PREWRITTEN=0
BATCH_LLM_SECTION_WORKERS=1
BATCH_LLM_MAX_PROMPT_CHARS=60000
BATCH_LLM_MAX_INPUT_TOKENS=24576
BATCH_LLM_MAX_NEW_TOKENS=3072
```

For cluster submission through the API:

```env
BATCH_CLUSTER_HOST=192.168.3.233
BATCH_CLUSTER_USER=gshroff
BATCH_CLUSTER_ACCOUNT=gshroff
BATCH_CLUSTER_CONDA_ENV=torch_clean
BATCH_CLUSTER_PARTITION=short
BATCH_CLUSTER_GRES=gpu:3g.40gb:1
BATCH_CLUSTER_TIME_LIMIT=02:00:00
BATCH_CLUSTER_MEMORY=64G
BATCH_CLUSTER_CPUS_PER_TASK=8
BATCH_CLUSTER_PROJECT_DIR=/Home/gshroff/myproject/meet-agent-main
BATCH_CLUSTER_LOG_DIR=/Home/gshroff/myproject/logs
```

## Quickstart: Direct Cluster Test

This is the command I use when I want to test directly on the GPU cluster. Put your video link in `VIDEO_URL`.

```bash
ssh gshroff@192.168.3.233
cd /Home/gshroff/myproject/meet-agent-main
mkdir -p /Home/gshroff/myproject/logs batch_mode/outputs batch_mode/tmp

VIDEO_URL="https://your-video-link.mp4"
RUN_NAME="my_cluster_test"

sbatch \
  --account gshroff \
  --partition short \
  --gres gpu:3g.40gb:1 \
  --time 02:00:00 \
  --mem 64G \
  --cpus-per-task 8 \
  --job-name batch_analyzer \
  --output /Home/gshroff/myproject/logs/${RUN_NAME}_%j.out \
  --error /Home/gshroff/myproject/logs/${RUN_NAME}_%j.err \
  --wrap "cd /Home/gshroff/myproject/meet-agent-main && export PYTHONPATH=/Home/gshroff/myproject/meet-agent-main:\${PYTHONPATH:-} && export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && export BATCH_DISABLE_SIGLIP=1 && export PYTHONUNBUFFERED=1 && export BATCH_SAMPLE_FPS=0.5 && export BATCH_LLM_BACKEND=hf-local && export BATCH_LLM_MODEL=unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit && export BATCH_HF_QUANTIZATION='' && export BATCH_LLM_USE_PREWRITTEN=0 && export BATCH_LLM_SECTION_WORKERS=1 && export BATCH_LLM_MAX_PROMPT_CHARS=60000 && export BATCH_LLM_MAX_INPUT_TOKENS=24576 && export BATCH_LLM_MAX_NEW_TOKENS=3072 && export BATCH_OUTPUT_DIR=/Home/gshroff/myproject/meet-agent-main/batch_mode/outputs && export BATCH_TEMP_DIR=/Home/gshroff/myproject/meet-agent-main/batch_mode/tmp && /home/gshroff/anaconda3/bin/conda run --no-capture-output -n torch_clean python batch_mode/myproject/test.py --source \"$VIDEO_URL\" --title \"$RUN_NAME\" --writer mistral --output-json batch_mode/outputs/${RUN_NAME}_result.json"
```

Monitor:

```bash
squeue -u gshroff
tail -f /Home/gshroff/myproject/logs/${RUN_NAME}_<JOB_ID>.out
tail -f /Home/gshroff/myproject/logs/${RUN_NAME}_<JOB_ID>.err
```

When it finishes:

```bash
cat batch_mode/outputs/${RUN_NAME}_result.json | head
```

The result JSON contains `output_path`; the final `report.md` is next to that `analysis.json`.

## Run Through The Local API

Start the API:

```powershell
cd c:\meet-agent\meet-agent-main
.\.venv\Scripts\Activate.ps1
python -m uvicorn batch_mode.api:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl.exe "http://127.0.0.1:8000/health"
```

Run locally through the API:

```powershell
$VIDEO_URL = "https://your-video-link.mp4"

$body = @{
  source = $VIDEO_URL
  title = "local-api-run"
  persist_artifacts = $true
  writer = "mistral"
} | ConvertTo-Json

curl.exe -X POST "http://127.0.0.1:8000/api/analyze" `
  -H "Content-Type: application/json" `
  --data-binary $body
```

Submit to the cluster through the API:

```powershell
$VIDEO_URL = "https://your-video-link.mp4"

$body = @{
  source = $VIDEO_URL
  title = "cluster-api-run"
  persist_artifacts = $true
  writer = "mistral"
} | ConvertTo-Json

$response = curl.exe -s -X POST "http://127.0.0.1:8000/api/analyze/cluster" `
  -H "Content-Type: application/json" `
  --data-binary $body

$job = $response | ConvertFrom-Json
$job.request_id
$job.job_id
```

Check status and download report:

```powershell
curl.exe "http://127.0.0.1:8000/api/jobs/$($job.request_id)"

curl.exe "http://127.0.0.1:8000/api/jobs/$($job.request_id)/report?download=true" `
  -o "batch_mode\outputs\$($job.request_id)_report.md"

curl.exe "http://127.0.0.1:8000/api/jobs/$($job.request_id)/result?include_full=true&download=true" `
  -o "batch_mode\outputs\$($job.request_id)_analysis.json"
```

## Run From The Command Line

Local CLI:

```powershell
$VIDEO_URL = "https://your-video-link.mp4"

python batch_mode\myproject\test.py `
  --source $VIDEO_URL `
  --title "local-cli-run" `
  --writer "mock" `
  --output-json "batch_mode\outputs\local_cli_result.json"
```

Regenerate a report from an existing `analysis.json`:

```powershell
python batch_mode\myproject\rebuild_report.py `
  --analysis-json "batch_mode\outputs\<run>\analysis.json" `
  --writer mistral
```

## Batch Submission

```powershell
$body = @{
  writer = "mistral"
  sequential = $true
  requests = @(
    @{
      source = "https://example.com/video_1.mp4"
      title = "video-1"
      persist_artifacts = $true
    },
    @{
      source = "https://example.com/video_2.mp4"
      title = "video-2"
      persist_artifacts = $true
    }
  )
} | ConvertTo-Json -Depth 5

curl.exe -X POST "http://127.0.0.1:8000/api/analyze/cluster/batch" `
  -H "Content-Type: application/json" `
  --data-binary $body
```

## Writer Presets

| Writer | Backend | Best use |
| --- | --- | --- |
| `mock` | deterministic local writer | smoke tests |
| `mistral` | local Hugging Face Mistral | cluster production reports |
| `qwen` | local Hugging Face Qwen | large-GPU experiments |
| `gemini` | Gemini API | fast cloud report generation |

## Useful Tuning

| Variable | Default | Meaning |
| --- | --- | --- |
| `BATCH_SAMPLE_FPS` | `0.5` | Frame sampling rate for batch analysis |
| `DEMO_INTERVAL_SEC` | `1.0` | Minimum interval between demo events |
| `SLIDE_INTERVAL_SEC` | `3.0` | Minimum interval between slide events |
| `CODE_INTERVAL_SEC` | `2.0` | Minimum interval between code events |
| `OMNIPARSER_BBOX_THRESHOLD` | `0.05` | OmniParser UI detection threshold |
| `OMNIPARSER_IOU_THRESHOLD` | `0.10` | OmniParser NMS threshold |
| `MACRO_LAYOUT_CONF` | `0.20` | yolo-ui tile detector threshold |
| `MACRO_LAYOUT_IOU` | `0.45` | yolo-ui NMS IoU |
| `BATCH_LLM_USE_PREWRITTEN` | `1` | Use deterministic sections when available |
| `BATCH_LLM_SECTION_WORKERS` | `4` | Parallel report section generation |

For local HF writers, set `BATCH_LLM_SECTION_WORKERS=1`.

## Regenerate The README Proof Images

```powershell
python docs\generate_segment_validation.py `
  --source "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4" `
  --output-dir "docs\assets" `
  --times "2,15,58,100,145,205" `
  --weights "real_time_mode\vendor\uitag\yolo-ui.pt" `
  --device "cpu"

python docs\build_readme_assets.py
```

## Tests

```powershell
python -m unittest batch_mode.tests.test_batch_mode
python -m py_compile batch_mode\macro_layout.py batch_mode\pipeline.py batch_mode\llm_adapters.py batch_mode\myproject\test.py docs\generate_segment_validation.py docs\build_readme_assets.py
```

Current smoke result:

```text
Ran 14 tests
OK
```

## Known Limitations

- The right transcript/webcam panel is currently the weakest macro class. It is often detected below `0.75`, which is why the low-confidence coverage path is important.
- The 1.03 GB OmniParser caption model is not committed. Download it with the command above if you need local semantic icon captions.
- OCR on compressed meeting recordings is noisy. The report writer treats OCR as screen evidence, not spoken transcript.
- Full Mistral/Qwen local writers need a CUDA GPU with enough memory and all Hugging Face models cached when cluster jobs run offline.

## My Recommended Production Flow

1. Put the video URL in `VIDEO_URL`.
2. Submit through `/api/analyze/cluster` or the direct `sbatch` command.
3. Poll the job.
4. Download `report.md`.
5. If the report looks off, inspect `analysis.json`, `macro_layout`, `screen_structure`, and `cursor_target` for the relevant timestamps.

That is the cleanest loop for debugging both quality and runtime.
