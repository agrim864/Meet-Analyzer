# Batch Mode

Batch mode is the offline research pipeline for recorded screen-share meetings. It converts a public MP4 URL or local video into structured visual events, transcript-linked state changes, topic units, and a final Markdown report.

The mode is designed for long demos, class/project evaluations, code walkthroughs, slide-heavy meetings, and cluster runs where the output must be auditable after the meeting ends.

## Validation Source

The committed validation assets use this Bunny sample:

```text
https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4
```

Five varied states are sampled: title screen, game rules, simulation environment, agent table, and algorithm logic.

<p align="center">
  <img src="../docs/assets/segment_validation_contact_sheet.jpg" alt="Bunny macro page-segment validation" width="920">
</p>

<p align="center">
  <img src="../docs/assets/micro_validation_contact_sheet.jpg" alt="Bunny micro element validation" width="920">
</p>

## Pipeline

```text
Video URL / local MP4
  -> source resolver and downloader
  -> optional Deepgram transcript
  -> frame sampling and mode routing
  -> cursor detection and cursor masking
  -> macro page segmentation
  -> OmniParser micro-element extraction
  -> OCR / Surya slide layout / code text extraction
  -> changed-region cache and state merge
  -> transcript-window alignment
  -> topic segmentation and global topic linking
  -> report section builder
  -> mock, Gemini, OpenRouter, local HF Mistral, or Qwen writer
  -> analysis.json + report.md
```

The key idea is evidence reuse. Stable frames are not treated as brand-new images. The analyzer caches macro layout, reparses only changed crops when safe, carries forward unchanged UI elements, and attaches each visual event to the transcript window that happened since the previous visual change.

## Outputs

Typical output directory:

```text
batch_mode/outputs/<run_name>/
+-- analysis.json       # Full structured state, timings, topics, sections, writer output
`-- report.md           # Human-readable final report
```

Important fields in `analysis.json`:

| Field | Meaning |
| --- | --- |
| `frame_events` | Per-frame screen state, visual payload, cursor target, transcript window |
| `topic_units` | Time-bounded units grouped by mode, OCR, visual summaries, and transcript |
| `topic_clusters` | Cross-unit semantic clusters from the global linker |
| `report_sections` | Evidence-backed section prompts and prewritten deterministic sections |
| `metrics` / timings | Runtime and stage timing metadata when available |

## Macro Segmentation Metrics

<p align="center">
  <img src="../docs/assets/metrics_dashboard.png" alt="Macro validation metrics" width="920">
</p>

| Metric | Value |
| --- | ---: |
| Sampled frames | `5` |
| Sample timestamps | `2s, 58s, 100s, 145s, 205s` |
| Mean grid coverage | `100.00%` |
| High-confidence regions | `11` |
| Low-confidence retained regions | `4` |

| Class | Detected frames | Example score | Source |
| --- | ---: | ---: | --- |
| `header` | `5 / 5` | `0.8049` | `uitag-yolo` |
| `left_sidebar` | `5 / 5` | `0.7755` | `uitag-yolo` |
| `main_content` | `5 / 5` | `0.8049` | `uitag-yolo-coverage` |

Sidebar precision is intentionally conservative. Wide left-edge content does not become a sidebar unless it has rail-like geometry, compact width, vertical repetition, and enough screen-height coverage.

## Micro Element Metrics

<p align="center">
  <img src="../docs/assets/micro_metrics_dashboard.png" alt="Micro validation metrics" width="920">
</p>

| Metric | Value |
| --- | ---: |
| Parsed Bunny frames | `5` |
| OmniParser elements | `296` |
| Mean elements per frame | `59.2` |
| Interactive elements | `259` |
| Elements with text or caption content | `296` |

Element type distribution:

| Type | Count |
| --- | ---: |
| `icon` | `234` |
| `text` | `37` |
| `button` | `21` |
| `tab` | `3` |
| `input` | `1` |

Micro elements are assigned back to their parent macro region before report construction. That lets the report say not only that a button appeared, but whether it appeared in the main content, header, navigation rail, or a changed crop.

## Models And Techniques

| Component | Implementation | Why it is used |
| --- | --- | --- |
| Mode router | SigLIP/SigLIP2 prompt bank with positive and negative prompts | Separates slide, demo, code, and other frames before expensive parsing |
| Sticky routing | Confidence floors, margins, and previous-mode guards | Prevents rapid mode flapping during mixed slide/demo screens |
| Macro layout | yolo-ui detector plus geometry post-processing | Finds header/sidebar/main-content regions while suppressing false sidebars |
| Coverage fallback | Grid coverage assignment to `main_content` | Ensures every important screen area belongs to a macro region |
| Micro parsing | OmniParser v2 icon detector and caption stack | Extracts controls, icons, text boxes, and visible UI elements |
| OCR | RapidOCR by default, EasyOCR for GPU-oriented dense text, optional PaddleOCR path | Reads code, slide, and UI text when captions are weak |
| Surya slide path | Surya layout/OCR predictors when installed, OCR-derived fallback otherwise | Produces slide title, block types, reading order, and duplicate-slide fingerprints |
| Cursor model | YOLO cursor detector with motion/template fallback | Links pointer movement to UI elements and macro regions |
| Change detection | SSIM, changed-pixel ratio, crop boxes, and mode thresholds | Avoids full reparse on unchanged frames |
| Topic linker | BGE-M3 embeddings plus time/mode heuristics | Groups screen events and transcript into coherent report sections |
| Report writers | Mock, Gemini, OpenRouter, local HF Mistral, Qwen | Supports deterministic smoke tests and production-quality Markdown |

## Demo-Mode Structure

Batch demo analysis writes a structured, auditable view of every detected screen state. The analyzer starts with macro regions, then runs OmniParser and OCR in the same changed crop set. Each macro region receives a top-to-bottom, left-to-right `children` list plus a flat `reading_order` list for report generation.

When OCR and OmniParser overlap, the structure preserves containment:

1. OmniParser element inside OCR text box: the UI element is nested under the OCR node.
2. OCR text inside a UI element: the OCR node is nested under the UI element.
3. Partial overlap: the UI node keeps the OCR text as attached evidence.

The change policy is the same as the live path: contained changes reparse only changed macro-region crops, while boundary-level changes trigger a fresh macro pass and a full-screen OmniParser/OCR parse.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r batch_mode/requirements.txt
cp batch_mode/.env.example batch_mode/.env
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r batch_mode\requirements.txt
Copy-Item batch_mode\.env.example batch_mode\.env
```

For a local smoke test with no LLM calls:

```env
BATCH_LLM_BACKEND=mock
BATCH_LLM_MOCK_ENABLED=1
BATCH_LLM_USE_PREWRITTEN=1
BATCH_EMBEDDING_DEVICE=cpu
OCR_MODE=cpu
```

For GPU OCR:

```env
OCR_MODE=gpu
OCR_BACKEND_GPU=easyocr
OCR_BACKEND_CPU=rapidocr
SURYA_ENABLED=1
```

## Missing Large Model

The OmniParser caption checkpoint is intentionally not committed:

```text
batch_mode/vendor/OmniParser/weights/icon_caption/model.safetensors
```

Download it after cloning:

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

Without this file, full OmniParser v2 will download the missing checkpoint when `HF_API_KEY` is available. With no token and no checkpoint, macro segmentation, cursor detection, OCR, and report assembly can still be tested, but OmniParser micro-element captioning is unavailable.

## Optional Surya Installation

Batch mode contains the full Surya predictor integration. It uses:

- `FoundationPredictor`
- `LayoutPredictor`
- `RecognitionPredictor`
- `DetectionPredictor`

Install the package that provides the `surya` module when slide layout/OCR quality matters:

```bash
pip install surya-ocr
```

If Surya is missing or `SURYA_ENABLED=0`, the slide analyzer falls back to OCR-derived layout blocks. The fallback still emits title, text/list/table/figure guesses, reading order, slide fingerprint, and confidence.

## Run Through The API

Start the service:

```bash
python -m uvicorn batch_mode.api:app --host 0.0.0.0 --port 8000
```

Analyze the Bunny sample locally:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4",
    "title": "Bunny validation sample",
    "persist_artifacts": true
  }'
```

Submit to a configured Slurm cluster:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4",
    "title": "Bunny cluster run",
    "persist_artifacts": true
  }'
```

Submit several videos:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/cluster/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequential": true,
    "items": [
      {
        "source": "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4",
        "title": "Bunny sample"
      }
    ]
  }'
```

## Run Directly On A Cluster

```bash
cd /path/to/Meet-Analyzer
cp batch_mode/.env.example batch_mode/.env
# edit batch_mode/.env with cluster paths, keys, partition, conda env, and writer backend

bash batch_mode/myproject/submit_cluster_source.sh \
  --writer mock \
  "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4" \
  bunny_480p.mp4
```

Cluster options are read from `batch_mode/.env`:

```env
BATCH_CLUSTER_HOST=
BATCH_CLUSTER_USER=
BATCH_CLUSTER_ACCOUNT=
BATCH_CLUSTER_CONDA_ENV=torch_clean
BATCH_CLUSTER_PARTITION=short
BATCH_CLUSTER_GRES=gpu:3g.40gb:1
BATCH_CLUSTER_PROJECT_DIR=/path/to/Meet-Analyzer
BATCH_CLUSTER_LOG_DIR=/path/to/Meet-Analyzer/batch_mode/myproject/logs
```

## Rebuild The Documentation Assets

```bash
python docs/generate_segment_validation.py \
  --source "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4" \
  --output-dir docs/assets \
  --times "2,58,100,145,205" \
  --weights real_time_mode/vendor/uitag/yolo-ui.pt \
  --device cpu

python docs/generate_micro_validation.py \
  --frames "docs/assets/frame_0002_raw.jpg,docs/assets/frame_0058_raw.jpg,docs/assets/frame_0100_raw.jpg,docs/assets/frame_0145_raw.jpg,docs/assets/frame_0205_raw.jpg" \
  --output-dir docs/assets \
  --repo-dir real_time_mode/vendor/OmniParser \
  --weights-dir real_time_mode/vendor/OmniParser/weights \
  --imgsz 512

python docs/build_readme_assets.py
```

## Verification

```bash
python -m pytest real_time_mode/tests batch_mode/tests -q
python -m py_compile real_time_mode/live_app.py real_time_mode/modal_app.py real_time_mode/config.py batch_mode/config.py batch_mode/api.py batch_mode/service.py
```

Current local result:

```text
47 passed, 4 third-party deprecation warnings
```
