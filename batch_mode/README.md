# Batch Mode

Batch mode is the offline pipeline for recorded screen-share meetings. It converts a public MP4 URL or local video into structured visual events, transcript-linked state changes, frame-level descriptions, topic units, and a final Markdown report.

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
  -> macro page segmentation with yolo-ui
  -> DocLayout-YOLO block extraction
  -> OmniParser micro-element extraction
  -> OCR / Surya slide layout / code text extraction
  -> changed-region cache and state merge
  -> hierarchical visual evidence: macro region > DocLayout/OmniParser/OCR containment > cursor
  -> transcript-window alignment
  -> frame-level LLM descriptions
  -> topic segmentation and global topic linking
  -> report section builder
  -> mock, Gemini, OpenRouter, local HF Mistral, or Qwen writer
  -> analysis.json + report.md
```

The key idea is evidence reuse. Stable frames are not treated as brand-new images. The analyzer caches macro layout, reparses only changed crops when safe, carries forward unchanged UI elements, and attaches each visual event to the transcript window that happened since the previous visual change.

## Step-By-Step Processing

### Local/API Mode

1. Resolve the video source. Public URLs are downloaded to `BATCH_TEMP_DIR`; local files are used directly.
2. Transcribe with Deepgram `nova-3` unless transcript segments are supplied in the request.
3. Sample frames using `BATCH_SAMPLE_FPS` plus the mode-specific intervals.
4. Route each frame with SigLIP into `code`, `slide`, `demo`, or `other`.
5. Detect and mask the cursor so cursor motion does not create false visual diffs.
6. For code frames, run OCR plus DocLayout-YOLO over the full frame, diff OCR against the previous code frame, and build a code state.
7. For slide frames, run Surya layout/OCR plus DocLayout-YOLO, fingerprint the slide, and skip duplicate slides.
8. For demo frames, run yolo-ui macro segmentation, then run DocLayout-YOLO, OmniParser, and OCR on changed crops in parallel.
9. Merge newly parsed elements with the cached state, assign each element to a macro region, and build the hierarchy: macro segment, then contained DocLayout/OmniParser/OCR nodes, then cursor.
10. Attach the transcript window since the last visual event.
11. Run the batch frame-level LLM pass when `BATCH_FRAME_DESCRIPTIONS_ENABLED=1`.
12. Build topic units with BGE-M3 embeddings, link related units globally, build report sections, and generate `report.md`.

### Cluster Mode

1. The API or `submit_cluster_source.sh` stages the source into `batch_mode/cluster_jobs/<request_id>/`.
2. The login/submission side prepares `request.json`; when needed it also transcribes before the Slurm job.
3. The Slurm job exports model and writer settings, including DocLayout and frame-description settings.
4. `batch_mode/myproject/test.py` runs the same local pipeline on the compute node.
5. Results are written to the cluster job directory and can be fetched through `/api/jobs/{request_id}/result`, `/report`, and `/artifacts`.

## Model Outputs

| Model | Output in `analysis.json` | Example image |
| --- | --- | --- |
| SigLIP router | `confidence.route` with mode scores and decision path | Mode labels appear in every frame event |
| Cursor YOLO/motion backend | `cursor`, `visual_payload.cursor_target`, `visual_payload.cursor_action` | `../docs/assets/realtime_cursor_output.jpg` |
| yolo-ui macro layout | `visual_payload.macro_layout`, `layout_groups` | `../docs/assets/segment_validation_contact_sheet.jpg` |
| DocLayout-YOLO DocStructBench | `visual_payload.doclayout_elements`, `confidence.doclayout_provider` | `../docs/assets/realtime_doclayout_ocr_current_output.jpg` |
| OmniParser v2 | `visual_payload.micro_elements` with `source=omniparser...` | `../docs/assets/micro_validation_contact_sheet.jpg` |
| OCR | `visual_payload.ocr_text`, `ocr_lines`, code OCR diffs | `../docs/assets/realtime_doclayout_ocr_current_output.jpg` |
| Surya slide analyzer | `visual_payload.layout`, `full_text`, slide fingerprint | Slide blocks are included in slide frame events |
| BGE-M3 embeddings | `topic_units`, `topic_clusters` | Semantic grouping is stored in JSON rather than an image |
| Frame/report LLMs | `visual_payload.llm_detail`, `llm_output.final_markdown` | Text output in `analysis.json` and `report.md` |

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
| `visual_payload.llm_detail` | Batch-only detailed frame descriptions, demo region descriptions, segment relationships, and page-change summary |
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
| DocLayout-YOLO | `juliozhao/DocLayout-YOLO-DocStructBench` | Adds document/UI block evidence such as text blocks, titles, tables, figures, formulas, captions, and lists |
| Coverage fallback | Grid coverage assignment to `main_content` | Ensures every important screen area belongs to a macro region |
| Micro parsing | OmniParser v2 icon detector and caption stack | Extracts controls, icons, text boxes, and visible UI elements |
| OCR | EasyOCR on GPU by default, RapidOCR for CPU throughput, optional PaddleOCR path | Reads code, slide, and UI text when captions are weak |
| Surya slide path | Surya layout/OCR predictors when installed, OCR-derived fallback otherwise | Produces slide title, block types, reading order, and duplicate-slide fingerprints |
| Cursor model | YOLO cursor detector with motion/template fallback | Links pointer movement to UI elements and macro regions |
| Change detection | SSIM, changed-pixel ratio, crop boxes, and mode thresholds | Avoids full reparse on unchanged frames |
| Frame descriptions | Mock/Gemini/OpenRouter/vLLM/Ollama/local HF writer | Generates detailed code, slide, and demo-region descriptions after visual processing |
| Topic linker | BGE-M3 embeddings plus time/mode heuristics | Groups screen events and transcript into coherent report sections |
| Report writers | Mock, Gemini, OpenRouter, local HF Mistral, Qwen | Supports deterministic smoke tests and production-quality Markdown |

Model-specific modifications and post-processing:

- yolo-ui is not used as raw boxes only. It is tiled, class-aware NMS is applied, coarse regions are converted into `header`, sidebars, footer, and `main_content`, and visual boundary evidence refines region edges.
- DocLayout-YOLO labels are normalized (`plain_text` -> `text_block`, captions -> `caption`, formulas/lists/tables kept), background-like labels are dropped, and duplicate layout blocks are removed by high overlap.
- OmniParser crop detections are rebased from crop coordinates to full-frame coordinates, normalized into UI element types, and merged with cached elements by IoU/text continuity.
- OCR is converted into structured text elements and filtered into changed regions when a full OCR fallback is used.
- Surya slide output is fingerprinted so repeated slides do not generate duplicate visual events.
- Cursor detections are associated with the smallest/highest-confidence element under the hotspot, with a macro-region fallback when no exact element is hit.
- LLM prompts are evidence-first: code/slide prompts receive OCR, DocLayout, diffs, cursor, transcript window, and previous frame context; demo prompts receive macro regions, nested visual hierarchy, spacing summaries, cursor target, and previous-page evidence.

## Demo-Mode Structure

Batch demo analysis writes a structured, auditable view of every detected screen state. The analyzer starts with macro regions, then runs DocLayout-YOLO, OmniParser, and OCR in the same changed crop set. Each macro region receives a top-to-bottom, left-to-right `children` list, a nested `hierarchy` list, and a flat `reading_order` list for report generation.

The hierarchy is:

1. Macro segment on top: `header`, `left_sidebar`, `main_content`, `right_sidebar`, `footer`.
2. DocLayout-YOLO blocks, OmniParser elements, and OCR lines are interchangeable evidence nodes. Larger boxes that contain smaller boxes become parents regardless of source.
3. Cursor evidence is appended at the exact target element when the cursor hotspot lands on a detected element; otherwise it is attached to the macro region.

When OCR, DocLayout, and OmniParser overlap, the structure preserves containment:

1. OmniParser element inside OCR text box: the UI element is nested under the OCR node.
2. OCR text inside a UI element: the OCR node is nested under the UI element.
3. DocLayout block containing OCR or OmniParser detections: the block becomes the parent layout node.
4. Partial overlap: the UI node keeps the nearby text as attached evidence.

The change policy is the same as the live path: contained changes reparse only changed macro-region crops, while boundary-level changes trigger a fresh macro pass and a full-screen OmniParser/OCR parse.

## Frame-Level LLM Pass

Batch mode keeps the visual processing path unchanged until after frame events have been produced and transcript windows have been attached. Then `BATCH_FRAME_DESCRIPTIONS_ENABLED=1` runs a second, offline LLM pass:

1. Code frames: one detailed LLM call describes the editor/terminal layout, visible code text, OCR diff, cursor focus, and likely change from the previous frame.
2. Slide frames: one detailed LLM call describes title/body layout, lists, figures/tables, DocLayout blocks, Surya/OCR text, cursor focus, and change from the previous frame.
3. Demo frames: `header`, `left_sidebar`, `right_sidebar`, and `footer` each get their own parallel LLM call when present.
4. Demo `main_content`: large gaps between layout/OCR/OmniParser nodes split the main area into up to `BATCH_DEMO_MAIN_MAX_SEGMENTS` subregions. Each subregion gets its own parallel LLM call.
5. Demo relationships: one LLM call explains how the segments relate spatially.
6. Demo bottom summary: one LLM call explains what is happening, what changed from the previous page/frame, cursor flow, and the likely reason for the change.
7. The final demo output is structured data, not another combination LLM call: `header`, `left_sidebar`, `main_content`, `right_sidebar`, `footer`, `relationships`, and `page_summary`.

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
OCR_MODE=gpu
```

For GPU OCR:

```env
OCR_MODE=gpu
OCR_BACKEND_GPU=easyocr
OCR_BACKEND_CPU=rapidocr
SURYA_ENABLED=1
DOCLAYOUT_ENABLED=1
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

DocLayout-YOLO downloads its checkpoint through Hugging Face when `DOCLAYOUT_MODEL_REF` is empty:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
)
print(path)
PY
```

PowerShell:

```powershell
@'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
)
print(path)
'@ | python -
```

To pin a local DocLayout checkpoint on the cluster:

```env
DOCLAYOUT_MODEL_REF=/path/to/doclayout_yolo_docstructbench_imgsz1024.pt
DOCLAYOUT_DEVICE=auto
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
    "persist_artifacts": true,
    "writer": "mock",
    "describe_frames": true
  }'
```

Submit to a configured Slurm cluster:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4",
    "title": "Bunny cluster run",
    "persist_artifacts": true,
    "writer": "gemini",
    "describe_frames": true
  }'
```

Submit several videos:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/cluster/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequential": true,
    "writer": "gemini",
    "requests": [
      {
        "source": "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4",
        "title": "Bunny sample",
        "describe_frames": true
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
DOCLAYOUT_MODEL_REF=/path/to/doclayout_yolo_docstructbench_imgsz1024.pt
BATCH_FRAME_DESCRIPTIONS_ENABLED=1
BATCH_FRAME_DESCRIPTION_WORKERS=6
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
python -m pytest batch_mode/tests -q
python -m py_compile batch_mode/config.py batch_mode/api.py batch_mode/service.py batch_mode/pipeline.py batch_mode/llm_adapters.py batch_mode/doclayout_adapter.py
```

Current local result:

```text
24 passed, 4 third-party deprecation warnings
```
