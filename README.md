# Meet Analyzer

Meet Analyzer turns screen-share meetings into structured, evidence-backed reports. It is built for the hard part of meeting understanding: connecting what was said with what was visible, what changed on screen, what UI regions mattered, and which evidence should survive into a final report.

The repository contains two analyzers that share the same visual understanding core:

| Mode | Folder | Primary use | Output |
| --- | --- | --- | --- |
| Batch analysis | [`batch_mode`](batch_mode) | Recorded meetings, demos, lecture walkthroughs, cluster evaluation | `analysis.json`, `report.md`, validation assets |
| Real-time analysis | [`real_time_mode`](real_time_mode) | Live screen sharing, low-latency visual events, frontend monitoring | Live summaries, annotated frames, Supabase-ready event payloads |

## Research Goal

The project treats a meeting as a multimodal state-estimation problem rather than a plain transcription task. Each processed frame is converted into a structured screen state:

- Macro regions: page header, left navigation/sidebar, main content, and coverage fallback regions.
- Micro elements: buttons, icons, text boxes, tabs, inputs, links, and OCR/caption text.
- Cursor state: pointer box, hotspot, movement, and nearest visual target.
- Screen mode: demo, slide, code, or other.
- Transcript alignment: speech windows attached to visual changes.
- Evidence-preserving report sections: timeline, screen flow, topics, actions, and transcript.

<p align="center">
  <img src="docs/assets/architecture_overview.png" alt="Meet Analyzer pipeline" width="920">
</p>

## Evidence Assets

All screenshots below were generated from the Bunny validation sample:

```text
https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4
```

The macro sheet uses five visually different states from the recording: title screen, game rules, simulation environment, agent table, and algorithm logic. The duplicate 15s title-frame sample was removed.

<p align="center">
  <img src="docs/assets/segment_validation_contact_sheet.jpg" alt="Bunny macro segment validation" width="920">
</p>

The micro sheet runs OmniParser on the same frames and draws element-level boxes.

<p align="center">
  <img src="docs/assets/micro_validation_contact_sheet.jpg" alt="Bunny micro element validation" width="920">
</p>

## Validation Snapshot

Macro page segmentation:

| Metric | Value |
| --- | ---: |
| Sampled Bunny frames | `5` |
| Sample timestamps | `2s, 58s, 100s, 145s, 205s` |
| Mean grid coverage | `100.00%` |
| High-confidence regions | `11` |
| Low-confidence retained regions | `4` |
| Detected macro classes | `header`, `left_sidebar`, `main_content` |

<p align="center">
  <img src="docs/assets/metrics_dashboard.png" alt="Macro validation metrics" width="920">
</p>

Micro element parsing:

| Metric | Value |
| --- | ---: |
| Parsed Bunny frames | `5` |
| OmniParser elements | `296` |
| Mean elements per frame | `59.2` |
| Interactive elements | `259` |
| Elements with text or captions | `296` |

<p align="center">
  <img src="docs/assets/micro_metrics_dashboard.png" alt="Micro validation metrics" width="920">
</p>

## Model Stack

| Layer | Implementation | Role |
| --- | --- | --- |
| Screen mode routing | SigLIP/SigLIP2 prompt bank with positive and negative prompts | Routes frames into demo, slide, code, or other modes |
| Macro segmentation | `swaylenhayes/uitag-yolo11s-ui-detect-v1` weights, committed as `real_time_mode/vendor/uitag/yolo-ui.pt` | Detects coarse page regions and rejects unsupported sidebar geometry |
| Micro UI parsing | OmniParser v2 icon detector plus OCR/caption content | Extracts clickable controls, icons, visible text, and UI element boxes |
| OCR | EasyOCR on GPU by default, RapidOCR on CPU, optional PaddleOCR path in the shared adapter | Reads code, slide, and UI text when captions are insufficient |
| Slide understanding | Batch mode uses Surya layout/OCR when installed, then falls back to OCR-derived layout | Produces slide blocks, reading order, title, fingerprint, and duplicate-slide checks |
| Cursor tracking | YOLO cursor detector plus motion/template fallback | Tracks pointer location and attaches cursor actions to UI targets |
| Change detection | SSIM, changed-pixel ratio, region crops, and mode-specific thresholds | Avoids reparsing stable frames and keeps real-time latency low |
| Topic linking | BGE-M3 embeddings with temporal heuristics | Groups visual and transcript evidence into coherent report units |
| Speech-to-text | Deepgram prerecorded or live transcript adapters | Provides transcript windows for event summaries |
| Report writing | Mock writer, Gemini, OpenRouter, local HF Mistral/Qwen | Turns structured evidence into Markdown |

Demo-mode events preserve both visual structure and text evidence: macro regions contain a top-to-bottom, left-to-right reading order of OmniParser elements and OCR lines. Overlapping OCR/UI boxes are nested by containment, so reports and downstream agents can read “what text was visible” and “which control it belonged to” from one coherent payload.

## Repository Shape

```text
Meet-Analyzer
+-- batch_mode/          # Offline analyzer, FastAPI API, Slurm submitter
+-- real_time_mode/      # Live FastAPI app, frontend, Modal deployment
+-- docs/assets/         # Macro/micro validation images and metric dashboards
+-- docs/examples/       # Example reports and analysis snapshots
`-- .env.example         # Minimal root template; mode-specific examples are preferred
```

## Quick Start

Install the shared Python stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r batch_mode/requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r batch_mode\requirements.txt
```

Copy the mode-specific environment templates:

```bash
cp batch_mode/.env.example batch_mode/.env
cp real_time_mode/.env.example real_time_mode/.env
```

Real `.env` files are ignored by git.

## Included And External Artifacts

Small model files that make the project runnable are committed. Very large caption checkpoints are documented in the mode READMEs.

| Artifact | Path | Purpose |
| --- | --- | --- |
| yolo-ui detector | `real_time_mode/vendor/uitag/yolo-ui.pt` | Macro page segments |
| Cursor detector | `batch_mode/cursor_class_best.pt`, `real_time_mode/cursor_class_best.pt` | Cursor box and hotspot detection |
| OmniParser icon detector | `batch_mode/vendor/OmniParser/weights/icon_detect/model.pt`, `real_time_mode/vendor/OmniParser/weights/icon_detect/model.pt` | UI element detection |
| OmniParser config files | `*/vendor/OmniParser/weights/icon_*/*.json`, `*.yaml` | Detector and caption model metadata |

Not committed:

```text
*/vendor/OmniParser/weights/icon_caption/model.safetensors
```

That file is about 1 GB and should be downloaded after clone when full OmniParser semantic captions are required.

## Reproduce The Validation Assets

When the Bunny URL is reachable:

```bash
python docs/generate_segment_validation.py \
  --source "https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4" \
  --output-dir docs/assets \
  --times "2,58,100,145,205" \
  --weights real_time_mode/vendor/uitag/yolo-ui.pt \
  --device cpu
```

When the raw frames are already present:

```bash
python docs/generate_segment_validation.py \
  --raw-frames "docs/assets/frame_0002_raw.jpg,docs/assets/frame_0058_raw.jpg,docs/assets/frame_0100_raw.jpg,docs/assets/frame_0145_raw.jpg,docs/assets/frame_0205_raw.jpg" \
  --output-dir docs/assets \
  --weights real_time_mode/vendor/uitag/yolo-ui.pt \
  --device cpu
```

Micro validation requires the OmniParser caption model to be present locally:

```bash
python docs/generate_micro_validation.py \
  --frames "docs/assets/frame_0002_raw.jpg,docs/assets/frame_0058_raw.jpg,docs/assets/frame_0100_raw.jpg,docs/assets/frame_0145_raw.jpg,docs/assets/frame_0205_raw.jpg" \
  --output-dir docs/assets \
  --repo-dir real_time_mode/vendor/OmniParser \
  --weights-dir real_time_mode/vendor/OmniParser/weights \
  --imgsz 512

python docs/build_readme_assets.py
```

## Verification

Current local verification:

```text
47 passed, 4 third-party deprecation warnings
```

The tests cover batch report assembly, page segmentation, cursor handling, mode routing, stale-frame dropping, frontend concurrency behavior, crop rebasing, sidebar false-positive guards, and the Surya enable/disable flag.

## More Details

- [`batch_mode/README.md`](batch_mode/README.md)
- [`real_time_mode/README.md`](real_time_mode/README.md)
