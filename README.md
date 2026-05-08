# Meet Analyzer

Meet Analyzer turns screen-share meetings into structured, evidence-backed reports. It combines video sampling, speech-to-text, cursor tracking, page segmentation, UI parsing, OCR, slide/code understanding, topic grouping, and optional LLM writing.

The repository has two operating modes:

| Mode | Folder | Best for | Output |
| --- | --- | --- | --- |
| Batch analysis | [`batch_mode`](batch_mode) | Recorded meetings, evaluated demos, long presentations, cluster runs | `analysis.json`, `report.md`, validation assets |
| Real-time analysis | [`real_time_mode`](real_time_mode) | Live screen sharing, low-latency visual events, frontend monitoring | Live summaries, annotated frames, Supabase-ready event payloads |

## Project Shape

```text
Meet-Analyzer
+-- batch_mode/          # Offline analyzer, FastAPI service, Slurm submitter
+-- real_time_mode/      # Live FastAPI app, frontend, Modal deployment
+-- docs/assets/         # Bunny validation screenshots, metrics, UI proof images
+-- docs/examples/       # Example reports and analysis snapshots
`-- .env.example         # Minimal root template; use mode-specific templates for real runs
```

## Visual Proof

The current documentation assets are generated from the Bunny sample video:

```text
https://vz-e4eb54a1-90d.b-cdn.net/39864ba9-c74e-4cfa-a088-58ec2a08c699/play_480p.mp4
```

<p align="center">
  <img src="docs/assets/segment_validation_contact_sheet.jpg" alt="Bunny sample raw frames and generated page segments" width="920">
</p>

<p align="center">
  <img src="docs/assets/realtime_frontend_dashboard.png" alt="Real-time frontend dashboard" width="920">
</p>

## Quick Start

Use the mode-specific README for full setup:

- [`batch_mode/README.md`](batch_mode/README.md)
- [`real_time_mode/README.md`](real_time_mode/README.md)

Install the Python dependencies from the batch requirements file, which covers the shared runtime stack:

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

Copy one of the mode-specific environment templates:

```bash
cp batch_mode/.env.example batch_mode/.env
cp real_time_mode/.env.example real_time_mode/.env
```

The committed examples are safe templates. Real `.env` files are ignored by git.

## Included Model Artifacts

Small and practical model files are kept in the repository. Very large files are documented in the mode READMEs with download commands.

| Artifact | Path | Purpose |
| --- | --- | --- |
| yolo-ui detector | `real_time_mode/vendor/uitag/yolo-ui.pt` | Macro page segments: header, sidebar, main content, footer-like regions |
| Cursor detector | `batch_mode/cursor_class_best.pt`, `real_time_mode/cursor_class_best.pt` | Cursor box, hotspot, and pointer context |
| OmniParser icon detector | `batch_mode/vendor/OmniParser/weights/icon_detect/model.pt`, `real_time_mode/vendor/OmniParser/weights/icon_detect/model.pt` | UI element detection |
| OmniParser metadata/config | `batch_mode/vendor/OmniParser/weights/**`, `real_time_mode/vendor/OmniParser/weights/**` | Detector and caption configuration |

## Verification Snapshot

Current local verification:

```text
45 passed, 4 third-party deprecation warnings
```

The test suite covers batch report assembly, page segmentation, cursor handling, real-time mode routing, stale-frame dropping, and frontend concurrency behavior.

## Repository Hygiene

- `.env` files, generated logs, local videos, cluster job artifacts, and temporary outputs are ignored.
- Mode-specific `.env.example` files are committed.
- Validation screenshots and metrics under `docs/assets` are committed because they document the expected behavior.
