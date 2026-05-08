# Real-Time Mode

Real-time mode is the live screen-share analyzer. It captures frames in the browser, performs lightweight client-side change detection, sends only meaningful updates to the backend, and returns annotated visual events for a live meeting timeline.

<p align="center">
  <img src="../docs/assets/realtime_frontend_dashboard.png" alt="Real-time frontend dashboard" width="920">
</p>

## What It Does

- Captures screen-share frames directly in the browser.
- Runs a Web Worker diff pass before sending a frame.
- Sends changed frames with `client_frame_id` sequencing.
- Allows limited concurrent in-flight frame requests.
- Drops or ignores stale responses when a newer frame has already superseded them.
- Detects screen mode: demo, slide, code, or other.
- Tracks cursor location and cursor target.
- Segments the page into macro regions.
- Parses UI elements with OmniParser and OCR.
- Produces summaries, annotated images, and Supabase-ready payloads.

The practical result is that switching browser tabs during a screen share no longer waits behind a long queue of older frames. Newer full-screen changes can supersede stale work.

## Architecture

```text
Browser screen capture
  -> low-resolution Web Worker diff
  -> latest-frame scheduler
  -> FastAPI local endpoint
  -> optional Modal GPU endpoint
  -> mode router
  -> cursor detector
  -> macro page segmentation
  -> OmniParser / OCR / slide parser
  -> summary and annotated frame
  -> frontend display and optional Supabase persistence
```

## Models And Methods

| Component | Model / method | Purpose |
| --- | --- | --- |
| Capture scheduler | Browser canvas + JPEG compression | Sends screen frames at mode-specific intervals |
| Client diff | Web Worker grayscale pixel diff | Skips unchanged frames before network upload |
| Latest-frame protocol | `client_frame_id`, abort signals, stale response checks | Prevents 20-25 second catch-up queues after tab switches |
| Mode router | SigLIP and sticky routing | Classifies screens as demo, slide, code, or other |
| Macro segmentation | yolo-ui detector | Produces `header`, `left_sidebar`, and `main_content` when geometry supports them |
| UI parsing | OmniParser icon detector/caption stack | Extracts controls and visible UI elements |
| OCR | EasyOCR / RapidOCR adapter | Reads text for code, slides, and UI screens |
| Cursor | YOLO cursor detector plus fallback logic | Tracks pointer hotspot and target element |
| Live summary | Heuristic or Gemini-backed summarizer | Describes what changed on screen |
| Persistence | Supabase REST tables | Stores meetings, frame events, and transcript segments |
| Remote GPU | Modal app | Runs heavier visual parsing off the local frontend host |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r batch_mode/requirements.txt
cp real_time_mode/.env.example real_time_mode/.env
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r batch_mode\requirements.txt
Copy-Item real_time_mode\.env.example real_time_mode\.env
```

Edit `real_time_mode/.env` with API keys and optional Supabase/Modal settings.

## Missing Large Model

The OmniParser caption model is not committed because it is about 1 GB:

```text
real_time_mode/vendor/OmniParser/weights/icon_caption/model.safetensors
```

Download it when semantic icon captions are needed:

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_caption/model.safetensors",
    repo_type="model",
    token=os.environ.get("HF_API_KEY", ""),
    local_dir="real_time_mode/vendor/OmniParser/weights",
    local_dir_use_symlinks=False,
)
print("Downloaded OmniParser caption model.")
PY
```

The committed icon detector and OCR path still allow functional parsing without this file.

## Run Locally

```bash
python -m uvicorn real_time_mode.live_app:app --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Use **Start Session + Share**. The app requests screen-share permission first, warms the runtime, and then starts capture.

## Modal GPU Endpoint

Deploy the Modal app:

```bash
modal deploy real_time_mode/modal_app.py
```

Then set one of:

```env
MODAL_API_BASE_URL=https://<modal-app-url>
```

or:

```env
MODAL_PROCESS_FRAME_URL=https://<modal-app-url>/api/process-frame
```

The local frontend always posts to `/api/process-frame`; the local FastAPI app proxies to Modal when configured.

## Supabase Tables

The schema lives in:

```text
real_time_mode/supabase/schema.sql
```

Apply it in Supabase SQL editor, then set:

```env
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=
```

The frontend can write:

- `meetings`
- `frame_events`
- `transcript_segments`

## Latency Controls

Default real-time settings are tuned for responsiveness:

| Setting | Default | Why it matters |
| --- | ---: | --- |
| `VITE_CAPTURE_MAX_SIDE` | `512` | Reduces model input and network payload size |
| `VITE_FRAME_JPEG_QUALITY` | `0.46` | Keeps frames readable but compact |
| `VITE_MAX_IN_FLIGHT_FRAMES` | `2` | Allows fresh frames to move while older work is still finishing |
| `VITE_FULL_FRAME_CHANGE_RATIO` | `0.55` | Treats tab switches as superseding changes |
| `OMNIPARSER_IMGSZ` | `512` | Faster UI parsing for live updates |

The backend also rejects stale frames before and after the session lock, so superseded work does not overwrite newer UI state.

## Frontend Features

- Live capture and screen preview.
- Frame counters: sent, queued, skipped, dropped stale, in-flight.
- Latest frame gap and analysis lag.
- Current mode and confidence.
- Transcript controls.
- Annotated input and output images.
- Macro layout and micro element counts.
- Cursor target display.
- Raw JSON panel hidden by default to avoid unnecessary rendering cost.
- Debug timings for route, cursor, macro layout, OmniParser, summary, annotation, proxy, and endpoint total.

## Page Segmentation Proof

The real-time mode shares the same macro segmenter as batch mode. Current Bunny validation is shown below:

<p align="center">
  <img src="../docs/assets/segment_validation_contact_sheet.jpg" alt="Bunny page segmentation validation" width="920">
</p>

The current validation emits `header`, `left_sidebar`, and `main_content`. It intentionally suppresses unsupported right/sidebar labels when the geometry does not prove a coherent rail.

## Tests

Run the real-time tests:

```bash
python -m pytest real_time_mode/tests/test_live_latest_frames.py real_time_mode/tests/test_macro_layout.py real_time_mode/tests/test_pipeline_modes.py -q
```

Run the broader smoke set:

```bash
python -m pytest real_time_mode/tests/test_live_latest_frames.py real_time_mode/tests/test_macro_layout.py real_time_mode/tests/test_pipeline_modes.py batch_mode/tests/test_batch_mode.py -q
```

Current result:

```text
45 passed, 4 third-party deprecation warnings
```

Frontend verification was run with Playwright against the live app shell using mocked screen-share frames and mocked slow processing responses. The test sent frame IDs `[1, 2, 3]`, observed `maxConcurrent = 2`, applied frame `3` before the intentionally slow first response, and verified that the segment display stayed on `main_content` without a false `left_sidebar`.

## Operational Notes

- Use local mode for development and UI testing.
- Use Modal for lower latency when the local machine is CPU-bound.
- Keep `.env` out of git; commit only `.env.example`.
- For production demos, keep the raw JSON panel collapsed unless debugging.
- If tab switching feels slow, check the **Dropped stale**, **In flight**, and **Analysis lag** counters first.
