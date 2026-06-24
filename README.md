# Object Detection Dashboard

Real-time object detection with a live browser dashboard, built on YOLOv8, OpenCV, and FastAPI.

## Features

- Real-time webcam (or RTSP) capture with YOLOv8n inference on every motion-triggered frame
- **Motion gating** — inference is skipped entirely when no motion is detected, saving CPU/GPU cycles
- **Polygon detection zones** — restrict detections to a region of interest via bounding-box centre containment
- Live annotated MJPEG video stream served straight to the browser
- Per-class **cooldown** to suppress duplicate logging of the same object class in quick succession
- Append-only **JSON Lines** event log (`events.jsonl`) with timestamp, class, confidence, bounding box, and geolocation
- Automatic **snapshot capture** (JPG) for every logged detection
- Startup **geolocation lookup** (via ip-api.com) stamped onto every event
- Browser **dashboard** — live feed, summary stat cards (total detections today, top class, last seen), class filter, and a scrollable event log with snapshot thumbnails

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Detection | YOLOv8n (Ultralytics) | Object inference on each frame |
| Frame capture | OpenCV (`cv2`) | Webcam / RTSP stream access |
| Motion gating | OpenCV frame differencing | Skip inference when no motion is present |
| Zone filtering | OpenCV (`cv2.pointPolygonTest`) | Restrict detections to a polygon region |
| Backend | FastAPI + Uvicorn | REST API, MJPEG stream, event endpoints |
| Frontend | Vanilla HTML + JS | Live feed, log table, stats cards |
| Storage | JSON Lines (`.jsonl`) | Append-only detection event log |
| Geolocation | ip-api.com | Lat/lon stamped on each event at startup |
| Config | `config.yaml` | User-facing settings |

## Quick Start

```bash
git clone <repo-url> && cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser.

> The first run downloads the `yolov8n.pt` model weights automatically.

## Configuration Reference

All settings live in `config.yaml`, loaded at startup.

| Key | Default | Description |
|---|---|---|
| `classes` | `["person", "dog", "cat"]` | COCO class names to detect |
| `confidence_threshold` | `0.5` | Minimum confidence score to log |
| `cooldown_seconds` | `30` | Per-class alert suppression window |
| `camera_source` | `0` | Webcam index or RTSP URL string |
| `save_snapshots` | `true` | Save full frame JPG on each detection |
| `motion_pixel_threshold` | `5000` | Non-zero pixel count required to trigger inference |
| `detection_zone` | `[]` | Polygon points `[[x,y], ...]`; empty = full frame |

## Project Structure

```
project/
├── detect.py            # Core detection loop, YOLOv8 inference, event logging
├── server.py             # FastAPI app, MJPEG stream, /events and /stats routes
├── motion.py              # MotionDetector class — frame differencing
├── zone.py                 # DetectionZone class — polygon filtering via cv2
├── config.yaml             # User-facing configuration
├── templates/
│   └── index.html          # Dashboard — live feed, stats cards, detection log table
├── snapshots/               # Auto-created; stores detection snapshot JPGs
├── events.jsonl              # Auto-created; append-only detection event log
└── requirements.txt           # Pinned Python dependencies
```

## Roadmap

- [ ] Multi-camera support (RTSP streams with camera ID per event)
- [ ] Object tracking across frames
- [ ] Activity heatmap (where objects appear most)
- [ ] Face blur / privacy mode for recordings
- [ ] Edge deployment (Raspberry Pi / Jetson)

## License

MIT
