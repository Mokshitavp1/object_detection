import json
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from zone import DetectionZone
from alerting import send_alert

from detect import (
    draw_boxes,
    encode_frame,
    get_location,
    load_config,
    log_event,
    open_camera,
    run_inference,
    save_snapshot,
)

# ---------------------------------------------------------------------------
# Shared state — importable by route handlers
# ---------------------------------------------------------------------------
latest_frame: bytes | None = None
frame_lock = Lock()

# ---------------------------------------------------------------------------
# Detection loop — runs in a daemon thread
# ---------------------------------------------------------------------------
def detection_loop(config: dict, model, cap: cv2.VideoCapture, lat: float, lon: float) -> None:
    global latest_frame
    cooldowns: dict[str, float] = {}
    zone = DetectionZone(config.get("zone_polygon", []))


    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            detections = run_inference(frame, model, config)
            frame = draw_boxes(frame, detections)

            with frame_lock:
                latest_frame = encode_frame(frame)

            now = time.time()
            for det in detections:
                if not zone.is_inside(det["bbox"]):
                    continue
                cls = det["class"]
                if now - cooldowns.get(cls, 0) < config["cooldown_seconds"]:
                    continue
                cooldowns[cls] = now
                snap_name = None
                if config.get("save_snapshots"):
                    snap_name = save_snapshot(frame, det, Path("snapshots"))
                log_event(det, lat, lon, snap_name, Path("events.jsonl"))
                send_alert(Path("snapshots") / snap_name if snap_name else None, det)

        except Exception as e:
            print(f"Detection error: {e}")


def start_detection(config_path: Path = Path("config.yaml")) -> None:
    config = load_config(config_path)
    lat, lon = get_location()
    print(f"Location: {lat}, {lon}")
    model = YOLO("yolov8n.pt")           # downloads on first run
    cap = open_camera(config["camera_source"])
    t = threading.Thread(target=detection_loop, args=(config, model, cap, lat, lon), daemon=True)
    t.start()
    print("Detection thread started.")

# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("snapshots").mkdir(exist_ok=True)
    start_detection(Path("config.yaml"))
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# MJPEG stream generator
# ---------------------------------------------------------------------------
def mjpeg_generator():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps cap

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _read_events() -> list[dict]:
    """Return all valid parsed events from events.jsonl, oldest first."""
    log_path = Path("events.jsonl")
    if not log_path.exists():
        return []
    events = []
    for line in log_path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except:  # noqa: E722 — silently skip malformed lines
            pass
    return events


@app.get("/events")
def get_events(
    cls: str | None = Query(default=None, alias="class"),
    limit: int = Query(default=50, ge=1),
):
    events = _read_events()
    if cls is not None:
        events = [e for e in events if e.get("class") == cls]
    return JSONResponse(content=list(reversed(events[-limit:])))


@app.get("/stats")
def get_stats():
    today = datetime.utcnow().date().isoformat()
    today_events = [e for e in _read_events() if e.get("timestamp", "").startswith(today)]
    if not today_events:
        return JSONResponse(content={
            "total_today": 0,
            "most_seen_class": None,
            "last_detection_time": None,
        })
    counter = Counter(e["class"] for e in today_events)
    return JSONResponse(content={
        "total_today": len(today_events),
        "most_seen_class": counter.most_common(1)[0][0],
        "last_detection_time": today_events[-1]["timestamp"],
    })
