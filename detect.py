import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests
import yaml
from ultralytics import YOLO

LOCATION_API_URL = "http://ip-api.com/json"
LOCATION_TIMEOUT = 5
# pyrefly: ignore [missing-import]
from motion import MotionDetector
from zone import DetectionZone

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{path.resolve()}'. "
            "Ensure config.yaml exists in the working directory."
        )
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_location() -> tuple[float, float]:
    try:
        response = requests.get(LOCATION_API_URL, timeout=LOCATION_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return float(data["lat"]), float(data["lon"])
    except Exception as exc:
        print(f"[WARNING] Could not determine location: {exc}. Falling back to (0.0, 0.0).")
        return 0.0, 0.0


def open_camera(source: int | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")
    return cap


def run_inference(frame, model, config: dict) -> list[dict]:
    result = model(frame, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        if class_name not in config["classes"] or confidence < config["confidence_threshold"]:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        detections.append({"class": class_name, "confidence": confidence, "bbox": [x1, y1, x2, y2]})
    return detections


def draw_boxes(frame, detections: list[dict]):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def encode_frame(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def save_snapshot(frame, detection: dict, snap_dir: Path) -> str:
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"snap_{ts}_{detection['class']}.jpg"
    cv2.imwrite(str(snap_dir / filename), frame)
    return filename


def log_event(
    detection: dict,
    lat: float,
    lon: float,
    snapshot_name: str | None,
    log_path: Path = Path("events.jsonl"),
) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "class": detection["class"],
        "confidence": round(detection["confidence"], 4),
        "bbox": detection["bbox"],
        "lat": lat,
        "lon": lon,
        "camera_id": "cam_0",
        "snapshot": snapshot_name,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    config = load_config(Path("config.yaml"))
    lat, lon = get_location()
    print(f"Config loaded. Location: {lat}, {lon}")
    model = YOLO("yolov8n.pt")          # downloads on first run
    cap = open_camera(config["camera_source"])
    motion_detector = MotionDetector(
        pixel_threshold=config.get("motion_pixel_threshold", 5000)
    )
    zone = DetectionZone(config.get("detection_zone", []))
    cooldowns: dict[str, float] = {}
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame — skipping.")
                continue
            if not motion_detector.has_motion(frame):
                cv2.imshow("Detection Feed", frame)      # still show the raw frame
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue                                 # skip inference entirely
            detections = run_inference(frame, model, config)
            frame = draw_boxes(frame, detections)
            now = time.time()
            for det in detections:
                if not zone.is_inside(det["bbox"]):     # zone filter
                    continue
                cls = det["class"]
                if now - cooldowns.get(cls, 0) < config["cooldown_seconds"]:
                    continue
                cooldowns[cls] = now
                snap_name = None
                if config.get("save_snapshots"):
                    snap_name = save_snapshot(frame, det, Path("snapshots"))
                log_event(det, lat, lon, snap_name, Path("events.jsonl"))
            cv2.imshow("Detection Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()