import cv2
import numpy as np


class DetectionZone:
    """Filter detections to those whose bounding-box centre lies inside a polygon."""

    def __init__(self, polygon_points: list) -> None:
        if not polygon_points:
            self.polygon = None
        else:
            self.polygon = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))

    def is_inside(self, bbox: list) -> bool:
        """Return True if the bbox centre is inside the zone (or zone is disabled)."""
        if self.polygon is None:
            return True
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        result = cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), measureDist=False)
        return result >= 0
