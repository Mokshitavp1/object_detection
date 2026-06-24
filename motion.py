"""
Motion detection via frame differencing.

Usage:
    detector = MotionDetector(pixel_threshold=5000)
    if detector.has_motion(frame):
        # run inference
"""

import cv2


class MotionDetector:
    """Detect motion between consecutive frames using absolute difference."""

    def __init__(self, pixel_threshold: int = 5000) -> None:
        self.pixel_threshold = pixel_threshold
        self.prev_gray = None

    def has_motion(self, frame) -> bool:
        """Return True if enough pixels changed since the last call."""
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = blurred
            return False

        diff       = cv2.absdiff(self.prev_gray, blurred)
        thresh     = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        non_zero   = cv2.countNonZero(thresh)
        self.prev_gray = blurred

        return non_zero > self.pixel_threshold
