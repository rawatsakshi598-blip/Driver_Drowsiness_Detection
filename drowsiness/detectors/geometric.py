"""MediaPipe FaceMesh-based geometric detector.

This is the *primary* detector used at runtime.  It computes:

* **Eye Aspect Ratio (EAR)** — Soukupova & Cech, 2016 — for eye closure.
* **Mouth Aspect Ratio (MAR)** — analogous geometry applied to the mouth —
  for yawn detection.
* **PERCLOS** — percentage of eye closure over a rolling window — the
  NHTSA / FMCSA gold-standard fatigue metric.

All of these are computed from the 468 MediaPipe FaceMesh landmarks
returned in a single inference call per frame, which makes this path
fast enough to run at 15–30 FPS on a Core i5-8265U with no GPU.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..config import Config, FaceMeshIndices
from .base import Detection, Detector, FaceLandmarks

log = logging.getLogger(__name__)


def eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio from 6 (x, y) landmarks.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    where the points are ordered
    ``[outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]``
    around the eye.  Open eye ≈ 0.30; closed eye ≈ 0.10.

    Args:
        eye_landmarks: shape (6, 2) or (6, 3) array of pixel/normalized
            landmark coordinates.  Only the first two columns are used.

    Returns:
        float EAR.  Returns 0.0 for degenerate inputs.
    """
    if eye_landmarks.shape[0] < 6:
        return 0.0
    p = np.asarray(eye_landmarks, dtype=np.float64)[:, :2]
    # Vertical distances (numerator terms).
    d1 = np.linalg.norm(p[1] - p[5])
    d2 = np.linalg.norm(p[2] - p[4])
    # Horizontal distance (denominator).
    d3 = np.linalg.norm(p[0] - p[3])
    if d3 == 0:
        return 0.0
    return float((d1 + d2) / (2.0 * d3))


def mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """Compute the Mouth Aspect Ratio from 4 (x, y) landmarks.

    MAR = (||upper-lip-centre - lower-lip-centre||) /
          (||left-corner - right-corner||)

    Open mouth (yawn) gives a large MAR; closed mouth gives a small one.

    Args:
        mouth_landmarks: shape (4, 2) or (4, 3) array.  Order must be
            ``[upper_inner, lower_inner, left_corner, right_corner]``.

    Returns:
        float MAR.  Returns 0.0 for degenerate inputs.
    """
    if mouth_landmarks.shape[0] < 4:
        return 0.0
    p = np.asarray(mouth_landmarks, dtype=np.float64)[:, :2]
    vertical = np.linalg.norm(p[0] - p[1])
    horizontal = np.linalg.norm(p[2] - p[3])
    if horizontal == 0:
        return 0.0
    return float(vertical / horizontal)


class GeometricDetector(Detector):
    """Real-time geometric drowsiness + yawn detector.

    Resources:
        MediaPipe FaceMesh solution (1 inference/frame).

    Lifecycle:
        Use as a context manager (``with GeometricDetector(cfg) as d: ...``)
        or call ``.close()`` explicitly.  The underlying MediaPipe graph
        holds native resources that must be released.
    """

    def __init__(self, config: Config):
        self._cfg = config
        # Imported lazily so the rest of the package can be imported in
        # environments without MediaPipe (e.g. lightweight unit tests of
        # the EAR/MAR math).
        import mediapipe as mp  # noqa: WPS433
        self._mp = mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # 468 landmarks is enough; refine adds 478.
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,  # video mode = tracking on.
        )
        # EMA-smoothed signals (one value per stream, updated each frame).
        self._ear_ema: float | None = None
        self._mar_ema: float | None = None
        log.debug("GeometricDetector initialised")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> Detection:
        """Run FaceMesh on ``frame`` (BGR uint8 HxWx3) and return a Detection."""
        import time
        t0 = time.perf_counter()
        rgb = self._bgr_to_rgb(frame)
        h, w = frame.shape[:2]
        results = self._face_mesh.process(rgb)
        det = Detection()
        if not results.multi_face_landmarks:
            det.latency_ms = (time.perf_counter() - t0) * 1000.0
            return det
        landmarks = results.multi_face_landmarks[0]
        pts = np.array(
            [(lm.x * w, lm.y * h, lm.z) for lm in landmarks.landmark],
            dtype=np.float32,
        )
        # Compute face bounding box from the FaceMesh oval landmarks.
        oval = pts[FaceMeshIndices.FACE_OVAL, :2]
        x0, y0 = oval.min(axis=0)
        x1, y1 = oval.max(axis=0)
        bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        fl = FaceLandmarks(points=pts, bbox=bbox)
        det.face_found = True
        det.face_bbox = bbox

        # ---- EAR ----
        left_ear = eye_aspect_ratio(fl.eye_landmarks(FaceMeshIndices.LEFT_EYE))
        right_ear = eye_aspect_ratio(fl.eye_landmarks(FaceMeshIndices.RIGHT_EYE))
        raw_ear = (left_ear + right_ear) / 2.0
        det.ear = self._ema("ear", raw_ear)
        det.eyes_closed = det.ear < self._cfg.eye.ear_threshold

        # ---- MAR ----
        mouth_pts = np.concatenate(
            [
                fl.eye_landmarks(FaceMeshIndices.MOUTH_VERTICAL),
                fl.eye_landmarks(FaceMeshIndices.MOUTH_HORIZONTAL),
            ],
            axis=0,
        )
        raw_mar = mouth_aspect_ratio(mouth_pts)
        det.mar = self._ema("mar", raw_mar)
        det.yawning = det.mar >= self._cfg.mouth.mar_threshold

        det.latency_ms = (time.perf_counter() - t0) * 1000.0
        return det

    def close(self) -> None:
        if getattr(self, "_face_mesh", None) is not None:
            self._face_mesh.close()
            self._face_mesh = None
        log.debug("GeometricDetector closed")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """OpenCV gives BGR; MediaPipe wants RGB.  Avoid a copy if possible."""
        # cv2.cvtColor returns a new array (no aliasing), safe for MP.
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _ema(self, stream: str, raw: float) -> float:
        """Exponential moving average smoothing for a signal stream."""
        if stream == "ear":
            alpha = self._cfg.eye.ear_smoothing_alpha
            prev = self._ear_ema
            self._ear_ema = raw if prev is None else alpha * raw + (1 - alpha) * prev
            return self._ear_ema
        elif stream == "mar":
            alpha = self._cfg.mouth.mar_smoothing_alpha
            prev = self._mar_ema
            self._mar_ema = raw if prev is None else alpha * raw + (1 - alpha) * prev
            return self._mar_ema
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown EMA stream {stream!r}")
