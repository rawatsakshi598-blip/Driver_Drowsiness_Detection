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
        # API selector: 'legacy' (mediapipe 0.10.x, has mp.solutions.face_mesh)
        # or 'tasks' (mediapipe 1.x, uses mp.tasks.python.vision.FaceLandmarker).
        self._api: str = ""
        # Underlying handle — FaceMesh (legacy) or FaceLandmarker (tasks).
        self._face_mesh = None
        self._init_face_mesh()
        # EMA-smoothed signals (one value per stream, updated each frame).
        self._ear_ema: float | None = None
        self._mar_ema: float | None = None
        log.debug("GeometricDetector initialised (api=%s)", self._api)

    # ------------------------------------------------------------------ #
    # MediaPipe API detection / initialisation
    # ------------------------------------------------------------------ #
    def _init_face_mesh(self) -> None:
        """Detect which MediaPipe API is available and initialise the
        appropriate FaceLandmarker / FaceMesh instance.

        MediaPipe shipped two APIs over time:

        1. Legacy API (mediapipe 0.10.x):
              mp.solutions.face_mesh.FaceMesh(...)
              results.multi_face_landmarks -> list of NormalizedLandmarkList
        2. Tasks API (mediapipe 1.x):
              mp.tasks.python.vision.FaceLandmarker
              requires a downloaded .task model file

        We try the legacy API first (it's bundled with the wheel), and
        fall back to the Tasks API only if the legacy one is missing.
        """
        mp = self._mp
        # ---- Try legacy API ----
        legacy_face_mesh_cls = None
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                legacy_face_mesh_cls = mp.solutions.face_mesh.FaceMesh
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Legacy mp.solutions.face_mesh unavailable: %s", exc)

        if legacy_face_mesh_cls is not None:
            self._face_mesh = legacy_face_mesh_cls(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
            )
            self._api = "legacy"
            return

        # ---- Try Tasks API ----
        try:
            from mediapipe.tasks.python import vision  # noqa: WPS433
            from mediapipe.tasks import python as mp_python  # noqa: WPS433
            self._tasks_vision = vision
            self._mp_python = mp_python
            # Need to download the FaceLandmarker .task model.
            model_path = self._ensure_face_landmarker_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_mesh = vision.FaceLandmarker.create_from_options(options)
            self._api = "tasks"
            self._tasks_frame_index = 0  # monotonic counter for timestamps
            return
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialise MediaPipe FaceLandmarker with either "
                f"the legacy 'mp.solutions.face_mesh' API or the new "
                f"'mp.tasks.python.vision.FaceLandmarker' API.\n\n"
                f"Detected mediapipe version: "
                f"{getattr(mp, '__version__', 'unknown')}\n\n"
                f"Original error: {exc}\n\n"
                f"FIX: install the legacy mediapipe version that bundles "
                f"the FaceMesh solution:\n\n"
                f"    pip install --force-reinstall --no-deps "
                f"mediapipe==0.10.14 'numpy<2.2' 'protobuf<5'\n\n"
                f"Then re-run pytest."
            ) from exc

    @staticmethod
    def _ensure_face_landmarker_model() -> "Path":
        """Download the FaceLandmarker .task model if not already cached.

        Only needed when using the Tasks API (mediapipe 1.x). The legacy
        API (mediapipe 0.10.x) bundles the model in the wheel.
        """
        from pathlib import Path
        import urllib.request
        cache_dir = Path.home() / ".cache" / "drowsiness"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "face_landmarker.task"
        if model_path.exists() and model_path.stat().st_size > 1_000_000:
            return model_path
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/"
               "face_landmarker.task")
        log.info("Downloading MediaPipe FaceLandmarker model from %s", url)
        urllib.request.urlretrieve(url, model_path)
        if not model_path.exists() or model_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Failed to download FaceLandmarker model to {model_path}"
            )
        return model_path

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> Detection:
        """Run FaceMesh on ``frame`` (BGR uint8 HxWx3) and return a Detection."""
        import time
        t0 = time.perf_counter()
        rgb = self._bgr_to_rgb(frame)
        h, w = frame.shape[:2]
        if self._api == "legacy":
            results = self._face_mesh.process(rgb)
            face_landmarks_list = (
                results.multi_face_landmarks if results.multi_face_landmarks else []
            )
        else:  # tasks API
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(self._tasks_frame_index * 33)  # ~30ms per frame
            self._tasks_frame_index += 1
            results = self._face_mesh.detect_for_video(mp_image, ts_ms)
            face_landmarks_list = results.face_landmarks
        det = Detection()
        if not face_landmarks_list:
            det.latency_ms = (time.perf_counter() - t0) * 1000.0
            return det
        # Normalize the landmark access:
        # - Legacy API: face_landmarks_list[0] is a NormalizedLandmarkList
        #   with a .landmark attribute (a list of NormalizedLandmark).
        # - Tasks API: face_landmarks_list[0] is directly a list of
        #   NormalizedLandmark objects.
        first_face = face_landmarks_list[0]
        if self._api == "legacy":
            lm_list = first_face.landmark
        else:
            lm_list = first_face  # already a list of NormalizedLandmark
        pts = np.array(
            [(lm.x * w, lm.y * h, lm.z) for lm in lm_list],
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
        # Both APIs expose .close() but the Tasks API also requires
        # explicit release to free the underlying TFLite interpreter.
        if getattr(self, "_face_mesh", None) is not None:
            try:
                self._face_mesh.close()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("Error closing face_mesh: %s", exc)
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
