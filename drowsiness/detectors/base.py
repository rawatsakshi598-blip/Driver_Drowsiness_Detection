"""Detector protocol & shared dataclasses.

Every detector (geometric, CNN, hybrid) returns a ``Detection`` describing
what it saw in the current frame.  The orchestrator combines detections
across frames to decide whether to raise an alarm.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FaceLandmarks:
    """The 468 MediaPipe FaceMesh landmarks for a single face.

    ``points`` is shape (N, 3) — (x, y, z) in normalized image coords.
    ``bbox`` is the (x, y, w, h) of the face bounding box in *pixel* coords.
    """
    points: np.ndarray  # shape (N, 3), float32
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels

    def eye_landmarks(self, indices: List[int]) -> np.ndarray:
        """Return the (N, 2) pixel-coord landmarks for the given indices."""
        return self.points[indices, :2]


@dataclass
class Detection:
    """A single-frame detection result.

    All numeric fields are ``None`` when the corresponding signal was not
    computed (e.g. no face found, or the CNN verifier was not run this
    frame).  This explicit None-sentinel avoids the "0.0 vs missing"
    ambiguity that plagues real-time systems.
    """
    # Wall-clock time the detection was produced.
    timestamp: float = field(default_factory=time.time)
    # Whether a face was found at all in the frame.
    face_found: bool = False
    # Raw Eye Aspect Ratio (averaged over both eyes) — float in [0, ~0.4].
    ear: Optional[float] = None
    # Whether the eyes are *currently* closed (boolean).
    eyes_closed: bool = False
    # Raw Mouth Aspect Ratio.
    mar: Optional[float] = None
    # Whether the mouth is currently open enough to count as a yawn.
    yawning: bool = False
    # CNN verifier outputs (only set when CNN was actually run).
    cnn_class: Optional[str] = None
    cnn_confidence: Optional[float] = None
    # Bounding box of the primary face in pixel coords (x, y, w, h).
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    # Processing latency of this frame, in milliseconds.
    latency_ms: Optional[float] = None
    # Frame index (monotonic counter maintained by the orchestrator).
    frame_index: int = 0


class Detector(abc.ABC):
    """Abstract base class for detectors."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> Detection:
        """Process a BGR frame and return a ``Detection``."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources (MediaPipe graph, TF session, etc.)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
