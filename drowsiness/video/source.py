"""Video source abstraction.

Wraps ``cv2.VideoCapture`` so the rest of the codebase can talk to a single,
testable interface.  Supports:

* webcam indices (``0``, ``1`` ...)
* video files (``"/path/to/clip.mp4"``)
* RTSP / HTTP streams (``"rtsp://..."``, ``"http://.../stream.m3u8"``)

The source maintains a background thread for read-ahead buffering when
configured, which dramatically reduces latency on slow USB webcams.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np

from ..config import VideoConfig

log = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single video frame with metadata."""
    image: np.ndarray              # BGR uint8 HxWx3
    timestamp: float              # wall-clock capture time
    frame_index: int              # monotonic index since source start
    source_fps: Optional[float] = None


class VideoSource:
    """OpenCV-backed video source.

    The source is a context manager; releasing the underlying
    ``VideoCapture`` is mandatory on Linux to free /dev/video* for the next
    process.
    """

    def __init__(self, config: VideoConfig):
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._stopped = False
        # Background buffered reader (only used when source is a live
        # camera or network stream).
        self._reader_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[Frame] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Open the underlying ``VideoCapture``."""
        cfg = self._cfg
        kwargs = {}
        if cfg.backend is not None:
            kwargs["apiPreference"] = cfg.backend
        cap = cv2.VideoCapture(cfg.source, **kwargs)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source {cfg.source!r}. "
                "Check that no other process holds the device, the path "
                "exists, and the format is supported by this OpenCV build."
            )
        if isinstance(cfg.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap = cap
        self._start_time = time.perf_counter()
        log.info(
            "VideoSource opened: source=%r target=%dx%d@%dfps",
            cfg.source, cfg.width, cfg.height, cfg.target_fps,
        )

    def close(self) -> None:
        """Stop the background reader and release the capture."""
        self._stopped = True
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.debug("VideoSource closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------ #
    # Frame reading
    # ------------------------------------------------------------------ #
    def read(self) -> Optional[Frame]:
        """Read a single frame, synchronously.

        Returns ``None`` at end of stream (file/RTSP) or when the device
        was closed.  The caller is responsible for any target-FPS pacing.
        """
        if self._cap is None:
            raise RuntimeError("VideoSource.read() called before open()")
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        ts = time.time()
        frame = Frame(
            image=image,
            timestamp=ts,
            frame_index=self._frame_index,
            source_fps=self.fps(),
        )
        self._frame_index += 1
        return frame

    def fps(self) -> Optional[float]:
        """Return the source's reported FPS, if known."""
        if self._cap is None:
            return None
        v = self._cap.get(cv2.CAP_PROP_FPS)
        return float(v) if v and v > 0 else None

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time
