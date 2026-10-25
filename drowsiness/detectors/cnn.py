"""Optional CNN verifier — the original Keras model is loaded lazily
and used only as a tie-breaker for ambiguous cases.

This class is intentionally lazy: importing TensorFlow at module load time
adds ~1.5 s to startup on a cold machine, which is unacceptable for a CLI
that may just want to dump its config.  We therefore defer all TensorFlow
imports until ``__init__`` is called, and the orchestrator only calls
``__init__`` when the user opts in.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..config import Config
from .base import Detection, Detector, FaceLandmarks

log = logging.getLogger(__name__)

# Silence TF's chatty INFO logs at import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class CNNVerifier(Detector):
    """Loads the legacy Keras model and runs it on cropped face ROIs.

    The CNN produces a 4-class softmax: ``(yawn, no_yawn, Closed, Open)``.
    We treat indices 0 and 1 as *yawn vs no_yawn* and indices 2 and 3 as
    *Closed vs Open* eyes.  Because the model was jointly trained on
    both eye crops and full-face (mouth) crops, the orchestrator decides
    *which* crop to feed based on what it is asking about.
    """

    def __init__(self, config: Config):
        self._cfg = config
        if not config.cnn.model_path:
            raise ValueError(
                "CNNVerifier enabled but cnn.model_path is empty."
            )
        # Defer the import: TensorFlow is heavy.
        from tensorflow.keras.models import load_model  # noqa: WPS433
        log.info("Loading CNN model from %s ...", config.cnn.model_path)
        self._model = load_model(config.cnn.model_path, compile=False)
        self._input_size = config.cnn.input_size
        log.info(
            "CNN model loaded (params=%d, input=%dx%d).",
            self._model.count_params(),
            self._input_size, self._input_size,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> Detection:
        """Run the CNN on the largest detected face region.

        Note: this detector does *not* run face detection itself; the
        orchestrator passes it a pre-cropped face ROI.  We expose
        ``classify_eye`` and ``classify_face`` separately for clarity.
        """
        # If called directly, classify the whole frame.
        return self.classify_face(frame, bbox=None)

    def classify_eye(self, eye_crop_bgr: np.ndarray) -> tuple[str, float]:
        """Classify an eye crop as 'Open' or 'Closed'."""
        return self._predict(eye_crop_bgr, expected_subset=("Closed", "Open"))

    def classify_face(self, face_crop_bgr: np.ndarray,
                      bbox: Optional[tuple[int, int, int, int]] = None) -> Detection:
        """Classify a face crop.  Returns a Detection with the CNN fields set.

        ``bbox`` is the (x, y, w, h) of the face in the *original* frame, for
        reporting purposes only.
        """
        import time
        t0 = time.perf_counter()
        det = Detection(face_found=face_crop_bgr.size > 0, face_bbox=bbox)
        # The model was trained on a 4-class softmax.  We pick the
        # argmax and use its confidence.
        try:
            x = self._preprocess(face_crop_bgr)
            probs = self._model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            det.cnn_class = self._cfg.cnn.class_names[idx]
            det.cnn_confidence = float(probs[idx])
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("CNN predict failed: %s", exc)
            det.cnn_class = None
            det.cnn_confidence = None
        det.latency_ms = (time.perf_counter() - t0) * 1000.0
        return det

    def close(self) -> None:
        # Keras doesn't expose an explicit session close in TF 2.x, but we
        # release our reference to allow GC of the underlying graph.
        self._model = None
        log.debug("CNNVerifier closed")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Resize to model input size and normalise to [0, 1]."""
        import cv2
        from tensorflow.keras.preprocessing.image import img_to_array
        resized = cv2.resize(bgr_crop, (self._input_size, self._input_size))
        arr = resized.astype("float32") / 255.0
        arr = img_to_array(arr)
        return np.expand_dims(arr, axis=0)

    def _predict(self, bgr_crop: np.ndarray,
                expected_subset: tuple[str, ...]) -> tuple[str, float]:
        """Run the model and map the 4-class output onto the requested subset.

        Returns the most-likely class *within the subset* and its probability.
        For example, ``expected_subset=("Closed", "Open")`` returns the more
        probable of "Closed" vs "Open", summing their probabilities first.
        """
        x = self._preprocess(bgr_crop)
        probs = self._model.predict(x, verbose=0)[0]
        names = self._cfg.cnn.class_names
        # Sum the probabilities of the subset members and re-normalise.
        subset_idx = [names.index(c) for c in expected_subset]
        subset_probs = probs[subset_idx]
        winner = int(np.argmax(subset_probs))
        total = float(subset_probs.sum())
        return expected_subset[winner], (subset_probs[winner] / total if total > 0 else 0.0)
