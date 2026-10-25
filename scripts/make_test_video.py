"""Generate deterministic synthetic test videos for E2E tests.

Each test video is built from one or more *real face photos* (stored under
``tests/assets/face/``) so that MediaPipe FaceMesh can detect them — the
cartoon faces used in early prototypes were too abstract for FaceMesh to
lock onto, which made the tests non-deterministic.

The same input file always produces the same output bytes, so E2E tests
can assert specific alarm counts / latencies after running the detector
over the produced MP4.

Patterns:
    alert   — a single alert face for the whole clip (eyes open, mouth closed)
    drowsy  — alert face for the first half, then a sleepy face for the rest
    yawning — alert face with a yawn face overlaid every 1 s for 0.7 s
    mixed   — alert for 2 s, sleepy for 2 s, yawning for 2 s
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Face image selection
# --------------------------------------------------------------------------- #
FACE_ASSET_DIR = Path(__file__).resolve().parent.parent / "tests" / "assets" / "face"
ALERT_FACE = "face2.jpg"           # EAR ≈ 0.45, MAR ≈ 0.06 — alert baseline
SLEEPY_FACE = "sleepy2.jpg"       # EAR ≈ 0.25, MAR ≈ 0.03 — eyes drooping
YAWN_FACE = "yawn_3bf78d5140f2.jpg"  # EAR ≈ 0.04, MAR ≈ 0.97 — full yawn


def _load_face(filename: str, target_w: int, target_h: int) -> np.ndarray:
    """Load a face image and letterbox it to (target_w, target_h).

    Letterboxing preserves aspect ratio and avoids the warped faces that
    MediaPipe sometimes fails on when stretched.
    """
    path = FACE_ASSET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Test asset missing: {path}. Run scripts/download_face_assets.py."
        )
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read {path}")
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pad to target size with a neutral background.
    out = np.full((target_h, target_w, 3), 200, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def make_test_video(
    path: str | os.PathLike,
    *,
    fps: int = 15,
    width: int = 640,
    height: int = 480,
    duration_s: float = 6.0,
    pattern: Literal["alert", "drowsy", "yawning", "mixed"] = "mixed",
) -> Path:
    """Write a deterministic test MP4 to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    # Pre-render all needed frames so the video writer just dumps them.
    alert_img = _load_face(ALERT_FACE, width, height)
    sleepy_img = _load_face(SLEEPY_FACE, width, height)
    yawn_img = _load_face(YAWN_FACE, width, height)

    total_frames = int(fps * duration_s)
    for i in range(total_frames):
        t = i / fps
        if pattern == "alert":
            frame = alert_img
        elif pattern == "drowsy":
            # Last 3 s: switch to sleepy face.
            frame = sleepy_img if t > duration_s - 3.0 else alert_img
        elif pattern == "yawning":
            # Open mouth for ~0.7 s every 1 s.
            phase = t % 1.0
            frame = yawn_img if phase < 0.7 else alert_img
        elif pattern == "mixed":
            if t < 2.0:
                frame = alert_img
            elif t < 4.0:
                frame = sleepy_img
            else:
                phase = (t - 4.0) % 1.0
                frame = yawn_img if phase < 0.7 else alert_img
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown pattern {pattern!r}")
        writer.write(frame)
    writer.release()
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="tests/assets/test_mixed.mp4")
    p.add_argument("--pattern", default="mixed",
                   choices=["alert", "drowsy", "yawning", "mixed"])
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args(argv)
    path = make_test_video(args.out, pattern=args.pattern,
                           duration_s=args.duration, fps=args.fps)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
