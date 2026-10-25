"""Non-functional system tests: FPS, latency, memory, determinism.

These tests do NOT verify correctness (the unit/E2E tests do that) — they
verify the system meets its non-functional requirements:

* FPS ≥ 8 on a CPU-only machine (i5-class).
* Per-frame latency < 200 ms (geometric detector only).
* No memory leak over 100 frames (RSS stays within +50 MB).
* Deterministic output: the same video yields the same alarm state across runs.
"""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest

from drowsiness.config import load_config
from drowsiness.detectors.geometric import GeometricDetector
from drowsiness.state import DrowsinessTracker
from drowsiness.video.source import VideoSource
from drowsiness.config import VideoConfig

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _rss_mb() -> float:
    """Return current process RSS in MB (Linux-only, falls back to 0)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def test_fps_meets_minimum_on_cpu():
    """Pipeline should run at ≥ 8 FPS on CPU (target is 15)."""
    cfg = load_config(overrides={"show_window": False,
                                 "cnn": {"model_path": ""}})
    src = VideoSource(VideoConfig(source=str(ASSETS / "test_alert.mp4"),
                                   width=cfg.video.width,
                                   height=cfg.video.height,
                                   target_fps=cfg.video.target_fps))
    det = GeometricDetector(cfg)
    tr = DrowsinessTracker(cfg)
    src.open()
    times = []
    try:
        idx = 0
        while True:
            frame = src.read()
            if frame is None:
                break
            t0 = time.perf_counter()
            d = det.detect(frame.image)
            d.frame_index = idx
            idx += 1
            tr.update(d)
            times.append(time.perf_counter() - t0)
    finally:
        src.close()
        det.close()
    assert len(times) > 0
    fps = 1.0 / statistics.mean(times)
    assert fps >= 8.0, f"FPS too low: {fps:.2f}"


def test_per_frame_latency_under_200ms():
    """Each frame's processing latency (geometric detector) < 200 ms."""
    cfg = load_config(overrides={"show_window": False,
                                 "cnn": {"model_path": ""}})
    src = VideoSource(VideoConfig(source=str(ASSETS / "test_alert.mp4"),
                                   width=cfg.video.width,
                                   height=cfg.video.height,
                                   target_fps=cfg.video.target_fps))
    det = GeometricDetector(cfg)
    tr = DrowsinessTracker(cfg)
    src.open()
    latencies_ms = []
    try:
        idx = 0
        while True:
            frame = src.read()
            if frame is None:
                break
            d = det.detect(frame.image)
            d.frame_index = idx
            idx += 1
            tr.update(d)
            if d.latency_ms is not None:
                latencies_ms.append(d.latency_ms)
    finally:
        src.close()
        det.close()
    assert latencies_ms
    p95 = statistics.quantiles(latencies_ms, n=20)[18]  # 95th percentile
    assert p95 < 200.0, f"p95 latency too high: {p95:.1f} ms"


def test_no_memory_leak_over_100_frames():
    """RSS should not grow more than 50 MB over 100 frames."""
    cfg = load_config(overrides={"show_window": False,
                                 "cnn": {"model_path": ""}})
    src = VideoSource(VideoConfig(source=str(ASSETS / "test_alert.mp4"),
                                   width=cfg.video.width,
                                   height=cfg.video.height,
                                   target_fps=cfg.video.target_fps))
    det = GeometricDetector(cfg)
    tr = DrowsinessTracker(cfg)
    src.open()
    rss_before = _rss_mb()
    try:
        idx = 0
        # Loop the video if it's shorter than 100 frames.
        while idx < 100:
            frame = src.read()
            if frame is None:
                src.close()
                src = VideoSource(VideoConfig(source=str(ASSETS / "test_alert.mp4"),
                                              width=cfg.video.width,
                                              height=cfg.video.height,
                                              target_fps=cfg.video.target_fps))
                src.open()
                continue
            d = det.detect(frame.image)
            d.frame_index = idx
            idx += 1
            tr.update(d)
    finally:
        src.close()
        det.close()
    rss_after = _rss_mb()
    delta = rss_after - rss_before
    assert delta < 50.0, f"Memory grew by {delta:.1f} MB (possible leak)"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_pipeline_is_deterministic_across_runs():
    """Running the same video twice must produce identical alarm state."""
    results = []
    for _ in range(2):
        cfg = load_config(overrides={"show_window": False,
                                     "cnn": {"model_path": ""}})
        src = VideoSource(VideoConfig(source=str(ASSETS / "test_mixed.mp4"),
                                       width=cfg.video.width,
                                       height=cfg.video.height,
                                       target_fps=cfg.video.target_fps))
        det = GeometricDetector(cfg)
        tr = DrowsinessTracker(cfg)
        src.open()
        try:
            idx = 0
            while True:
                frame = src.read()
                if frame is None:
                    break
                d = det.detect(frame.image)
                d.frame_index = idx
                idx += 1
                tr.update(d)
        finally:
            src.close()
            det.close()
        st = tr.state
        results.append((
            st.frames_processed,
            round(st.perclos, 4),
            st.yawn_count_window,
            st.eye_alarm, st.yawn_alarm, st.perclos_alarm,
            round(st.latest_ear or 0, 4),
            round(st.latest_mar or 0, 4),
        ))
    assert results[0] == results[1], (
        f"Non-deterministic! Run 1: {results[0]}, Run 2: {results[1]}"
    )
