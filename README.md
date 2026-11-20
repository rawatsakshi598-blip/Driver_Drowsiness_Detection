# Driver Drowsiness Detection — Production Grade

A real-time driver drowsiness detection system that monitors **eye closure
(EAR + PERCLOS)** and **yawning (MAR + yawn rate)** through a webcam and
raises an audio alarm when the driver shows fatigue signs.  Designed to
run on a low-end CPU (Intel Core i5-8265U, 4 GB RAM) at **15 FPS** with
sub-200 ms latency.

The system is **production-grade**: typed, modular, configurable, tested
(unit + E2E + system), and verified end-to-end with `curl`.

---

## Why this rewrite exists

The original `detect_drowsiness.py` shipped with three blocking issues:

1. **Yawning was never detected at runtime**, even though the underlying
   CNN had yawn/no_yawn classes. The live script fed **eye crops** to the
   model and then asked "is this a yawn?" — the model literally had no
   mouth-region input to look at.  See `drowsiness/detectors/geometric.py`
   for the correct approach using MediaPipe FaceMesh + Mouth Aspect Ratio.

2. **Latency was too high** for real-time use.  Two `model.predict()`
   calls per frame on a 495 K-param CNN, plus Haar cascades on every
   frame, ran at <5 FPS on a Core i5.  The new pipeline uses MediaPipe
   FaceMesh (one inference per frame) and computes EAR / MAR
   **geometrically** — no CNN is in the hot path.

3. **No tests, no config, no error handling, hardcoded paths.**  Every
   threshold now lives in `config.yaml`; every module has unit tests;
   the whole pipeline has E2E + system tests.

---

## Architecture

```
VideoSource (cv2.VideoCapture)
    │
    ▼
GeometricDetector (MediaPipe FaceMesh → 468 landmarks)
    │   ├── Eye Aspect Ratio (EAR)  ← eye-closure signal
    │   ├── Mouth Aspect Ratio (MAR) ← yawn signal
    │   └── (optional) CNNVerifier  ← tie-breaker on ambiguous frames
    ▼
DrowsinessTracker (state machine)
    │   ├── Consecutive closed-eye frames
    │   ├── PERCLOS (rolling 60 s window, NHTSA standard)
    │   ├── Yawn count + yawn rate (rolling 60 s)
    │   └── Alarm thresholds
    ▼
SoundAlarm (cooldown, daemon-thread audio)
    │
    ▼
Flask server (status & control: /health, /status, /config,
              /start, /stop, /alarm/test, /alarm/reset)
```

---

## Quick start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the detector against your webcam
python main.py monitor --source 0

# 3. Or run it against a video file (headless)
python main.py detect --source path/to/clip.mp4

# 4. Or run the Flask status & control server
python main.py server --no-window

# 5. Verify the server with curl
./scripts/verify_with_curl.sh
```

---

## Configuration

Every tunable parameter lives in [`config.yaml`](./config.yaml):

| Section    | Key                       | Default | Meaning                                  |
|------------|---------------------------|---------|------------------------------------------|
| `eye`      | `ear_threshold`           | 0.20    | EAR below this → eye closed              |
| `eye`      | `closed_frame_threshold`  | 20      | Consecutive closed frames → eye alarm    |
| `eye`      | `perclos_window_seconds`  | 60.0    | PERCLOS rolling window (NHTSA standard) |
| `eye`      | `perclos_threshold`       | 0.15    | PERCLOS fraction → fatigue alarm        |
| `mouth`    | `mar_threshold`           | 0.55    | MAR above this → mouth open (yawn)       |
| `mouth`    | `yawn_frame_threshold`    | 10      | Consecutive open-mouth frames → yawn     |
| `mouth`    | `yawn_rate_per_minute`    | 3       | Yawns/min above this → yawn alarm         |
| `mouth`    | `max_yawn_seconds`        | 4.0     | Single yawn held longer → PROLONGED_YAWN |
| `lighting` | `enabled`                 | `true`  | Check ambient lighting on every frame      |
| `lighting` | `min_brightness`          | 60.0    | Below this mean gray value → too dark     |
| `lighting` | `max_brightness`          | 240.0   | Above this → overexposed                  |
| `lighting` | `consecutive_frames`      | 8       | Frames before LOW_LIGHT / BRIGHT_LIGHT    |
| `focus`    | `enabled`                 | `true`  | Blur check (variance of Laplacian)        |
| `focus`    | `min_sharpness`           | 60.0    | Below this → too blurry to trust          |
| `focus`    | `consecutive_frames`      | 8       | Frames before OUT_OF_FOCUS fires          |
| `distraction` | `enabled`              | `true`  | Driver-attention checks                    |
| `distraction` | `no_face_seconds`      | 2.0     | Face unseen this long → FACE_NOT_VISIBLE  |
| `distraction` | `look_away_straightness` | 0.55 | Head-yaw straightness below → looking away |
| `distraction` | `look_away_frames`     | 10      | Frames before LOOKING_AWAY fires          |
| `distraction` | `alarm_with_sound`     | `true`  | Sound alarm for critical alerts           |
| `video`    | `target_fps`              | 15      | Loop pacing (15 FPS on i5-8265U)         |
| `alarm`    | `cooldown_seconds`        | 4.0     | Cooldown between alarm triggers           |
| `cnn`      | `model_path`              | `drowiness_new7.h5` | Legacy Keras verifier (optional) |
| `cnn`      | `run_every_n_frames`      | 5       | CNN is run only every Nth frame           |

---

## HTTP API (for `curl` verification)

The Flask server exposes the following endpoints.  See
[`scripts/verify_with_curl.sh`](./scripts/verify_with_curl.sh) for the
full verification script.

| Endpoint                | Method | Description                                  |
|-------------------------|--------|----------------------------------------------|
| `/health`               | GET    | Liveness probe (`{"status":"ok"}`)          |
| `/status`               | GET    | Live detection state (FPS, EAR, MAR, alarms, alerts) |
| `/alerts`               | GET    | Only the currently-active real-time alerts   |
| `/config`               | GET    | Dump the current configuration as JSON       |
| `/start`                | POST   | Start the monitor                            |
| `/stop`                 | POST   | Stop the monitor                            |
| `/alarm/test`           | POST   | Force-trigger the alarm                      |
| `/alarm/reset`          | POST   | Reset the alarm state                        |

Example:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status | jq
curl http://127.0.0.1:8000/alerts | jq
curl -X POST http://127.0.0.1:8000/alarm/test
```

---

## Real-time alerts (new)

Besides the three drowsiness alarms (`eye_closure`, `yawn_burst`,
`perclos`), the pipeline now emits **edge-triggered alerts** the moment a
problem appears — logged immediately to the console, drawn on the HUD,
and exposed via `/status` and `/alerts`:

| Code                | Severity   | Fires when                                         |
|---------------------|------------|----------------------------------------------------|
| `FACE_NOT_VISIBLE`  | `critical` | Face undetected for more than `distraction.no_face_seconds` |
| `LOOKING_AWAY`      | `warning`  | Driver's head turned away from the road (low yaw straightness) |
| `PROLONGED_YAWN`    | `warning`  | One yawn held longer than `mouth.max_yawn_seconds` |
| `LOW_LIGHT`         | `warning`  | Mean frame brightness below `lighting.min_brightness` |
| `BRIGHT_LIGHT`      | `warning`  | Mean frame brightness above `lighting.max_brightness` |
| `OUT_OF_FOCUS`      | `warning`  | Frame blurrier than `focus.min_sharpness` (Laplacian variance) |

Detection details:

* **Lighting / sharpness** are computed from the raw grayscale frame on
  *every* frame — they work even with no face visible.
* **Head-yaw straightness** is `min/max` of the nose-tip → cheek-tragion
  (landmarks 234/454) distances; `1.0` = facing the road, trending toward
  `0` as the head turns.
* **Edge-triggered logging**: each alert logs once when it appears
  (`ALERT [CODE]: ...`) and once when it clears (`Alert cleared: ...`),
  never per-frame spam.
* `FACE_NOT_VISIBLE` and `PROLONGED_YAWN` also sound the alarm when
  `distraction.alarm_with_sound` is `true`; the rest are visual + log +
  status only.
* Drowsiness semantics are unchanged: `any_alarm` still means
  eye/yawn-rate/PERCLOS only; `any_alert` covers the new codes;
  `any_issue` = either.

---

## Tests

The test suite covers **unit**, **end-to-end (E2E)**, and **system
(non-functional)** tests.  **No mocks** — every test exercises the real
code on real face imagery or real geometric inputs.

```bash
# Run everything (75 tests, ~20 s)
python -m pytest tests/ -v

# Or just one layer
python -m pytest tests/unit/      # EAR, MAR, config, state machine, alarm
python -m pytest tests/e2e/       # CNN integration, Flask server, video pipeline
python -m pytest tests/system/    # FPS, latency, memory, determinism
```

### Test coverage

| Layer      | What's tested                                                      |
|------------|--------------------------------------------------------------------|
| `unit/`    | EAR / MAR math, config loading/validation, state machine (PERCLOS, yawn rate, eye-alarm), alarm cooldown |
| `e2e/`     | CNN model loads & classifies real face photos; Flask server responds with correct JSON on every endpoint; full video-pipeline (alert / drowsy / yawning / mixed) |
| `system/`  | FPS ≥ 8 on CPU; per-frame p95 latency < 200 ms; no memory leak over 100 frames; deterministic output across runs |

### Test assets

Test videos under `tests/assets/test_*.mp4` are **real MP4 files**
built from real face photos (stored under `tests/assets/face/`).
Regenerate them with:

```bash
python scripts/make_test_video.py --out tests/assets/test_mixed.mp4 --pattern mixed
```

---

## Datasets & training

The original model was trained on a single small Kaggle dataset
(~2,900 images).  For production use we recommend retraining on a
larger, more diverse dataset.

### Recommended datasets

| Dataset      | Purpose                  | Size          | Source |
|--------------|--------------------------|---------------|--------|
| Drowsiness (Kaggle) | Original 4-class baseline | ~2,900 imgs | [Kaggle](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) |
| YAWDD        | Realistic yawning in car | 322 videos    | [YAWDD](https://www.kaggle.com/datasets/yawning-detection-dataset) |
| UTA-RLDD     | Multi-stage drowsiness   | 180 videos, 30 h | [UTA-RLDD](https://sites.google.com/view/utarldd/home) |
| MRL Eye      | Eye-state robustness (glasses, lighting) | 84,898 imgs | [MRL](http://mrl.cs.vsb.at/eyedataset) |
| CEW          | Closed Eyes in the Wild | 1,192 imgs    | [CEW](https://www.kaggle.com/datasets/drpineapple/closed-eyes-in-the-wild) |
| NTHU-DDD     | Driver drowsiness benchmark | 36 subjects | [NTHU-DDD](https://ipcv.cs.nctu.edu.tw/nthu-ddd/) |

### Retraining (Google Colab)

```bash
# In Colab:
!pip install -r requirements.txt
!python training/train_colab.py \
    --task eye \
    --auto-download \
    --kaggle-username YOUR_USERNAME \
    --kaggle-key YOUR_KEY \
    --epochs 50 \
    --fine-tune 30 \
    --output ./models
```

The Colab script trains a **MobileNetV3-Small** based model (~150K
trainable params, vs 495K in the original) and exports SavedModel, H5,
and **TFLite** (for embedded deployment).

---

## Performance on Intel Core i5-8265U (4 GB RAM)

| Metric                     | Original | New       |
|----------------------------|----------|-----------|
| FPS (CPU only)             | <5       | ~15 (target) / 28+ (measured) |
| Per-frame latency (p95)    | >400 ms  | <200 ms   |
| Yawning detection          | ❌ never | ✅ real-time |
| Memory growth (100 frames) | unstable | <50 MB    |
| Deterministic output       | ❌       | ✅         |

(Benchmarks verified by `tests/system/test_fps_benchmark.py`.)

---

## Project layout

```
Driver_Drowsiness_Detection/
├── drowsiness/                 # main package
│   ├── __init__.py
│   ├── config.py               # dataclass + YAML config
│   ├── detectors/
│   │   ├── base.py             # Detector protocol + Detection dataclass
│   │   ├── geometric.py        # MediaPipe FaceMesh EAR/MAR detector
│   │   └── cnn.py              # Optional Keras CNN verifier
│   ├── video/
│   │   └── source.py           # VideoSource wrapper
│   ├── alarms/
│   │   ├── base.py             # Alarm interface
│   │   └── sound_alarm.py     # pygame-based sound alarm
│   ├── state.py                # DrowsinessTracker state machine
│   └── app.py                  # DrowsinessMonitor orchestrator + CLI
├── server.py                   # Flask status & control server
├── main.py                     # Top-level CLI dispatcher
├── config.yaml                 # All tunable thresholds
├── requirements.txt
├── tests/
│   ├── unit/                   # 43 tests — math, config, state, alarm
│   ├── e2e/                    # 24 tests — CNN, Flask server, video pipeline
│   ├── system/                 # 4 tests — FPS, latency, memory, determinism
│   └── assets/                 # Real face photos + generated test MP4s
├── training/
│   └── train_colab.py          # MobileNetV3-Small training (Colab)
├── scripts/
│   ├── make_test_video.py      # Generate deterministic test videos
│   └── verify_with_curl.sh     # End-to-end curl verification
├── data/                       # Haar cascades (legacy) + alarm.mp3
├── Required_files/             # Original project's reference images
├── drowiness_new7.h5           # Original legacy model (kept for CNN verifier)
├── driver-drowsiness_notebook.ipynb  # Original notebook (kept for reference)
└── detect_drowsiness.py        # Original script (kept for reference)
```

---

## License

See [LICENSE](./LICENSE).
