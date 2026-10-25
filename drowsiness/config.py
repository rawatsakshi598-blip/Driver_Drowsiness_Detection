"""
Configuration loader for the Driver Drowsiness Detection system.

All tunable parameters live in ``config.yaml`` at the project root and are
deserialised into a typed ``Config`` dataclass tree.  Nothing in the runtime
code reads magic numbers — every threshold comes from this object so the
system is fully reproducible from a single file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices — these are *constants of the model*
# (see MediaPipe FaceMesh canonical landmark map), NOT tunable thresholds.
# They are kept here so they live next to the configuration that uses them.
# ---------------------------------------------------------------------------
class FaceMeshIndices:
    # Six landmark indices per eye used by Soukupova & Cech (2016) adapted
    # to the MediaPipe FaceMesh topology.  These were validated against the
    # canonical mesh shown in https://github.com/google/mediapipe.
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    # Mouth landmarks used to compute Mouth Aspect Ratio (MAR).
    # 13 = upper inner lip centre, 14 = lower inner lip centre,
    # 78 = left mouth corner, 308 = right mouth corner.
    MOUTH_VERTICAL = [13, 14]
    MOUTH_HORIZONTAL = [78, 308]
    # Head pose / face oval landmarks (used for face box rendering only).
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109,
    ]


@dataclass(frozen=True)
class EyeConfig:
    """Eye Aspect Ratio thresholds and timing."""
    # EAR below this value means the eye is considered closed.
    # 0.20 is the canonical value from Soukupova & Cech (2016); we keep it
    # as the default and let the user override per-driver in config.yaml.
    ear_threshold: float = 0.20
    # Number of consecutive closed-eye frames needed before an alarm.
    # At 15 FPS this is ~1.3 s; at 30 FPS ~0.7 s.  Tuned for responsiveness
    # while suppressing single-frame blinks.
    closed_frame_threshold: int = 20
    # PERCLOS window length in seconds (industry standard = 60 s).
    perclos_window_seconds: float = 60.0
    # PERCLOS fraction above which we trigger a fatigue alarm
    # (NHTSA convention: 0.08 = 8 % of the window closed).
    perclos_threshold: float = 0.15
    # Smoothing factor for the EMA over the raw EAR signal
    # (0 = no smoothing, 1 = frozen). 0.3 = light smoothing.
    ear_smoothing_alpha: float = 0.3


@dataclass(frozen=True)
class MouthConfig:
    """Mouth Aspect Ratio thresholds for yawn detection."""
    # MAR above this value indicates an open mouth / yawn.
    mar_threshold: float = 0.55
    # Number of consecutive frames a yawn must persist before we treat it
    # as a real yawn (filters out talking, sneezing, brief mouth openings).
    yawn_frame_threshold: int = 10
    # Maximum yawns per minute above which a fatigue alarm is raised.
    yawn_rate_per_minute: int = 3
    # Window over which the yawn rate is computed.
    yawn_window_seconds: float = 60.0
    mar_smoothing_alpha: float = 0.3


@dataclass(frozen=True)
class VideoConfig:
    """Camera / video source parameters."""
    # 0 = default webcam; paths/URLs are accepted by VideoSource.
    source: object = 0
    # Capture width/height — smaller frames run faster on a low-end CPU.
    width: int = 640
    height: int = 480
    # Target FPS — MediaPipe processes every Nth frame, where N is
    # chosen so we never exceed this rate.  15 FPS keeps latency low on
    # an i5-8265U without losing temporal information.
    target_fps: int = 15
    # Force a backend; None lets OpenCV pick. Use CAP_DSHOW on Windows.
    backend: Optional[int] = None


@dataclass(frozen=True)
class AlarmConfig:
    """Alarm sound configuration."""
    sound_path: str = "data/alarm.mp3"
    # Cooldown between alarm triggers, in seconds, to avoid retriggering
    # while the driver is still drowsy but acknowledged the alarm.
    cooldown_seconds: float = 4.0
    # If True, plays the alarm; if False, only logs (used in headless tests).
    enabled: bool = True


@dataclass(frozen=True)
class CNNConfig:
    """Optional CNN verifier configuration (loaded lazily — only used if
    the geometric detector is ambiguous or the user opts in)."""
    # Path to the .h5 / SavedModel.  Empty string disables the CNN verifier.
    model_path: str = "drowiness_new7.h5"
    # Input size the model was trained on.
    input_size: int = 145
    # Confidence threshold above which we trust the CNN verdict.
    confidence_threshold: float = 0.7
    # Run the CNN verifier only every N frames (it is the slowest component
    # on a CPU-only machine, so we throttle it).
    run_every_n_frames: int = 5
    # Class ordering from the original notebook — DO NOT change without
    # re-training.  Indices: 0=yawn, 1=no_yawn, 2=Closed, 3=Open.
    class_names: tuple = ("yawn", "no_yawn", "Closed", "Open")


@dataclass(frozen=True)
class ServerConfig:
    """Optional Flask status & control server (for curl-based verification)."""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class Config:
    eye: EyeConfig = field(default_factory=EyeConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    # If True, render overlay rectangles / text.  Disable for headless runs.
    show_window: bool = True
    # If True, emit verbose logging.
    debug: bool = False
    # Maximum number of seconds the monitor runs before auto-exiting
    # (used by E2E tests; 0 = run forever).
    run_for_seconds: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "Config":
        """Load a ``Config`` from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a ``Config`` from a (possibly partial) dict.

        Unknown keys are rejected loudly so silent typos in config.yaml
        can never silently fall back to defaults.
        """
        # Map nested keys to their sub-dataclasses.
        sub_keys = {
            "eye": EyeConfig,
            "mouth": MouthConfig,
            "video": VideoConfig,
            "alarm": AlarmConfig,
            "cnn": CNNConfig,
            "server": ServerConfig,
        }
        kwargs = {}
        for key, value in data.items():
            if key in sub_keys:
                if value is None:
                    continue
                sub_cls = sub_keys[key]
                # Validate that every key the user passed is a real field.
                allowed = set(sub_cls.__dataclass_fields__.keys())
                unknown = set(value.keys()) - allowed
                if unknown:
                    raise ValueError(
                        f"Unknown keys in '{key}' config: {sorted(unknown)}"
                    )
                kwargs[key] = sub_cls(**value)
            elif key in cls.__dataclass_fields__:
                kwargs[key] = value
            else:
                raise ValueError(
                    f"Unknown top-level config key '{key}'. "
                    f"Allowed: {sorted(list(sub_keys.keys()) + list(cls.__dataclass_fields__.keys()))}"
                )
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Recursively convert to a plain dict (handy for the status endpoint)."""
        return asdict(self)


# Module-level default — instantiated lazily to keep import side-effects free.
_DEFAULT: Optional[Config] = None


def get_default_config_path() -> Path:
    """Return the default config.yaml path (project root)."""
    here = Path(__file__).resolve().parent.parent
    return here / "config.yaml"


def load_config(path: Optional[str | os.PathLike] = None,
                overrides: Optional[dict] = None) -> Config:
    """Load + optionally override config.

    ``overrides`` is merged on top of the YAML contents at the top level
    only — nested overrides should go in the YAML file itself.
    """
    global _DEFAULT
    cfg_path = Path(path) if path else get_default_config_path()
    if cfg_path.exists():
        cfg = Config.from_yaml(cfg_path)
    else:
        cfg = Config()  # all defaults
    if overrides:
        merged = cfg.to_dict()
        merged.update(overrides)
        cfg = Config.from_dict(merged)
    _DEFAULT = cfg
    return cfg


def get_config() -> Config:
    """Return the most recently loaded config (or load the default)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_config()
    return _DEFAULT
