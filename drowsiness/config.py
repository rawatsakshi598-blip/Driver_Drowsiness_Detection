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
    # DEPRECATED: kept so old config.yaml files still parse; the confirm
    # path is now time-based via ``yawn_confirm_seconds``.
    yawn_frame_threshold: int = 10
    # Seconds MAR must stay above mar_threshold to confirm a yawn.
    # ~0.45 s ≈ fires within ~0.5 s of the yawn opening, at any FPS.
    yawn_confirm_seconds: float = 0.45
    # Seconds a single yawn may last before we flag a PROLONGED_YAWN alert.
    # Long-held yawns are a strong microsleep precursor — they should not
    # wait for the per-minute yawn rate to trip.
    max_yawn_seconds: float = 4.0
    # Maximum yawns per minute above which a fatigue alarm is raised.
    yawn_rate_per_minute: int = 3
    # Window over which the yawn rate is computed.
    yawn_window_seconds: float = 60.0
    mar_smoothing_alpha: float = 0.3


@dataclass(frozen=True)
class LightingConfig:
    """Ambient-lighting checks applied to every frame (face or no face)."""
    # If False, no lighting alerts are raised.
    enabled: bool = True
    # Mean grayscale brightness below which the frame is "too dark".
    # 0 = black, 255 = white; 60 is a practical floor for road lighting.
    min_brightness: float = 60.0
    # Mean grayscale brightness above which the frame is washed out /
    # direct sunlight into the lens.
    max_brightness: float = 240.0
    # Consecutive out-of-range frames before the lighting alert fires
    # (avoids strobing on a single flash).
    consecutive_frames: int = 8
    # Hysteresis: once LOW_LIGHT has fired, brightness must climb above
    # THIS value before it clears (enter < min_brightness, exit > recover).
    recover_brightness: float = 65.0


@dataclass(frozen=True)
class FocusConfig:
    """Camera focus / sharpness check (variance of Laplacian)."""
    # If False, no out-of-focus alerts are raised.
    enabled: bool = True
    # Variance of the Laplacian below which the frame is too blurry to
    # trust eye/mouth geometry.  Tune for your camera: higher = stricter.
    min_sharpness: float = 60.0
    # Consecutive blurry frames before the OUT_OF_FOCUS alert fires.
    consecutive_frames: int = 8


@dataclass(frozen=True)
class DistractionConfig:
    """Driver-attention checks — face missing, or looking away."""
    # If False, no distraction alerts are raised.
    enabled: bool = True
    # Seconds the face may stay undetected before a FACE_NOT_VISIBLE
    # alert (severity: critical) fires.  At 15 FPS this is ~30 frames.
    no_face_seconds: float = 2.0
    # Head-yaw straightness (min/max of nose-to-cheek distances, 1.0 =
    # facing the road).  Below this the driver is considered to be
    # looking away from the road.  Calibration (nose→tragion geometry):
    # ~0.58 ≈ 25–30° head turn; natural small movements stay above 0.7.
    look_away_straightness: float = 0.58
    # Seconds the head must stay turned away before LOOKING_AWAY fires.
    # Time-based (robust at low FPS); ~1.0 s per discussion.
    look_away_seconds: float = 1.0
    # DEPRECATED: frame-count variant kept for old config files; ignored
    # when ``look_away_seconds`` is set.
    look_away_frames: int = 10
    # If True, FACE_NOT_VISIBLE and PROLONGED_YAWN also sound the alarm.
    # Low light / blur / look-away always stay visual + log only.
    alarm_with_sound: bool = True


@dataclass(frozen=True)
class AudioAlertConfig:
    """Per-alert-type audio settings (one entry per alert type)."""
    # Enabled for this alert type (primary on/off switch).
    enabled: bool = True
    # Higher priority wins: a higher-priority alert may pre-empt a
    # lower-priority one; a lower one never blocks a higher one.
    priority: int = 0
    # Minimum seconds between repeat beeps while the alert stays ACTIVE.
    repeat_seconds: float = 4.0
    # PENDING -> ACTIVE delay inside the audio state machine.  Usually
    # 0.0 because the tracker already gates the condition in time (e.g.
    # distraction.look_away_seconds); raise it to add an extra hold-off.
    min_seconds: float = 0.0
    # Playback volume 0.0–1.0.
    volume: float = 0.6
    # Tone frequency for synthesised beeps (ignored for DROWSY, which
    # plays alarm.sound_path when the file exists).
    frequency_hz: float = 440.0


@dataclass(frozen=True)
class AudioConfig:
    """Audio manager configuration — single CONFIG block for ALL alerts.

    DROWSY > LOOK_AWAY > YAWN > LOW_LIGHT priority order by default.
    """
    # Master switch for all alert audio (detection continues regardless).
    enabled: bool = True
    # DEBUG: one line per alert transition (type, state, value, threshold,
    # timestamp).  Off by default to avoid log noise.
    debug: bool = False
    drowsy: AudioAlertConfig = field(
        default_factory=lambda: AudioAlertConfig(
            priority=100, repeat_seconds=4.0, min_seconds=0.0,
            volume=1.0, frequency_hz=880.0,
        )
    )
    look_away: AudioAlertConfig = field(
        default_factory=lambda: AudioAlertConfig(
            priority=60, repeat_seconds=2.0, min_seconds=0.0,
            volume=0.8, frequency_hz=660.0,
        )
    )
    yawn: AudioAlertConfig = field(
        default_factory=lambda: AudioAlertConfig(
            priority=40, repeat_seconds=2.0, min_seconds=0.0,
            volume=0.7, frequency_hz=520.0,
        )
    )
    low_light: AudioAlertConfig = field(
        default_factory=lambda: AudioAlertConfig(
            priority=20, repeat_seconds=4.0, min_seconds=0.0,
            volume=0.35, frequency_hz=330.0,
        )
    )


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
    lighting: LightingConfig = field(default_factory=LightingConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    distraction: DistractionConfig = field(default_factory=DistractionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
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
            "lighting": LightingConfig,
            "focus": FocusConfig,
            "distraction": DistractionConfig,
            "video": VideoConfig,
            "alarm": AlarmConfig,
            "cnn": CNNConfig,
            "server": ServerConfig,
        }
        kwargs = {}
        for key, value in data.items():
            if key == "audio":
                kwargs["audio"] = _build_audio_config(value)
                continue
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


def _build_audio_config(value) -> AudioConfig:
    """Build the nested ``audio:`` section with strict validation."""
    if value is None:
        return AudioConfig()
    if not isinstance(value, dict):
        raise ValueError("'audio' config must be a mapping")
    alert_fields = ("drowsy", "look_away", "yawn", "low_light")
    scalar = {"enabled", "debug"}
    unknown = set(value) - set(alert_fields) - scalar
    if unknown:
        raise ValueError(f"Unknown keys in 'audio' config: {sorted(unknown)}")
    kwargs = {}
    for k in scalar:
        if k in value:
            kwargs[k] = bool(value[k])
    defaults = AudioConfig()
    allowed = set(AudioAlertConfig.__dataclass_fields__)
    for name in alert_fields:
        if name not in value or value[name] is None:
            continue
        sub = value[name]
        if not isinstance(sub, dict):
            raise ValueError(f"'audio.{name}' must be a mapping")
        unknown_sub = set(sub) - allowed
        if unknown_sub:
            raise ValueError(
                f"Unknown keys in 'audio.{name}' config: {sorted(unknown_sub)}"
            )
        base = asdict(getattr(defaults, name))
        base.update(sub)
        kwargs[name] = AudioAlertConfig(**base)
    return AudioConfig(**kwargs)


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
