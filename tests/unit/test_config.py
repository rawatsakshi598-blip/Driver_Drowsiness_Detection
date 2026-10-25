"""Unit tests for the configuration loader."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from drowsiness.config import (
    Config,
    EyeConfig,
    MouthConfig,
    load_config,
    get_default_config_path,
)


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(data, f)
    return p


def test_default_config_has_sensible_values():
    cfg = Config()
    assert cfg.eye.ear_threshold == 0.20
    assert cfg.eye.closed_frame_threshold == 20
    assert cfg.mouth.mar_threshold == 0.55
    assert cfg.mouth.yawn_frame_threshold == 10
    assert cfg.video.target_fps == 15
    assert cfg.alarm.cooldown_seconds == 4.0
    assert cfg.cnn.input_size == 145
    assert cfg.cnn.class_names == ("yawn", "no_yawn", "Closed", "Open")


def test_load_default_config_yaml_at_project_root():
    """Loading with no path should find ``config.yaml`` at project root."""
    cfg = load_config()
    # The project's config.yaml sets these specific values.
    assert cfg.eye.ear_threshold == 0.20
    assert cfg.video.target_fps == 15


def test_load_yaml_overrides_defaults(tmp_path):
    p = _write_yaml(tmp_path, {
        "eye": {"ear_threshold": 0.15},
        "video": {"target_fps": 30, "width": 1280, "height": 720},
    })
    cfg = Config.from_yaml(p)
    assert cfg.eye.ear_threshold == 0.15
    # Other eye fields keep their defaults.
    assert cfg.eye.closed_frame_threshold == 20
    assert cfg.video.target_fps == 30
    assert cfg.video.width == 1280


def test_load_yaml_rejects_unknown_top_level_key(tmp_path):
    p = _write_yaml(tmp_path, {"nonsense_key": 1})
    with pytest.raises(ValueError, match="Unknown top-level config key"):
        Config.from_yaml(p)


def test_load_yaml_rejects_unknown_nested_key(tmp_path):
    p = _write_yaml(tmp_path, {"eye": {"no_such_field": 1.0}})
    with pytest.raises(ValueError, match="Unknown keys in 'eye'"):
        Config.from_yaml(p)


def test_load_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "does_not_exist.yaml")


def test_load_config_with_overrides():
    """Top-level overrides are merged on top of the YAML."""
    cfg = load_config(overrides={"show_window": False})
    assert cfg.show_window is False
    # Underlying values from config.yaml still apply.
    assert cfg.eye.ear_threshold == 0.20


def test_config_to_dict_roundtrip():
    """``to_dict`` + ``from_dict`` should round-trip without loss."""
    cfg = Config()
    d = cfg.to_dict()
    cfg2 = Config.from_dict(d)
    assert cfg2.to_dict() == d


def test_config_is_immutable():
    """Config dataclasses are frozen — assignment must raise."""
    cfg = Config()
    with pytest.raises(Exception):
        cfg.eye.ear_threshold = 0.99  # type: ignore[misc]


def test_default_config_path_is_project_root():
    p = get_default_config_path()
    assert p.name == "config.yaml"
    assert p.exists(), f"Default config file missing at {p}"
