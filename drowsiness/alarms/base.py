"""Alarm interface."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class AlarmEvent:
    """A single alarm activation."""
    timestamp: datetime
    reason: str
    metric_value: float


class Alarm(abc.ABC):
    """Abstract alarm."""

    @abc.abstractmethod
    def trigger(self, reason: str, metric_value: float = 0.0) -> AlarmEvent:
        """Raise the alarm."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear the alarm state (used when the driver recovers)."""

    @abc.abstractmethod
    def is_active(self) -> bool:
        """Whether the alarm is currently active."""

    @abc.abstractmethod
    def last_event(self) -> Optional[AlarmEvent]:
        """Return the most recent activation, if any."""
