"""Geometry and hysteresis-based push-up repetition tracking."""

from dataclasses import dataclass
from math import acos, degrees, hypot


def joint_angle(a: tuple[float, float], pivot: tuple[float, float],
                b: tuple[float, float]) -> float:
    """Return the angle at pivot in degrees for three pixel-coordinate points."""
    ux, uy = a[0] - pivot[0], a[1] - pivot[1]
    vx, vy = b[0] - pivot[0], b[1] - pivot[1]
    lengths = hypot(ux, uy) * hypot(vx, vy)
    if lengths == 0:
        raise ValueError("A joint-angle segment has zero length")
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / lengths))
    return degrees(acos(cosine))


@dataclass
class RepCounter:
    """Count down-then-up cycles; thresholds are adjustable prototype heuristics."""

    down_angle: float = 90.0
    up_angle: float = 160.0
    min_body_angle: float = 150.0
    phase: str = "up"
    repetitions: int = 0
    rejected: int = 0
    clean_cycle: bool = True

    def __post_init__(self):
        if not (0 < self.down_angle < self.up_angle <= 180):
            raise ValueError("Require 0 < down_angle < up_angle <= 180")

    def update(self, elbow_angle: float, body_angle: float | None = None) -> str:
        if not (0 <= elbow_angle <= 180):
            raise ValueError("Elbow angle must be within 0..180")
        good_alignment = body_angle is None or body_angle >= self.min_body_angle
        if self.phase == "up" and elbow_angle <= self.down_angle:
            self.phase, self.clean_cycle = "down", good_alignment
        elif self.phase == "down":
            self.clean_cycle &= good_alignment
            if elbow_angle >= self.up_angle:
                if self.clean_cycle:
                    self.repetitions += 1
                else:
                    self.rejected += 1
                self.phase, self.clean_cycle = "up", True
        return "Keep body aligned" if not good_alignment else "Good form"

