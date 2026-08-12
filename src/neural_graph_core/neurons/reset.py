"""Explicit potential-reset rules for integrate-and-fire neurons."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .base_values import finite_float


@runtime_checkable
class ResetRule(Protocol):
    """Calculate the retained potential after one spike."""

    def potential_after_spike(self, *, candidate: float, threshold: float) -> float:
        """Return the next potential after a threshold crossing."""
        ...


@dataclass(frozen=True, slots=True)
class HardReset:
    """Discard all candidate potential after a spike."""

    def potential_after_spike(self, *, candidate: float, threshold: float) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class SubtractiveReset:
    """Retain the candidate potential above the threshold."""

    def potential_after_spike(self, *, candidate: float, threshold: float) -> float:
        return candidate - threshold


@dataclass(frozen=True, slots=True)
class FixedResidualReset:
    """Replace the potential with a configured finite value after a spike."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", finite_float(self.value, "value"))

    def potential_after_spike(self, *, candidate: float, threshold: float) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class PercentageReset:
    """Retain a configured fraction of the candidate after a spike."""

    fraction: float

    def __post_init__(self) -> None:
        converted = finite_float(self.fraction, "fraction")
        if not 0.0 <= converted <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        object.__setattr__(self, "fraction", converted)

    def potential_after_spike(self, *, candidate: float, threshold: float) -> float:
        return candidate * self.fraction
