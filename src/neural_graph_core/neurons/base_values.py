"""Small validation helpers shared by neuron and reset implementations."""

from math import isfinite


def finite_float(value: float, name: str) -> float:
    """Convert a numeric value to a finite float."""
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def binary_spike(value: int, name: str = "spike") -> int:
    """Validate and return an integer spike whose value is exactly 0 or 1."""
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"{name} must be the integer 0 or 1")
    return value
