"""Directed weighted connections between neurons."""

from math import isfinite


class Synapse:
    """A directed connection with immutable endpoints and mutable controls.

    ``source_id`` and ``target_id`` cannot change because their ordered pair is
    the key used by ``Network``. ``weight`` and ``enabled`` may be changed
    between ticks through validated property setters.

    The transmitted contribution is ``source.output * weight`` when enabled,
    otherwise zero. Input neurons provide analog outputs; all built-in internal
    neurons provide binary outputs.
    """

    __slots__ = ("_source_id", "_target_id", "_weight", "_enabled")

    def __init__(
        self,
        *,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be a non-empty string")
        self._source_id = source_id
        self._target_id = target_id
        self._weight = self._validate_weight(weight)
        self._enabled = self._validate_enabled(enabled)

    @property
    def source_id(self) -> str:
        """Return the immutable source-neuron identifier."""
        return self._source_id

    @property
    def target_id(self) -> str:
        """Return the immutable target-neuron identifier."""
        return self._target_id

    @property
    def weight(self) -> float:
        """Return the current finite signal multiplier."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        """Set any finite weight without imposing an artificial range."""
        self._weight = self._validate_weight(value)

    @property
    def enabled(self) -> bool:
        """Return whether this connection currently transmits a signal."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable transmission while preserving the connection."""
        self._enabled = self._validate_enabled(value)

    @property
    def key(self) -> tuple[str, str]:
        """Return the immutable ordered endpoint pair used by ``Network``."""
        return self._source_id, self._target_id

    @staticmethod
    def _validate_weight(value: float) -> float:
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("weight must be finite")
        return converted

    @staticmethod
    def _validate_enabled(value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("enabled must be a boolean")
        return value

    def __repr__(self) -> str:
        return (
            f"Synapse(source_id={self._source_id!r}, "
            f"target_id={self._target_id!r}, weight={self._weight!r}, "
            f"enabled={self._enabled!r})"
        )
