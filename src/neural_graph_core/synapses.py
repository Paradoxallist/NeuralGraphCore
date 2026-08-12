"""Directed weighted connections between neurons."""

from math import isfinite


class Synapse:
    """An immutable directed connection between two neurons.

    A future network runner will calculate the transmitted signal as::

        source.state * weight

    ``source_id`` and ``target_id`` form a unique key, so a network permits at
    most one synapse between an ordered pair of neurons. A self-loop is valid,
    which means both identifiers may be equal.

    Args:
        source_id: Identifier of the neuron sending the signal.
        target_id: Identifier of the neuron receiving the signal.
        weight: Finite multiplier applied to the source state.
        enabled: Whether the synapse participates in signal propagation.

    Example:
        Connect an input to a hidden neuron::

            synapse = Synapse(
                source_id="sensor",
                target_id="hidden",
                weight=0.5,
            )
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
        converted_weight = float(weight)
        if not isfinite(converted_weight):
            raise ValueError("weight must be finite")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        self._source_id = source_id
        self._target_id = target_id
        self._weight = converted_weight
        self._enabled = enabled

    @property
    def source_id(self) -> str:
        """Return the identifier of the source neuron."""
        return self._source_id

    @property
    def target_id(self) -> str:
        """Return the identifier of the target neuron."""
        return self._target_id

    @property
    def weight(self) -> float:
        """Return the multiplier applied to the source state."""
        return self._weight

    @property
    def enabled(self) -> bool:
        """Return whether the synapse participates in signal propagation."""
        return self._enabled

    @property
    def key(self) -> tuple[str, str]:
        """Return ``(source_id, target_id)`` for network indexing."""
        return self._source_id, self._target_id

    def __repr__(self) -> str:
        """Return a constructor-like representation useful for debugging."""
        return (
            f"Synapse(source_id={self._source_id!r}, "
            f"target_id={self._target_id!r}, weight={self._weight!r}, "
            f"enabled={self._enabled!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare synapses by endpoints, weight, and enabled state."""
        if not isinstance(other, Synapse):
            return NotImplemented
        return (
            self._source_id == other._source_id
            and self._target_id == other._target_id
            and self._weight == other._weight
            and self._enabled == other._enabled
        )

    def __hash__(self) -> int:
        """Return a stable hash because every stored field is read-only."""
        return hash((self._source_id, self._target_id, self._weight, self._enabled))
