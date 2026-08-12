"""Shared contracts and update data for every neuron implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from .roles import NeuronRole


def finite_float(value: float, name: str) -> float:
    """Convert a numeric value to a finite float.

    Args:
        value: Numeric value to validate.
        name: Parameter name used in error messages.

    Returns:
        The value converted to ``float``.

    Raises:
        TypeError: If the value cannot be converted to ``float``.
        ValueError: If the converted value is NaN or infinite.
    """
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class NeuronUpdate:
    """A prepared state change used during a synchronous network update.

    A network runner first prepares an update for every neuron and only then
    applies all updates. This prevents one neuron from observing another
    neuron's state from the new tick while the current tick is still running.

    Attributes:
        state: Public neuron value for the next tick.
        internal_state: Optional implementation-specific state. For example,
            ``PulsatingNeuron`` uses it for its next local timer value.
    """

    state: float
    internal_state: object | None = None


class Neuron(ABC):
    """Base contract implemented by every neuron.

    The constructor stores values privately. Read access is provided through
    properties, while state changes are restricted to ``apply_update`` and
    ``reset``. This keeps a future ``Network`` in control of synchronous state
    transitions.

    Args:
        id: Non-empty identifier. It must be unique within one network. The
            future ``Network.add_neuron`` method will enforce this because the
            neuron itself does not know which network owns it.
        state: Initial public output value and the value restored by ``reset``.

    Example:
        Create a concrete implementation rather than the abstract base class::

            neuron = StatefulNeuron(id="memory", state=0.25)
            assert neuron.id == "memory"
            assert neuron.state == 0.25
    """

    __slots__ = ("_id", "_state", "_initial_state")

    def __init__(self, *, id: str, state: float = 0.0) -> None:
        if not isinstance(id, str) or not id:
            raise ValueError("id must be a non-empty string")
        self._id = id
        self._state = finite_float(state, "state")
        self._initial_state = self._state

    @property
    def id(self) -> str:
        """Return the identifier used by a network and its synapses."""
        return self._id

    @property
    def state(self) -> float:
        """Return the current public output value of the neuron."""
        return self._state

    @property
    @abstractmethod
    def role(self) -> NeuronRole:
        """Return the semantic role used by a network to classify the neuron."""
        raise NotImplementedError

    @property
    @abstractmethod
    def accepts_incoming(self) -> bool:
        """Return whether a synapse may target this neuron."""
        raise NotImplementedError

    @property
    @abstractmethod
    def emits_outgoing(self) -> bool:
        """Return whether a synapse may originate from this neuron."""
        raise NotImplementedError

    @abstractmethod
    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Calculate the next state without mutating the neuron.

        Args:
            weighted_input: Sum of signals from enabled incoming synapses after
                their weights have been applied.
            external_input: Value supplied by the environment. Only neuron
                implementations that explicitly support external input accept it.

        Returns:
            An update that can later be committed with ``apply_update``.

        A network runner is expected to call this method. Application code
        normally interacts with the higher-level network API instead.
        """
        raise NotImplementedError

    def apply_update(self, update: NeuronUpdate) -> None:
        """Commit a previously prepared update.

        Args:
            update: Result returned by this neuron's ``prepare_update`` method.
        """
        self._state = finite_float(update.state, "update.state")

    def reset(self) -> None:
        """Restore the public state supplied to the constructor."""
        self._state = self._initial_state
