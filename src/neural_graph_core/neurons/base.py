"""Shared contracts and update data for every neuron implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..step_state import NeuronStepState
from .base_values import finite_float
from .roles import NeuronRole


@dataclass(frozen=True, slots=True)
class NeuronUpdate:
    """A fully validated state change prepared for synchronous commit.

    ``prepare_update`` must validate every value placed in this object. A valid
    update returned by ``prepare_update`` must be safe for ``apply_update`` to
    commit without performing calculations that may fail.

    Attributes:
        output: Signal exposed by the neuron after commit.
        internal_state: Private implementation-specific state to commit.
        step_state: Immutable public diagnostic snapshot for ``StepResult``.
    """

    output: float
    internal_state: object | None
    step_state: NeuronStepState


class Neuron(ABC):
    """Base contract implemented by every neuron.

    ``output`` is the value this neuron transmits through its outgoing
    synapses. It is unrelated to the ``OutputNeuron`` class name. Input neurons
    expose an analog float. All built-in internal neurons expose a binary
    signal represented as ``0.0`` or ``1.0``. Internal potential is never an
    output and is never transmitted directly.

    Args:
        id: Non-empty identifier that must be unique within one network.
        output: Initial signal restored by ``reset``.
    """

    __slots__ = ("_id", "_output", "_initial_output")

    def __init__(self, *, id: str, output: float = 0.0) -> None:
        if not isinstance(id, str) or not id:
            raise ValueError("id must be a non-empty string")
        self._id = id
        self._output = finite_float(output, "output")
        self._initial_output = self._output

    @property
    def id(self) -> str:
        """Return the identifier used by the network and synapses."""
        return self._id

    @property
    def output(self) -> float:
        """Return the committed value transmitted by outgoing synapses.

        This property exists on every neuron type and must not be confused
        with the sink-only ``OutputNeuron`` class.
        """
        return self._output

    @property
    @abstractmethod
    def role(self) -> NeuronRole:
        """Return the semantic role used to classify the neuron."""
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
        """Calculate and validate the next state without mutating the neuron."""
        raise NotImplementedError

    def apply_update(self, update: NeuronUpdate) -> None:
        """Commit output already validated by ``prepare_update``.

        The runner passes only the update returned by this neuron's successful
        preparation. Implementations must keep commit simple and non-failing
        for such valid updates.
        """
        self._output = update.output

    def reset(self) -> None:
        """Restore the currently configured initial public output."""
        self._output = self._initial_output
