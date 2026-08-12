"""Analog external source used to pass environment values into a network."""

from ..step_state import InputStepState
from .base import Neuron, NeuronUpdate
from .base_values import finite_float
from .roles import NeuronRole


class InputNeuron(Neuron):
    """A strict analog ``environment -> network`` boundary.

    Input values are supplied through ``NetworkRunner.step`` and are available
    to downstream neurons during that same tick. Missing known inputs become
    ``0.0``. Incoming synapses are forbidden.

    Args:
        id: Identifier that must be unique within one network.
        value: Initial analog value restored by ``reset``.
    """

    __slots__ = ()

    def __init__(self, *, id: str, value: float = 0.0) -> None:
        super().__init__(id=id, output=finite_float(value, "value"))

    @property
    def value(self) -> float:
        """Return the currently committed analog external value."""
        return self._output

    @property
    def role(self) -> NeuronRole:
        return "input"

    @property
    def accepts_incoming(self) -> bool:
        return False

    @property
    def emits_outgoing(self) -> bool:
        return True

    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Prepare and validate the external value for the current tick."""
        if weighted_input != 0.0:
            raise ValueError("InputNeuron cannot receive weighted input")
        if external_input is None:
            raise ValueError("InputNeuron requires an external input")
        value = finite_float(external_input, "external_input")
        return NeuronUpdate(
            output=value,
            internal_state=None,
            step_state=InputStepState(value=value),
        )
