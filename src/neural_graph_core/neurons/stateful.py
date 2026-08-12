"""General-purpose neuron used for internal network dynamics."""

from collections.abc import Callable

from .base import Neuron, NeuronUpdate, finite_float
from .roles import NeuronRole


def identity(value: float) -> float:
    """Return a value unchanged and act as the default linear activation."""
    return value


class StatefulNeuron(Neuron):
    """A hidden neuron computed from the sum of its incoming signals.

    The update formula is::

        next_state = activation(weighted_input + bias)

    Both incoming and outgoing synapses are allowed, so this implementation is
    suitable for hidden nodes, recurrent loops, and self-loops. Its previous
    state affects a later tick only when the graph feeds that state back.

    Args:
        id: Identifier that must be unique within one network.
        state: Initial public output value.
        bias: Constant added before the activation function.
        activation: Any callable accepting and returning a float. The default
            ``identity`` function produces a linear neuron.

    Example:
        Create a neuron with a ReLU-like activation::

            neuron = StatefulNeuron(
                id="hidden",
                bias=-0.5,
                activation=lambda value: max(0.0, value),
            )
            update = neuron.prepare_update(weighted_input=1.25)
            neuron.apply_update(update)
            assert neuron.state == 0.75
    """

    __slots__ = ("_bias", "_activation")

    def __init__(
        self,
        *,
        id: str,
        state: float = 0.0,
        bias: float = 0.0,
        activation: Callable[[float], float] = identity,
    ) -> None:
        super().__init__(id=id, state=state)
        self._bias = finite_float(bias, "bias")
        if not callable(activation):
            raise TypeError("activation must be callable")
        self._activation = activation

    @property
    def bias(self) -> float:
        """Return the constant added before activation."""
        return self._bias

    @property
    def activation(self) -> Callable[[float], float]:
        """Return the callable used to calculate the next state."""
        return self._activation

    @property
    def role(self) -> NeuronRole:
        """Return the ``hidden`` role."""
        return "hidden"

    @property
    def accepts_incoming(self) -> bool:
        """Return ``True`` because hidden neurons consume network signals."""
        return True

    @property
    def emits_outgoing(self) -> bool:
        """Return ``True`` because hidden neurons feed the graph."""
        return True

    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Calculate ``activation(weighted_input + bias)``.

        Args:
            weighted_input: Finite sum of weighted incoming signals.
            external_input: Unsupported for this neuron and therefore required
                to remain ``None``.

        Raises:
            ValueError: If an external value is supplied or the calculation
                produces a non-finite result.
        """
        if external_input is not None:
            raise ValueError("StatefulNeuron cannot receive external input")
        total = finite_float(weighted_input, "weighted_input") + self._bias
        return NeuronUpdate(finite_float(self._activation(total), "activation result"))
