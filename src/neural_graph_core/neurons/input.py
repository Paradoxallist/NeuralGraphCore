"""Neuron used to pass external values into a network."""

from .base import Neuron, NeuronUpdate, finite_float
from .roles import NeuronRole


class InputNeuron(Neuron):
    """A network input whose value is supplied exclusively by the environment.

    ``InputNeuron`` represents the ``environment -> network`` boundary. It
    rejects incoming synapses and accepts one external value on every tick.
    The value becomes its next state without a bias or activation function.

    Example:
        Manually prepare and commit an external value::

            neuron = InputNeuron(id="temperature")
            update = neuron.prepare_update(external_input=0.75)
            neuron.apply_update(update)
            assert neuron.state == 0.75

        A future network runner will perform these calls automatically.
    """

    __slots__ = ()

    @property
    def role(self) -> NeuronRole:
        """Return the ``input`` role."""
        return "input"

    @property
    def accepts_incoming(self) -> bool:
        """Return ``False`` because an input cannot be a synapse target."""
        return False

    @property
    def emits_outgoing(self) -> bool:
        """Return ``True`` because the external value can feed other neurons."""
        return True

    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Prepare an external value as the next public state.

        Args:
            weighted_input: Must remain ``0.0`` because incoming synapses are
                forbidden.
            external_input: Required finite numeric value from the environment.

        Raises:
            ValueError: If a weighted signal is supplied, the external value is
                missing, or the external value is not finite.
        """
        if weighted_input != 0.0:
            raise ValueError("InputNeuron cannot receive weighted input")
        if external_input is None:
            raise ValueError("InputNeuron requires an external input")
        return NeuronUpdate(finite_float(external_input, "external_input"))
