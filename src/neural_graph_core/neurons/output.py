"""Terminal integrate-and-fire neuron exposed to the environment."""

from .roles import NeuronRole
from .stateful import StatefulNeuron


class OutputNeuron(StatefulNeuron):
    """A sink-only integrate-and-fire network output.

    Potential integration, thresholding, retention, reset, and binary spike
    behavior are inherited unchanged from ``StatefulNeuron``. The network may
    use this neuron as a target and external readout, but never as a source.
    """

    __slots__ = ()

    @property
    def role(self) -> NeuronRole:
        """Return the ``output`` role."""
        return "output"

    @property
    def emits_outgoing(self) -> bool:
        """Return ``False`` because an output is a terminal graph node."""
        return False
