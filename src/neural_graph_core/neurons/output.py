"""Terminal neuron used to expose a network result to its environment."""

from .roles import NeuronRole
from .stateful import StatefulNeuron


class OutputNeuron(StatefulNeuron):
    """A stateful network output that cannot have outgoing synapses.

    Calculation is inherited from ``StatefulNeuron``. The only behavioral
    difference is its terminal role: a network may use this neuron as a
    synapse target and expose its state as output, but may not connect it to
    another neuron as a source.

    Example:
        Create and manually update a linear output::

            output = OutputNeuron(id="action")
            update = output.prepare_update(weighted_input=0.8)
            output.apply_update(update)
            assert output.state == 0.8
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
