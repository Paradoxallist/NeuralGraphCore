"""Built-in neuron implementations.

InputNeuron
    ``environment -> network`` boundary that accepts only external values.
StatefulNeuron
    General-purpose hidden neuron for internal and recurrent connections.
OutputNeuron
    ``network -> environment`` boundary with no outgoing synapses.
PulsatingNeuron
    Autonomous periodic signal source with its own local timer.

Every implementation inherits from :class:`Neuron`. A new implementation can
be added by implementing the abstract properties and ``prepare_update``.
"""

from .base import Neuron
from .input import InputNeuron
from .output import OutputNeuron
from .pulsating import PulsatingNeuron
from .roles import NeuronRole
from .stateful import StatefulNeuron
