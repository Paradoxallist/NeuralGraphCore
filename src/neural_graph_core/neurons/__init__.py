"""Built-in neuron implementations.

InputNeuron
    ``environment -> network`` boundary that accepts only external values.
StatefulNeuron
    Binary integrate-and-fire neuron with retained potential.
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
from .reset import (
    FixedResidualReset,
    HardReset,
    PercentageReset,
    ResetRule,
    SubtractiveReset,
)
from .roles import NeuronRole
from .stateful import StatefulNeuron
