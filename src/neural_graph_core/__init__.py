"""Deterministic integrate-and-fire graph-network simulation core.

Classes can be imported directly from the package::

    from neural_graph_core import (
        InputNeuron,
        Network,
        NetworkRunner,
        OutputNeuron,
        StatefulNeuron,
    )

    network = Network()
    sensor = network.add_neuron(InputNeuron(id="sensor"))
    hidden = network.add_neuron(
        StatefulNeuron(id="hidden", threshold=1.0, retention=0.5)
    )
    output = network.add_neuron(OutputNeuron(id="output"))

    network.connect(sensor, hidden, weight=0.5)
    network.connect(hidden, output)

    runner = NetworkRunner(network)
    result = runner.step(inputs={"sensor": 1.0})
"""

__version__ = "0.0.1"

from .neurons import (
    FixedResidualReset,
    HardReset,
    InputNeuron,
    Neuron,
    NeuronRole,
    OutputNeuron,
    PercentageReset,
    PulsatingNeuron,
    ResetRule,
    StatefulNeuron,
    SubtractiveReset,
)
from .network import Network
from .runner import NetworkRunner
from .step_result import StepResult
from .step_state import (
    InputStepState,
    NeuronStepState,
    PulsatingStepState,
    StatefulStepState,
)
from .synapses import Synapse
