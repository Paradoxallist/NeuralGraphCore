"""Core building blocks for a graph-based neural network.

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
    hidden = network.add_neuron(StatefulNeuron(id="hidden", bias=0.1))
    output = network.add_neuron(OutputNeuron(id="output"))

    network.connect(sensor, hidden, weight=0.5)
    network.connect(hidden, output)

    runner = NetworkRunner(network)
    result = runner.step(inputs={"sensor": 1.0})
"""

from .neurons import (
    InputNeuron,
    Neuron,
    NeuronRole,
    OutputNeuron,
    PulsatingNeuron,
    StatefulNeuron,
)
from .network import Network
from .runner import NetworkRunner
from .step_result import StepResult
from .synapses import Synapse
