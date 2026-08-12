import pytest

from neural_graph_core import InputNeuron, Network, OutputNeuron, StatefulNeuron


def test_positive_and_negative_weights_change_incoming_signal() -> None:
    network = Network()
    positive = network.add_neuron(InputNeuron(id="positive"))
    negative = network.add_neuron(InputNeuron(id="negative"))
    target = network.add_neuron(StatefulNeuron(id="target", threshold=10.0))
    network.connect(positive, target, weight=2.0)
    network.connect(negative, target, weight=-3.0)

    from neural_graph_core import NetworkRunner

    state = NetworkRunner(network).step(
        {"positive": 0.5, "negative": 0.25}
    ).states["target"]

    assert state.incoming_signal == 0.25


def test_synapse_weight_change_affects_next_tick() -> None:
    from neural_graph_core import NetworkRunner

    network = Network()
    source = network.add_neuron(InputNeuron(id="source"))
    target = network.add_neuron(StatefulNeuron(id="target", threshold=10.0))
    synapse = network.connect(source, target, weight=1.0)
    runner = NetworkRunner(network)

    first = runner.step({"source": 1.0})
    synapse.weight = -2.0
    second = runner.step({"source": 1.0})

    assert first.states["target"].incoming_signal == 1.0
    assert second.states["target"].incoming_signal == -2.0


def test_disabled_synapse_contributes_zero() -> None:
    from neural_graph_core import NetworkRunner

    network = Network()
    source = network.add_neuron(InputNeuron(id="source"))
    target = network.add_neuron(StatefulNeuron(id="target"))
    network.connect(source, target, weight=10.0, enabled=False)

    state = NetworkRunner(network).step({"source": 1.0}).states["target"]

    assert state.incoming_signal == 0.0
    assert state.spike == 0


def test_synapse_endpoints_are_read_only_and_controls_are_validated() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="source"))
    target = network.add_neuron(StatefulNeuron(id="target"))
    synapse = network.connect(source, target)

    synapse.weight = -2.5
    synapse.enabled = False

    assert synapse.weight == -2.5
    assert synapse.enabled is False
    with pytest.raises(AttributeError):
        synapse.source_id = "other"
    with pytest.raises(ValueError):
        synapse.weight = float("nan")
    with pytest.raises(TypeError):
        synapse.enabled = 1


def test_input_cannot_receive_and_output_cannot_emit() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="source"))
    hidden = network.add_neuron(StatefulNeuron(id="hidden"))
    output = network.add_neuron(OutputNeuron(id="output"))

    with pytest.raises(ValueError):
        network.connect(hidden, source)
    with pytest.raises(ValueError):
        network.connect(output, hidden)


def test_duplicate_synapse_is_rejected() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="source"))
    target = network.add_neuron(StatefulNeuron(id="target"))
    network.connect(source, target)

    with pytest.raises(ValueError):
        network.connect(source, target)


def test_removing_neuron_removes_connected_synapses() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="source"))
    hidden = network.add_neuron(StatefulNeuron(id="hidden"))
    output = network.add_neuron(OutputNeuron(id="output"))
    network.connect(source, hidden)
    network.connect(hidden, output)

    network.remove_neuron(hidden)

    assert len(network.synapses) == 0
