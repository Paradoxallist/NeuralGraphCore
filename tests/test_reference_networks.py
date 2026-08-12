from neural_graph_core import (
    HardReset,
    InputNeuron,
    Network,
    NetworkRunner,
    StatefulNeuron,
)


def test_input_a_b_reference_timing() -> None:
    """External input is current-tick; a new internal spike is next-tick."""
    network = Network()
    source = network.add_neuron(InputNeuron(id="input"))
    a = network.add_neuron(
        StatefulNeuron(id="a", threshold=1.0, retention=0.0, reset=HardReset())
    )
    b = network.add_neuron(
        StatefulNeuron(id="b", threshold=1.0, retention=0.0, reset=HardReset())
    )
    network.connect(source, a, weight=1.0)
    network.connect(a, b, weight=1.0)
    runner = NetworkRunner(network)

    tick_1 = runner.step({"input": 1.0})
    tick_2 = runner.step({"input": 0.0})

    assert (tick_1.states["a"].spike, tick_1.states["b"].spike) == (1, 0)
    assert (tick_2.states["a"].spike, tick_2.states["b"].spike) == (0, 1)


def test_recurrent_reference_spike_circulates() -> None:
    network = Network()
    a = network.add_neuron(
        StatefulNeuron(
            id="a",
            threshold=1.0,
            retention=0.0,
            reset=HardReset(),
            spike=1,
        )
    )
    b = network.add_neuron(
        StatefulNeuron(
            id="b",
            threshold=1.0,
            retention=0.0,
            reset=HardReset(),
            spike=0,
        )
    )
    network.connect(a, b, weight=1.0)
    network.connect(b, a, weight=1.0)

    results = NetworkRunner(network).run(({}, {}, {}, {}))

    assert [
        (result.states["a"].spike, result.states["b"].spike)
        for result in results
    ] == [(0, 1), (1, 0), (0, 1), (1, 0)]


def test_stateful_self_loop_uses_previous_spike() -> None:
    network = Network()
    neuron = network.add_neuron(
        StatefulNeuron(
            id="a",
            threshold=1.0,
            retention=0.0,
            reset=HardReset(),
            spike=1,
        )
    )
    network.connect(neuron, neuron, weight=1.0)

    result = NetworkRunner(network).step({})

    assert result.states["a"].incoming_signal == 1.0
    assert result.states["a"].spike == 1


def test_neuron_storage_order_does_not_change_reference_dynamics() -> None:
    def build(order: tuple[str, str]) -> NetworkRunner:
        network = Network()
        source = network.add_neuron(InputNeuron(id="input"))
        neurons = {
            name: network.add_neuron(
                StatefulNeuron(id=name, threshold=1.0, retention=0.0)
            )
            for name in order
        }
        network.connect(source, neurons["a"])
        network.connect(neurons["a"], neurons["b"])
        return NetworkRunner(network)

    sequence = ({"input": 1.0}, {}, {})
    left = build(("a", "b")).run(sequence)
    right = build(("b", "a")).run(sequence)

    assert [
        (result.states["a"].spike, result.states["b"].spike)
        for result in left
    ] == [
        (result.states["a"].spike, result.states["b"].spike)
        for result in right
    ]
