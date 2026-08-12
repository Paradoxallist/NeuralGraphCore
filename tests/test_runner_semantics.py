from dataclasses import FrozenInstanceError

import pytest

from neural_graph_core import (
    InputNeuron,
    InputStepState,
    Network,
    NetworkRunner,
    OutputNeuron,
    PulsatingNeuron,
    PulsatingStepState,
    StatefulNeuron,
    StatefulStepState,
)


def test_missing_known_input_is_zero_and_unknown_input_is_error() -> None:
    network = Network()
    network.add_neuron(InputNeuron(id="known", value=1.0))
    runner = NetworkRunner(network)

    assert runner.step({}).states["known"].value == 0.0

    with pytest.raises(KeyError):
        runner.step({"unknown": 1.0})


def test_failed_preparation_is_atomic() -> None:
    class FailingReset:
        def potential_after_spike(self, *, candidate: float, threshold: float) -> float:
            raise RuntimeError("failure")

    network = Network()
    source = network.add_neuron(InputNeuron(id="input"))
    good = network.add_neuron(StatefulNeuron(id="good"))
    bad = network.add_neuron(StatefulNeuron(id="bad", reset=FailingReset()))
    network.connect(source, good)
    network.connect(source, bad)
    runner = NetworkRunner(network)

    with pytest.raises(RuntimeError):
        runner.step({"input": 1.0})

    assert runner.tick == 0
    assert source.value == 0.0
    assert good.spike == 0
    assert good.potential == 0.0


def test_step_result_contains_immutable_typed_snapshots() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="input"))
    hidden = network.add_neuron(StatefulNeuron(id="hidden"))
    output = network.add_neuron(OutputNeuron(id="output"))
    clock = network.add_neuron(PulsatingNeuron(id="clock", period_ticks=2))
    network.connect(source, hidden)
    network.connect(hidden, output)

    result = NetworkRunner(network).step({"input": 1.0})

    assert isinstance(result.states["input"], InputStepState)
    assert isinstance(result.states["hidden"], StatefulStepState)
    assert isinstance(result.states["clock"], PulsatingStepState)
    assert isinstance(result.outputs["output"], StatefulStepState)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.states["hidden"].spike = 0
    with pytest.raises(TypeError):
        result.states["new"] = result.states["hidden"]


def test_reset_restores_dynamic_state_without_changing_topology() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="input", value=0.25))
    hidden = network.add_neuron(
        StatefulNeuron(id="hidden", potential=0.5, spike=1)
    )
    clock = network.add_neuron(
        PulsatingNeuron(id="clock", period_ticks=3, ticks_since_spike=1, spike=1)
    )
    synapse = network.connect(source, hidden, weight=2.0)
    runner = NetworkRunner(network)
    runner.step({"input": 1.0})
    synapse.weight = -3.0

    runner.reset()

    assert runner.tick == 0
    assert source.value == 0.25
    assert hidden.potential == 0.5
    assert hidden.spike == 1
    assert clock.spike == 1
    assert clock.ticks_since_spike == 1
    assert network.get_synapse(source, hidden) is synapse
    assert synapse.weight == -3.0


def test_run_matches_successive_step_calls() -> None:
    def build_runner() -> NetworkRunner:
        network = Network()
        source = network.add_neuron(InputNeuron(id="input"))
        target = network.add_neuron(StatefulNeuron(id="target"))
        network.connect(source, target)
        return NetworkRunner(network)

    sequence = ({"input": 0.4}, {"input": 0.6}, {})
    run_results = build_runner().run(sequence)
    step_runner = build_runner()
    step_results = tuple(step_runner.step(inputs) for inputs in sequence)

    assert run_results == step_results
