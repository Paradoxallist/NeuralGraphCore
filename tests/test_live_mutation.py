import pytest

from neural_graph_core import (
    HardReset,
    InputNeuron,
    Network,
    NetworkRunner,
    OutputNeuron,
    PercentageReset,
    PulsatingNeuron,
    StatefulNeuron,
    SubtractiveReset,
)


def test_threshold_live_change_preserves_state_and_affects_next_tick() -> None:
    neuron = StatefulNeuron(id="neuron", threshold=1.0, potential=0.8)

    neuron.threshold = 0.75
    update = neuron.prepare_update()

    assert neuron.potential == 0.8
    assert neuron.spike == 0
    assert update.step_state.spike == 1


def test_retention_live_change_preserves_state_and_affects_next_tick() -> None:
    neuron = StatefulNeuron(id="neuron", threshold=2.0, potential=0.8)

    neuron.retention = 0.5
    update = neuron.prepare_update()

    assert neuron.potential == 0.8
    assert update.step_state.candidate == 0.4
    assert update.step_state.spike == 0


def test_reset_rule_live_change_is_used_by_next_spike() -> None:
    neuron = StatefulNeuron(
        id="neuron",
        threshold=1.0,
        reset=HardReset(),
    )

    neuron.reset_rule = SubtractiveReset()
    update = neuron.prepare_update(weighted_input=1.4)

    assert isinstance(neuron.reset_rule, SubtractiveReset)
    assert update.step_state.potential == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("threshold", 0.0, ValueError),
        ("threshold", float("inf"), ValueError),
        ("retention", -0.1, ValueError),
        ("retention", 1.1, ValueError),
        ("reset_rule", object(), TypeError),
    ],
)
def test_stateful_configuration_setters_validate_without_partial_change(
    attribute: str,
    value,
    error: type[Exception],
) -> None:
    neuron = StatefulNeuron(id="neuron", threshold=1.0, retention=1.0)
    previous = (neuron.threshold, neuron.retention, neuron.reset_rule)

    with pytest.raises(error):
        setattr(neuron, attribute, value)

    assert (neuron.threshold, neuron.retention, neuron.reset_rule) == previous


def test_stateful_set_state_is_atomic_and_synchronizes_output() -> None:
    neuron = StatefulNeuron(id="neuron", potential=0.1, spike=0)

    neuron.set_state(potential=0.75)
    assert neuron.potential == 0.75
    assert neuron.spike == 0
    neuron.set_state(spike=1)

    assert neuron.potential == 0.75
    assert neuron.spike == 1
    assert neuron.output == 1.0

    with pytest.raises(ValueError):
        neuron.set_state(potential=0.5, spike=2)
    assert neuron.potential == 0.75
    assert neuron.spike == 1
    assert neuron.output == 1.0


def test_stateful_current_state_does_not_change_reset_target() -> None:
    neuron = StatefulNeuron(id="neuron", potential=0.1, spike=0)

    neuron.set_state(potential=0.8, spike=1)
    neuron.reset()

    assert neuron.potential == 0.1
    assert neuron.spike == 0
    assert neuron.output == 0.0


def test_stateful_set_initial_state_changes_only_future_reset() -> None:
    neuron = StatefulNeuron(id="neuron", potential=0.1, spike=0)
    neuron.set_state(potential=0.8, spike=0)

    neuron.set_initial_state(potential=0.3, spike=1)

    assert neuron.potential == 0.8
    assert neuron.spike == 0
    assert neuron.initial_potential == 0.3
    assert neuron.initial_spike == 1
    neuron.reset()
    assert neuron.potential == 0.3
    assert neuron.spike == 1
    assert neuron.output == 1.0


def test_stateful_can_copy_current_state_to_initial_state() -> None:
    neuron = StatefulNeuron(id="neuron")
    neuron.set_state(potential=0.65, spike=1)

    neuron.set_initial_state_from_current()
    neuron.set_state(potential=0.0, spike=0)
    neuron.reset()

    assert neuron.potential == 0.65
    assert neuron.spike == 1


def test_output_neuron_inherits_stateful_mutation_api() -> None:
    neuron = OutputNeuron(id="output")

    neuron.threshold = 2.0
    neuron.retention = 0.25
    neuron.reset_rule = PercentageReset(0.5)
    neuron.set_state(potential=0.7, spike=1)

    assert neuron.threshold == 2.0
    assert neuron.retention == 0.25
    assert neuron.potential == 0.7
    assert neuron.output == 1.0
    assert neuron.emits_outgoing is False


def test_pulsating_configure_timer_changes_values_atomically() -> None:
    neuron = PulsatingNeuron(
        id="clock",
        period_ticks=10,
        ticks_since_spike=8,
    )

    neuron.configure_timer(
        period_ticks=5,
        ticks_since_spike=2,
        initial_ticks_since_spike=3,
    )

    assert neuron.period_ticks == 5
    assert neuron.ticks_since_spike == 2
    assert neuron.initial_ticks_since_spike == 3


def test_pulsating_invalid_timer_change_preserves_every_value() -> None:
    neuron = PulsatingNeuron(
        id="clock",
        period_ticks=10,
        ticks_since_spike=8,
    )
    previous = (
        neuron.period_ticks,
        neuron.ticks_since_spike,
        neuron.initial_ticks_since_spike,
    )

    with pytest.raises(ValueError):
        neuron.configure_timer(period_ticks=5)

    assert (
        neuron.period_ticks,
        neuron.ticks_since_spike,
        neuron.initial_ticks_since_spike,
    ) == previous


def test_pulsating_set_state_is_atomic_and_synchronizes_output() -> None:
    neuron = PulsatingNeuron(id="clock", period_ticks=5)

    neuron.set_state(ticks_since_spike=3)
    assert neuron.spike == 0
    assert neuron.ticks_since_spike == 3
    neuron.set_state(spike=1)

    assert neuron.spike == 1
    assert neuron.output == 1.0
    assert neuron.ticks_since_spike == 3

    with pytest.raises(ValueError):
        neuron.set_state(spike=0, ticks_since_spike=5)
    assert neuron.spike == 1
    assert neuron.output == 1.0
    assert neuron.ticks_since_spike == 3


def test_pulsating_current_and_initial_states_are_independent() -> None:
    neuron = PulsatingNeuron(
        id="clock",
        period_ticks=5,
        ticks_since_spike=1,
        spike=0,
    )
    neuron.set_state(spike=1, ticks_since_spike=3)

    neuron.set_initial_state(spike=1, ticks_since_spike=2)

    assert neuron.ticks_since_spike == 3
    neuron.reset()
    assert neuron.spike == 1
    assert neuron.output == 1.0
    assert neuron.ticks_since_spike == 2


def test_pulsating_can_copy_current_state_to_initial_state() -> None:
    neuron = PulsatingNeuron(id="clock", period_ticks=5)
    neuron.set_state(spike=1, ticks_since_spike=4)

    neuron.set_initial_state_from_current()
    neuron.set_state(spike=0, ticks_since_spike=0)
    neuron.reset()

    assert neuron.spike == 1
    assert neuron.ticks_since_spike == 4


def test_live_synapse_change_affects_next_tick_without_resetting_neuron() -> None:
    network = Network()
    source = network.add_neuron(InputNeuron(id="input"))
    target = network.add_neuron(StatefulNeuron(id="target", threshold=10.0))
    synapse = network.connect(source, target, weight=1.0)
    runner = NetworkRunner(network)
    runner.step({"input": 0.5})
    assert target.potential == 0.5

    synapse.weight = 2.0
    synapse.enabled = True
    result = runner.step({"input": 0.5})

    assert result.states["target"].candidate == 1.5
    assert target.potential == 1.5
