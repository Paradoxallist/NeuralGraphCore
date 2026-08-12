import pytest

from neural_graph_core import (
    FixedResidualReset,
    HardReset,
    PercentageReset,
    PulsatingNeuron,
    StatefulNeuron,
    SubtractiveReset,
)


def test_stateful_neuron_without_input_retains_potential_by_default() -> None:
    neuron = StatefulNeuron(id="neuron", threshold=2.0, potential=0.8)

    update = neuron.prepare_update()

    assert neuron.retention == 1.0
    assert update.step_state.candidate == 0.8
    assert update.step_state.potential == 0.8
    assert update.step_state.spike == 0


def test_potential_accumulates_across_ticks() -> None:
    neuron = StatefulNeuron(id="neuron", threshold=1.0)

    first = neuron.prepare_update(weighted_input=0.4)
    neuron.apply_update(first)
    second = neuron.prepare_update(weighted_input=0.6)

    assert first.step_state.potential == 0.4
    assert second.step_state.candidate == 1.0
    assert second.step_state.spike == 1


def test_explicit_retention_decays_potential() -> None:
    neuron = StatefulNeuron(
        id="neuron",
        threshold=2.0,
        retention=0.5,
        potential=0.8,
    )

    first = neuron.prepare_update()
    neuron.apply_update(first)
    second = neuron.prepare_update()

    assert first.step_state.potential == 0.4
    assert second.step_state.potential == 0.2


@pytest.mark.parametrize(
    ("incoming_signal", "expected_spike"),
    [(0.99, 0), (1.0, 1), (1.01, 1)],
)
def test_threshold_is_inclusive(incoming_signal: float, expected_spike: int) -> None:
    neuron = StatefulNeuron(id="neuron", threshold=1.0)

    assert (
        neuron.prepare_update(weighted_input=incoming_signal).step_state.spike
        == expected_spike
    )


@pytest.mark.parametrize(
    ("reset_rule", "expected_potential"),
    [
        (HardReset(), 0.0),
        (SubtractiveReset(), 0.4),
        (FixedResidualReset(value=0.25), 0.25),
        (PercentageReset(fraction=0.5), 0.7),
    ],
)
def test_reset_rules_have_exact_results(reset_rule, expected_potential: float) -> None:
    neuron = StatefulNeuron(id="neuron", threshold=1.0, reset=reset_rule)

    state = neuron.prepare_update(weighted_input=1.4).step_state

    assert state.candidate == 1.4
    assert state.spike == 1
    assert state.potential == pytest.approx(expected_potential)


def test_neuron_emits_at_most_one_spike_per_tick() -> None:
    neuron = StatefulNeuron(
        id="neuron",
        threshold=1.0,
        reset=PercentageReset(fraction=1.0),
    )

    state = neuron.prepare_update(weighted_input=3.0).step_state

    assert state.spike == 1
    assert state.potential == 3.0


def test_initial_potential_and_spike_are_independent_and_resettable() -> None:
    neuron = StatefulNeuron(
        id="neuron",
        threshold=10.0,
        potential=-0.5,
        spike=1,
    )
    neuron.apply_update(neuron.prepare_update())

    neuron.reset()

    assert neuron.potential == -0.5
    assert neuron.spike == 1
    assert neuron.output == 1.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StatefulNeuron(id="bad", threshold=0.0),
        lambda: StatefulNeuron(id="bad", retention=1.1),
        lambda: StatefulNeuron(id="bad", spike=2),
        lambda: PercentageReset(fraction=-0.1),
    ],
)
def test_invalid_configuration_is_rejected(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_pulsating_neuron_uses_binary_local_period() -> None:
    neuron = PulsatingNeuron(id="clock", period_ticks=3, ticks_since_spike=1)

    first = neuron.prepare_update()
    neuron.apply_update(first)
    second = neuron.prepare_update()

    assert first.step_state.spike == 0
    assert first.step_state.ticks_since_spike == 2
    assert second.step_state.spike == 1
    assert second.output == 1.0
