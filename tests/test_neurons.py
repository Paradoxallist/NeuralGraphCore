import unittest

from neural_graph_core import (
    FixedResidualReset,
    HardReset,
    PercentageReset,
    PulsatingNeuron,
    StatefulNeuron,
    SubtractiveReset,
)


class StatefulNeuronTests(unittest.TestCase):
    def test_retention_participates_in_next_candidate(self) -> None:
        neuron = StatefulNeuron(
            id="neuron",
            threshold=2.0,
            retention=0.5,
            potential=0.8,
        )

        first = neuron.prepare_update()
        self.assertEqual(first.step_state.candidate, 0.4)
        self.assertEqual(first.step_state.potential, 0.4)
        self.assertEqual(first.step_state.spike, 0)
        neuron.apply_update(first)

        second = neuron.prepare_update()
        self.assertEqual(second.step_state.candidate, 0.2)

    def test_threshold_is_inclusive(self) -> None:
        below = StatefulNeuron(id="below", threshold=1.0)
        equal = StatefulNeuron(id="equal", threshold=1.0)
        above = StatefulNeuron(id="above", threshold=1.0)

        self.assertEqual(below.prepare_update(weighted_input=0.99).step_state.spike, 0)
        self.assertEqual(equal.prepare_update(weighted_input=1.0).step_state.spike, 1)
        self.assertEqual(above.prepare_update(weighted_input=1.01).step_state.spike, 1)

    def test_all_reset_rules_have_exact_results(self) -> None:
        cases = (
            (HardReset(), 0.0),
            (SubtractiveReset(), 0.4),
            (FixedResidualReset(value=0.25), 0.25),
            (PercentageReset(fraction=0.5), 0.7),
        )

        for rule, expected in cases:
            with self.subTest(rule=rule):
                neuron = StatefulNeuron(id="neuron", threshold=1.0, reset=rule)
                update = neuron.prepare_update(weighted_input=1.4)
                self.assertEqual(update.step_state.spike, 1)
                self.assertAlmostEqual(update.step_state.potential, expected)

    def test_one_spike_maximum_per_tick(self) -> None:
        neuron = StatefulNeuron(
            id="neuron",
            threshold=1.0,
            reset=PercentageReset(fraction=1.0),
        )

        update = neuron.prepare_update(weighted_input=3.0)

        self.assertEqual(update.step_state.spike, 1)
        self.assertEqual(update.step_state.potential, 3.0)

    def test_initial_potential_and_spike_are_independent_and_resettable(self) -> None:
        neuron = StatefulNeuron(
            id="neuron",
            threshold=10.0,
            potential=-0.5,
            spike=1,
        )
        neuron.apply_update(neuron.prepare_update())

        neuron.reset()

        self.assertEqual(neuron.potential, -0.5)
        self.assertEqual(neuron.spike, 1)
        self.assertEqual(neuron.output, 1.0)

    def test_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            StatefulNeuron(id="bad", threshold=0.0)
        with self.assertRaises(ValueError):
            StatefulNeuron(id="bad", retention=1.1)
        with self.assertRaises(ValueError):
            StatefulNeuron(id="bad", spike=2)
        with self.assertRaises(ValueError):
            PercentageReset(fraction=-0.1)


class PulsatingNeuronTests(unittest.TestCase):
    def test_pulses_are_binary_and_use_local_timer(self) -> None:
        neuron = PulsatingNeuron(id="clock", period_ticks=3, ticks_since_spike=1)

        first = neuron.prepare_update()
        neuron.apply_update(first)
        second = neuron.prepare_update()

        self.assertEqual(first.step_state.spike, 0)
        self.assertEqual(first.step_state.ticks_since_spike, 2)
        self.assertEqual(second.step_state.spike, 1)
        self.assertEqual(second.output, 1.0)


if __name__ == "__main__":
    unittest.main()
