import unittest

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


class RunnerSemanticsTests(unittest.TestCase):
    def test_external_input_is_current_tick_and_internal_spike_is_next_tick(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="input"))
        a = network.add_neuron(StatefulNeuron(id="a"))
        b = network.add_neuron(OutputNeuron(id="b"))
        network.connect(source, a)
        network.connect(a, b)
        runner = NetworkRunner(network)

        first = runner.step({"input": 1.0})
        second = runner.step({})

        self.assertEqual(first.states["a"].spike, 1)
        self.assertEqual(first.outputs["b"].spike, 0)
        self.assertEqual(second.states["a"].spike, 0)
        self.assertEqual(second.outputs["b"].spike, 1)

    def test_missing_input_is_zero_and_unknown_input_is_error(self) -> None:
        network = Network()
        network.add_neuron(InputNeuron(id="known", value=1.0))
        runner = NetworkRunner(network)

        result = runner.step({})
        self.assertEqual(result.states["known"].value, 0.0)

        with self.assertRaises(KeyError):
            runner.step({"unknown": 1.0})

    def test_neuron_insertion_order_does_not_change_dynamics(self) -> None:
        def build(order: tuple[str, str]) -> NetworkRunner:
            network = Network()
            source = network.add_neuron(InputNeuron(id="input"))
            neurons = {
                name: network.add_neuron(StatefulNeuron(id=name))
                for name in order
            }
            network.connect(source, neurons["a"])
            network.connect(neurons["a"], neurons["b"])
            return NetworkRunner(network)

        left = build(("a", "b")).run(({"input": 1.0}, {}, {}))
        right = build(("b", "a")).run(({"input": 1.0}, {}, {}))

        left_spikes = [(result.states["a"].spike, result.states["b"].spike) for result in left]
        right_spikes = [(result.states["a"].spike, result.states["b"].spike) for result in right]
        self.assertEqual(left_spikes, right_spikes)

    def test_recurrent_loop_alternates_committed_spikes(self) -> None:
        network = Network()
        a = network.add_neuron(StatefulNeuron(id="a", spike=1))
        b = network.add_neuron(StatefulNeuron(id="b", spike=0))
        network.connect(a, b)
        network.connect(b, a)
        results = NetworkRunner(network).run(({}, {}, {}))

        self.assertEqual(
            [(r.states["a"].spike, r.states["b"].spike) for r in results],
            [(0, 1), (1, 0), (0, 1)],
        )

    def test_self_loop_uses_previous_committed_spike(self) -> None:
        network = Network()
        neuron = network.add_neuron(StatefulNeuron(id="a", spike=1))
        network.connect(neuron, neuron)
        result = NetworkRunner(network).step({})

        self.assertEqual(result.states["a"].incoming_signal, 1.0)
        self.assertEqual(result.states["a"].spike, 1)

    def test_disabled_synapse_contributes_zero(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="input"))
        target = network.add_neuron(StatefulNeuron(id="target"))
        network.connect(source, target, weight=10.0, enabled=False)

        result = NetworkRunner(network).step({"input": 1.0})

        self.assertEqual(result.states["target"].incoming_signal, 0.0)
        self.assertEqual(result.states["target"].spike, 0)

    def test_failed_preparation_is_atomic(self) -> None:
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

        with self.assertRaises(RuntimeError):
            runner.step({"input": 1.0})

        self.assertEqual(runner.tick, 0)
        self.assertEqual(source.value, 0.0)
        self.assertEqual(good.spike, 0)
        self.assertEqual(good.potential, 0.0)

    def test_step_result_contains_typed_snapshots(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="input"))
        hidden = network.add_neuron(StatefulNeuron(id="hidden"))
        output = network.add_neuron(OutputNeuron(id="output"))
        clock = network.add_neuron(PulsatingNeuron(id="clock", period_ticks=2))
        network.connect(source, hidden)
        network.connect(hidden, output)

        result = NetworkRunner(network).step({"input": 1.0})

        self.assertIsInstance(result.states["input"], InputStepState)
        self.assertIsInstance(result.states["hidden"], StatefulStepState)
        self.assertIsInstance(result.states["clock"], PulsatingStepState)
        self.assertIsInstance(result.outputs["output"], StatefulStepState)

    def test_reset_restores_dynamic_state_without_changing_topology(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="input", value=0.25))
        hidden = network.add_neuron(
            StatefulNeuron(id="hidden", potential=0.5, spike=1)
        )
        synapse = network.connect(source, hidden, weight=2.0)
        runner = NetworkRunner(network)
        runner.step({"input": 1.0})
        synapse.weight = -3.0

        runner.reset()

        self.assertEqual(runner.tick, 0)
        self.assertEqual(source.value, 0.25)
        self.assertEqual(hidden.potential, 0.5)
        self.assertEqual(hidden.spike, 1)
        self.assertIs(network.get_synapse(source, hidden), synapse)
        self.assertEqual(synapse.weight, -3.0)


if __name__ == "__main__":
    unittest.main()
