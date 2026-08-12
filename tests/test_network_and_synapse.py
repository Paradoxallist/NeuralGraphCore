import unittest

from neural_graph_core import InputNeuron, Network, OutputNeuron, StatefulNeuron


class NetworkAndSynapseTests(unittest.TestCase):
    def test_synapse_controls_are_mutable_and_endpoints_are_read_only(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="source"))
        target = network.add_neuron(StatefulNeuron(id="target"))
        synapse = network.connect(source, target)

        synapse.weight = -2.5
        synapse.enabled = False

        self.assertEqual(synapse.weight, -2.5)
        self.assertFalse(synapse.enabled)
        with self.assertRaises(AttributeError):
            synapse.source_id = "other"
        with self.assertRaises(ValueError):
            synapse.weight = float("nan")
        with self.assertRaises(TypeError):
            synapse.enabled = 1

    def test_direction_and_uniqueness_invariants_remain_enforced(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="source"))
        hidden = network.add_neuron(StatefulNeuron(id="hidden"))
        output = network.add_neuron(OutputNeuron(id="output"))
        network.connect(source, hidden)

        with self.assertRaises(ValueError):
            network.connect(source, hidden)
        with self.assertRaises(ValueError):
            network.connect(hidden, source)
        with self.assertRaises(ValueError):
            network.connect(output, hidden)

    def test_removing_neuron_removes_connected_synapses(self) -> None:
        network = Network()
        source = network.add_neuron(InputNeuron(id="source"))
        hidden = network.add_neuron(StatefulNeuron(id="hidden"))
        output = network.add_neuron(OutputNeuron(id="output"))
        network.connect(source, hidden)
        network.connect(hidden, output)

        network.remove_neuron(hidden)

        self.assertEqual(len(network.synapses), 0)


if __name__ == "__main__":
    unittest.main()
