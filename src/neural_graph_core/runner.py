"""Synchronous execution of a neural graph."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from .network import Network
from .neurons import Neuron
from .neurons.base import NeuronUpdate
from .step_result import StepResult
from .synapses import Synapse


class NetworkRunner:
    """Advance a ``Network`` through synchronous discrete ticks.

    External input is applied at the beginning of the current tick. Therefore,
    an ``InputNeuron`` can affect downstream neurons immediately. Signals from
    every other neuron are taken from its previously committed state, including
    pulses produced by ``PulsatingNeuron`` instances.

    Missing values for registered input neurons are treated as ``0.0``. An
    unknown input identifier or an identifier belonging to a non-input neuron
    is rejected.

    The runner calculates every ``NeuronUpdate`` before committing any of them.
    If signal aggregation or ``prepare_update`` fails, all neuron states remain
    unchanged.

    Args:
        network: Network whose state will be advanced. The runner uses the live
            network object, so structural changes made between ``step`` calls
            are visible on the next tick.

    Example:
        Execute one tick::

            runner = NetworkRunner(network)
            result = runner.step(inputs={"sensor": 0.75})
            print(result.tick)
            print(result.outputs)
    """

    __slots__ = ("_network", "_tick")

    def __init__(self, network: Network) -> None:
        if not isinstance(network, Network):
            raise TypeError("network must be a Network instance")
        self._network = network
        self._tick = 0

    @property
    def network(self) -> Network:
        """Return the network controlled by this runner."""
        return self._network

    @property
    def tick(self) -> int:
        """Return the number of successfully completed ticks."""
        return self._tick

    def step(self, inputs: Mapping[str, float] | None = None) -> StepResult:
        """Execute one synchronous network tick.

        Args:
            inputs: External values indexed by input-neuron ID. Missing known
                inputs default to ``0.0``. ``None`` is equivalent to an empty
                mapping.

        Returns:
            A read-only snapshot of all states and output states after commit.

        Raises:
            TypeError: If ``inputs`` is not a mapping.
            KeyError: If an input identifier is not present in the network.
            ValueError: If an identifier belongs to a non-input neuron or any
                neuron rejects its calculated update.
        """
        supplied_inputs = self._copy_inputs(inputs)
        neurons = tuple(self._network.neurons.values())
        synapses = tuple(self._network.synapses.values())

        self._validate_input_ids(supplied_inputs)
        updates, input_signals = self._prepare_input_updates(
            neurons,
            supplied_inputs,
        )
        weighted_inputs = self._aggregate_signals(
            neurons,
            synapses,
            input_signals,
        )

        for neuron in neurons:
            if neuron.role == "input":
                continue
            updates[neuron.id] = neuron.prepare_update(
                weighted_input=weighted_inputs[neuron.id]
            )

        for neuron in neurons:
            neuron.apply_update(updates[neuron.id])

        self._tick += 1
        return self._create_result(neurons)

    def run(
        self,
        input_steps: Iterable[Mapping[str, float] | None],
    ) -> tuple[StepResult, ...]:
        """Execute one tick for every mapping in an input sequence.

        Args:
            input_steps: Iterable containing the external inputs for successive
                ticks. Use ``None`` or an empty mapping for a tick with zeroed
                inputs.

        Returns:
            Results in the same order as the supplied input steps.

        Example:
            Execute three ticks::

                results = runner.run([
                    {"sensor": 1.0},
                    {"sensor": 0.5},
                    None,
                ])
        """
        if isinstance(input_steps, Mapping):
            raise TypeError("input_steps must be an iterable of mappings, not a mapping")
        return tuple(self.step(inputs) for inputs in input_steps)

    def reset(self) -> None:
        """Reset every neuron and set the completed tick count to zero."""
        for neuron in self._network.neurons.values():
            neuron.reset()
        self._tick = 0

    @staticmethod
    def _copy_inputs(
        inputs: Mapping[str, float] | None,
    ) -> dict[str, float]:
        """Copy external inputs so they cannot change during a tick."""
        if inputs is None:
            return {}
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping or None")
        return dict(inputs)

    def _validate_input_ids(self, inputs: Mapping[str, float]) -> None:
        """Ensure every supplied identifier belongs to an input neuron."""
        for neuron_id in inputs:
            neuron = self._network.get_neuron(neuron_id)
            if neuron.role != "input":
                raise ValueError(f"neuron is not an input: {neuron_id!r}")

    @staticmethod
    def _prepare_input_updates(
        neurons: tuple[Neuron, ...],
        inputs: Mapping[str, float],
    ) -> tuple[dict[str, NeuronUpdate], dict[str, float]]:
        """Prepare input states and expose them as current-tick signals."""
        updates: dict[str, NeuronUpdate] = {}
        signals: dict[str, float] = {}
        for neuron in neurons:
            if neuron.role != "input":
                continue
            update = neuron.prepare_update(
                external_input=inputs.get(neuron.id, 0.0)
            )
            updates[neuron.id] = update
            signals[neuron.id] = update.state
        return updates, signals

    @staticmethod
    def _aggregate_signals(
        neurons: tuple[Neuron, ...],
        synapses: tuple[Synapse, ...],
        input_signals: Mapping[str, float],
    ) -> dict[str, float]:
        """Aggregate enabled signals using current inputs and old inner states."""
        weighted_inputs = {neuron.id: 0.0 for neuron in neurons}
        neurons_by_id = {neuron.id: neuron for neuron in neurons}

        for synapse in synapses:
            if not synapse.enabled:
                continue
            source = neurons_by_id[synapse.source_id]
            source_state = (
                input_signals[source.id]
                if source.role == "input"
                else source.state
            )
            weighted_inputs[synapse.target_id] += source_state * synapse.weight
        return weighted_inputs

    def _create_result(self, neurons: tuple[Neuron, ...]) -> StepResult:
        """Create detached read-only state snapshots after a successful commit."""
        states = {neuron.id: neuron.state for neuron in neurons}
        outputs = {
            neuron.id: neuron.state
            for neuron in neurons
            if neuron.role == "output"
        }
        return StepResult(
            tick=self._tick,
            states=MappingProxyType(states),
            outputs=MappingProxyType(outputs),
        )
