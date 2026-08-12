"""Graph container responsible for neurons, synapses, and structural rules."""

from types import MappingProxyType
from typing import Mapping

from .neurons import Neuron
from .synapses import Synapse


class Network:
    """Own the structure of a neural graph and enforce its invariants.

    A network stores neurons by their unique identifiers and synapses by their
    ordered ``(source_id, target_id)`` pairs. Structural changes are performed
    through methods on this class so invalid connections cannot enter the
    graph.

    This class does not execute network ticks. Use ``NetworkRunner`` to advance
    the stored neuron states synchronously.

    Example:
        Build a small graph using neuron objects::

            network = Network()
            sensor = network.add_neuron(InputNeuron(id="sensor"))
            hidden = network.add_neuron(StatefulNeuron(id="hidden"))
            output = network.add_neuron(OutputNeuron(id="output"))

            network.connect(sensor, hidden, weight=0.5)
            network.connect("hidden", "output", weight=1.0)

    Notes:
        A neuron identifier must be unique within this network. The same
        identifier may still be used in a different ``Network`` instance.

        Removing a neuron automatically removes all of its incoming and
        outgoing synapses.
    """

    __slots__ = ("_neurons", "_synapses")

    def __init__(self) -> None:
        self._neurons: dict[str, Neuron] = {}
        self._synapses: dict[tuple[str, str], Synapse] = {}

    @property
    def neurons(self) -> Mapping[str, Neuron]:
        """Return a read-only view of neurons indexed by identifier.

        The mapping itself cannot be changed by the caller. Neuron dynamic
        state may still change later through the network runner.
        """
        return MappingProxyType(self._neurons)

    @property
    def synapses(self) -> Mapping[tuple[str, str], Synapse]:
        """Return a read-only view of synapses indexed by endpoint pair."""
        return MappingProxyType(self._synapses)

    def add_neuron(self, neuron: Neuron) -> Neuron:
        """Add a neuron and return the registered instance.

        Args:
            neuron: Concrete neuron instance to register.

        Returns:
            The same instance, allowing assignment while constructing a graph.

        Raises:
            TypeError: If ``neuron`` is not a ``Neuron`` instance.
            ValueError: If another neuron already uses the same identifier.
        """
        if not isinstance(neuron, Neuron):
            raise TypeError("neuron must be a Neuron instance")
        if neuron.id in self._neurons:
            raise ValueError(f"neuron id already exists: {neuron.id!r}")
        self._neurons[neuron.id] = neuron
        return neuron

    def get_neuron(self, neuron_id: str) -> Neuron:
        """Return a neuron by identifier.

        Raises:
            KeyError: If the identifier is not registered in this network.
        """
        try:
            return self._neurons[neuron_id]
        except KeyError:
            raise KeyError(f"unknown neuron id: {neuron_id!r}") from None

    def remove_neuron(self, neuron: str | Neuron) -> Neuron:
        """Remove a neuron and all synapses connected to it.

        Args:
            neuron: Registered neuron instance or its identifier.

        Returns:
            The removed neuron instance.

        Raises:
            KeyError: If the identifier is unknown.
            ValueError: If an object with the same identifier is registered but
                it is not the supplied instance.
        """
        registered = self._resolve_neuron(neuron)
        connected_keys = [
            key
            for key in self._synapses
            if registered.id == key[0] or registered.id == key[1]
        ]
        for key in connected_keys:
            del self._synapses[key]
        del self._neurons[registered.id]
        return registered

    def add_synapse(self, synapse: Synapse) -> Synapse:
        """Validate and add a pre-built synapse.

        Both endpoint neurons must already belong to the network. The source
        must permit outgoing connections, and the target must permit incoming
        connections.

        Args:
            synapse: Synapse containing registered endpoint identifiers.

        Returns:
            The same synapse instance.

        Raises:
            TypeError: If the argument is not a ``Synapse`` instance.
            KeyError: If either endpoint is not registered.
            ValueError: If the endpoint roles reject the connection or the
                endpoint pair is already connected.
        """
        if not isinstance(synapse, Synapse):
            raise TypeError("synapse must be a Synapse instance")

        source = self.get_neuron(synapse.source_id)
        target = self.get_neuron(synapse.target_id)
        self._validate_connection(source, target)

        if synapse.key in self._synapses:
            raise ValueError(
                "synapse already exists: "
                f"{synapse.source_id!r} -> {synapse.target_id!r}"
            )
        self._synapses[synapse.key] = synapse
        return synapse

    def connect(
        self,
        source: str | Neuron,
        target: str | Neuron,
        *,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> Synapse:
        """Create and add a synapse between two registered neurons.

        ``source`` and ``target`` may independently be identifiers or concrete
        neuron objects. When an object is supplied, this method verifies that
        the network contains that exact instance. This prevents accidentally
        connecting an object owned by another network that happens to use the
        same identifier.

        Example:
            Both forms are supported::

                network.connect(source_neuron, target_neuron, weight=0.5)
                network.connect("source", "target", weight=0.5)

        Returns:
            The newly created and registered synapse.
        """
        source_neuron = self._resolve_neuron(source)
        target_neuron = self._resolve_neuron(target)
        return self.add_synapse(
            Synapse(
                source_id=source_neuron.id,
                target_id=target_neuron.id,
                weight=weight,
                enabled=enabled,
            )
        )

    def get_synapse(
        self,
        source: str | Neuron,
        target: str | Neuron,
    ) -> Synapse:
        """Return the synapse connecting an ordered endpoint pair.

        Raises:
            KeyError: If an endpoint or the requested synapse is unknown.
            ValueError: If a supplied neuron object is not the registered
                instance with that identifier.
        """
        source_neuron = self._resolve_neuron(source)
        target_neuron = self._resolve_neuron(target)
        key = (source_neuron.id, target_neuron.id)
        try:
            return self._synapses[key]
        except KeyError:
            raise KeyError(
                f"unknown synapse: {source_neuron.id!r} -> {target_neuron.id!r}"
            ) from None

    def disconnect(
        self,
        source: str | Neuron,
        target: str | Neuron,
    ) -> Synapse:
        """Remove and return the synapse between an ordered endpoint pair."""
        synapse = self.get_synapse(source, target)
        del self._synapses[synapse.key]
        return synapse

    def incoming(self, neuron: str | Neuron) -> tuple[Synapse, ...]:
        """Return all incoming synapses in insertion order."""
        registered = self._resolve_neuron(neuron)
        return tuple(
            synapse
            for synapse in self._synapses.values()
            if synapse.target_id == registered.id
        )

    def outgoing(self, neuron: str | Neuron) -> tuple[Synapse, ...]:
        """Return all outgoing synapses in insertion order."""
        registered = self._resolve_neuron(neuron)
        return tuple(
            synapse
            for synapse in self._synapses.values()
            if synapse.source_id == registered.id
        )

    def _resolve_neuron(self, neuron: str | Neuron) -> Neuron:
        """Resolve an identifier or verify an already registered instance."""
        if isinstance(neuron, str):
            return self.get_neuron(neuron)
        if not isinstance(neuron, Neuron):
            raise TypeError("neuron reference must be a string or Neuron instance")

        registered = self.get_neuron(neuron.id)
        if registered is not neuron:
            raise ValueError(
                f"a different neuron instance is registered as {neuron.id!r}"
            )
        return registered

    @staticmethod
    def _validate_connection(source: Neuron, target: Neuron) -> None:
        """Validate endpoint direction rules defined by neuron classes."""
        if not source.emits_outgoing:
            raise ValueError(
                f"{source.__class__.__name__} {source.id!r} cannot emit outgoing synapses"
            )
        if not target.accepts_incoming:
            raise ValueError(
                f"{target.__class__.__name__} {target.id!r} cannot accept incoming synapses"
            )
