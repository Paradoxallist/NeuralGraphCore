"""Autonomous neuron that produces periodic pulses."""

from .base import Neuron, NeuronUpdate, finite_float
from .roles import NeuronRole


class PulsatingNeuron(Neuron):
    """Generate a pulse according to an independent local timer.

    The timer belongs to the neuron and does not use an absolute network tick.
    Different instances can therefore have independent periods and phases, and
    a new instance can be inserted into an already running network.

    Every update increments ``ticks_since_spike``. When it reaches
    ``period_ticks``, the neuron emits ``spike_value`` and resets its timer to
    zero. On other ticks it emits ``resting_value``. Incoming synapses and
    external input are forbidden because the neuron is autonomous.

    Args:
        id: Identifier that must be unique within one network.
        state: Initial public output value.
        period_ticks: Positive number of local updates between pulses.
        ticks_since_spike: Initial phase in the range
            ``0 <= value < period_ticks``.
        spike_value: Public state emitted on a pulse.
        resting_value: Public state emitted between pulses.

    Example:
        Emit a pulse on every third update::

            neuron = PulsatingNeuron(id="clock", period_ticks=3)
            for _ in range(3):
                update = neuron.prepare_update()
                neuron.apply_update(update)
            assert neuron.state == 1.0
            assert neuron.ticks_since_spike == 0
    """

    __slots__ = (
        "_period_ticks",
        "_ticks_since_spike",
        "_initial_ticks_since_spike",
        "_spike_value",
        "_resting_value",
    )

    def __init__(
        self,
        *,
        id: str,
        period_ticks: int,
        state: float = 0.0,
        ticks_since_spike: int = 0,
        spike_value: float = 1.0,
        resting_value: float = 0.0,
    ) -> None:
        super().__init__(id=id, state=state)
        if (
            not isinstance(period_ticks, int)
            or isinstance(period_ticks, bool)
            or period_ticks <= 0
        ):
            raise ValueError("period_ticks must be a positive integer")
        if not isinstance(ticks_since_spike, int) or isinstance(
            ticks_since_spike, bool
        ):
            raise ValueError("ticks_since_spike must be an integer")
        if not 0 <= ticks_since_spike < period_ticks:
            raise ValueError("ticks_since_spike must be in [0, period_ticks)")

        self._period_ticks = period_ticks
        self._ticks_since_spike = ticks_since_spike
        self._initial_ticks_since_spike = ticks_since_spike
        self._spike_value = finite_float(spike_value, "spike_value")
        self._resting_value = finite_float(resting_value, "resting_value")

    @property
    def period_ticks(self) -> int:
        """Return the number of local updates between pulses."""
        return self._period_ticks

    @property
    def ticks_since_spike(self) -> int:
        """Return the current value of the private local timer."""
        return self._ticks_since_spike

    @property
    def spike_value(self) -> float:
        """Return the state emitted when the timer reaches the period."""
        return self._spike_value

    @property
    def resting_value(self) -> float:
        """Return the state emitted between pulses."""
        return self._resting_value

    @property
    def role(self) -> NeuronRole:
        """Return the ``pulsating`` role."""
        return "pulsating"

    @property
    def accepts_incoming(self) -> bool:
        """Return ``False`` because the pulse generator is autonomous."""
        return False

    @property
    def emits_outgoing(self) -> bool:
        """Return ``True`` because generated pulses feed other neurons."""
        return True

    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Prepare the next output value and local timer state.

        The method does not mutate the neuron. The next timer value is stored
        in ``NeuronUpdate.internal_state`` and committed by ``apply_update``.

        Raises:
            ValueError: If a weighted or external input is supplied.
        """
        if weighted_input != 0.0:
            raise ValueError("PulsatingNeuron cannot receive weighted input")
        if external_input is not None:
            raise ValueError("PulsatingNeuron cannot receive external input")

        next_ticks = self._ticks_since_spike + 1
        if next_ticks >= self._period_ticks:
            return NeuronUpdate(self._spike_value, 0)
        return NeuronUpdate(self._resting_value, next_ticks)

    def apply_update(self, update: NeuronUpdate) -> None:
        """Commit both the public state and the private local timer."""
        if not isinstance(update.internal_state, int):
            raise TypeError("PulsatingNeuron update requires an integer internal state")
        if not 0 <= update.internal_state < self._period_ticks:
            raise ValueError("PulsatingNeuron internal state is out of range")
        super().apply_update(update)
        self._ticks_since_spike = update.internal_state

    def reset(self) -> None:
        """Restore the initial public state and constructor-defined phase."""
        super().reset()
        self._ticks_since_spike = self._initial_ticks_since_spike
