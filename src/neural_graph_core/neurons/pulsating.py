"""Autonomous internal neuron that produces periodic binary spikes."""

from dataclasses import dataclass

from ..step_state import PulsatingStepState
from .base import Neuron, NeuronUpdate
from .base_values import binary_spike
from .roles import NeuronRole


@dataclass(frozen=True, slots=True)
class PulsatingInternalState:
    """Validated binary spike and local timer prepared for commit."""

    spike: int
    ticks_since_spike: int


class PulsatingNeuron(Neuron):
    """Generate binary spikes using an independent local timer.

    The timer does not depend on ``NetworkRunner.tick``. A spike generated on
    tick N is transmitted through outgoing synapses on tick N+1, like the spike
    of any other internal neuron.

    Args:
        id: Identifier that must be unique within one network.
        period_ticks: Positive number of local updates between spikes.
        ticks_since_spike: Initial timer phase in
            ``0 <= value < period_ticks``.
        spike: Initial committed binary output restored by ``reset``.
    """

    __slots__ = (
        "_period_ticks",
        "_ticks_since_spike",
        "_initial_ticks_since_spike",
        "_spike",
        "_initial_spike",
    )

    def __init__(
        self,
        *,
        id: str,
        period_ticks: int,
        ticks_since_spike: int = 0,
        spike: int = 0,
    ) -> None:
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
        converted_spike = binary_spike(spike)

        super().__init__(id=id, output=float(converted_spike))
        self._period_ticks = period_ticks
        self._ticks_since_spike = ticks_since_spike
        self._initial_ticks_since_spike = ticks_since_spike
        self._spike = converted_spike
        self._initial_spike = converted_spike

    @property
    def period_ticks(self) -> int:
        return self._period_ticks

    @property
    def ticks_since_spike(self) -> int:
        return self._ticks_since_spike

    @property
    def spike(self) -> int:
        """Return the currently committed binary pulse."""
        return self._spike

    @property
    def role(self) -> NeuronRole:
        return "pulsating"

    @property
    def accepts_incoming(self) -> bool:
        return False

    @property
    def emits_outgoing(self) -> bool:
        return True

    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Advance the timer and prepare exactly one binary pulse state."""
        if weighted_input != 0.0:
            raise ValueError("PulsatingNeuron cannot receive weighted input")
        if external_input is not None:
            raise ValueError("PulsatingNeuron cannot receive external input")

        next_ticks = self._ticks_since_spike + 1
        spike = int(next_ticks >= self._period_ticks)
        if spike:
            next_ticks = 0
        state = PulsatingInternalState(
            spike=spike,
            ticks_since_spike=next_ticks,
        )
        return NeuronUpdate(
            output=float(spike),
            internal_state=state,
            step_state=PulsatingStepState(
                spike=spike,
                ticks_since_spike=next_ticks,
            ),
        )

    def apply_update(self, update: NeuronUpdate) -> None:
        """Commit a binary pulse and timer validated by ``prepare_update``."""
        state = update.internal_state
        if not isinstance(state, PulsatingInternalState):
            raise TypeError("PulsatingNeuron requires PulsatingInternalState")
        super().apply_update(update)
        self._spike = state.spike
        self._ticks_since_spike = state.ticks_since_spike

    def reset(self) -> None:
        """Restore the constructor-provided binary spike and timer phase."""
        super().reset()
        self._spike = self._initial_spike
        self._ticks_since_spike = self._initial_ticks_since_spike
