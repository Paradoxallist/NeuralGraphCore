"""Autonomous internal neuron that produces periodic binary spikes."""

from dataclasses import dataclass
from typing import cast

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

    Use ``configure_timer`` for atomic live period/phase changes. Current and
    initial state are edited independently through ``set_state`` and
    ``set_initial_state``.
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
        """Return the current positive local pulse period."""
        return self._period_ticks

    @property
    def ticks_since_spike(self) -> int:
        """Return the current local timer phase."""
        return self._ticks_since_spike

    @property
    def spike(self) -> int:
        """Return the currently committed binary pulse."""
        return self._spike

    @property
    def initial_ticks_since_spike(self) -> int:
        """Return the local timer phase restored by ``reset``."""
        return self._initial_ticks_since_spike

    @property
    def initial_spike(self) -> int:
        """Return the binary pulse restored by ``reset``."""
        return self._initial_spike

    def configure_timer(
        self,
        *,
        period_ticks: int | None = None,
        ticks_since_spike: int | None = None,
        initial_ticks_since_spike: int | None = None,
    ) -> None:
        """Atomically configure period, current phase, and initial phase.

        Omitted values preserve their current counterparts. Both phases are
        validated against the resulting period before anything changes. A
        shorter period therefore raises an error when either retained phase is
        incompatible unless compatible replacements are supplied explicitly.
        No modulo or clamping is performed.
        """
        if (
            period_ticks is None
            and ticks_since_spike is None
            and initial_ticks_since_spike is None
        ):
            raise ValueError("configure_timer requires at least one value")

        next_period = self._period_ticks if period_ticks is None else period_ticks
        self._validate_period(next_period)
        next_current = (
            self._ticks_since_spike
            if ticks_since_spike is None
            else ticks_since_spike
        )
        next_initial = (
            self._initial_ticks_since_spike
            if initial_ticks_since_spike is None
            else initial_ticks_since_spike
        )
        self._validate_phase(next_current, next_period, "ticks_since_spike")
        self._validate_phase(
            next_initial,
            next_period,
            "initial_ticks_since_spike",
        )

        self._period_ticks = next_period
        self._ticks_since_spike = next_current
        self._initial_ticks_since_spike = next_initial

    def set_state(
        self,
        *,
        spike: int | None = None,
        ticks_since_spike: int | None = None,
    ) -> None:
        """Atomically edit only the current binary pulse and timer phase."""
        if spike is None and ticks_since_spike is None:
            raise ValueError("set_state requires spike or ticks_since_spike")
        next_spike = self._spike if spike is None else binary_spike(spike)
        next_phase = (
            self._ticks_since_spike
            if ticks_since_spike is None
            else ticks_since_spike
        )
        self._validate_phase(next_phase, self._period_ticks, "ticks_since_spike")

        self._spike = next_spike
        self._output = float(next_spike)
        self._ticks_since_spike = next_phase

    def set_initial_state(
        self,
        *,
        spike: int | None = None,
        ticks_since_spike: int | None = None,
    ) -> None:
        """Atomically edit only the pulse and phase restored by ``reset``."""
        if spike is None and ticks_since_spike is None:
            raise ValueError("set_initial_state requires spike or ticks_since_spike")
        next_spike = (
            self._initial_spike if spike is None else binary_spike(spike)
        )
        next_phase = (
            self._initial_ticks_since_spike
            if ticks_since_spike is None
            else ticks_since_spike
        )
        self._validate_phase(next_phase, self._period_ticks, "ticks_since_spike")

        self._initial_spike = next_spike
        self._initial_output = float(next_spike)
        self._initial_ticks_since_spike = next_phase

    def set_initial_state_from_current(self) -> None:
        """Use the current binary pulse and phase as the future reset target."""
        self._initial_spike = self._spike
        self._initial_output = float(self._spike)
        self._initial_ticks_since_spike = self._ticks_since_spike

    @staticmethod
    def _validate_period(period_ticks: int) -> None:
        """Validate one positive integer period."""
        if (
            not isinstance(period_ticks, int)
            or isinstance(period_ticks, bool)
            or period_ticks <= 0
        ):
            raise ValueError("period_ticks must be a positive integer")

    @staticmethod
    def _validate_phase(phase: int, period_ticks: int, name: str) -> None:
        """Validate one integer timer phase against a period."""
        if not isinstance(phase, int) or isinstance(phase, bool):
            raise ValueError(f"{name} must be an integer")
        if not 0 <= phase < period_ticks:
            raise ValueError(f"{name} must be in [0, period_ticks)")

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
        """Commit a binary pulse and timer fully validated during preparation."""
        state = cast(PulsatingInternalState, update.internal_state)
        super().apply_update(update)
        self._spike = state.spike
        self._ticks_since_spike = state.ticks_since_spike

    def reset(self) -> None:
        """Restore the currently configured initial spike and timer phase."""
        super().reset()
        self._spike = self._initial_spike
        self._ticks_since_spike = self._initial_ticks_since_spike
