"""Integrate-and-fire neuron used for stateful network dynamics."""

from dataclasses import dataclass
from typing import cast

from ..step_state import StatefulStepState
from .base import Neuron, NeuronUpdate
from .base_values import binary_spike, finite_float
from .reset import HardReset, ResetRule
from .roles import NeuronRole


@dataclass(frozen=True, slots=True)
class StatefulInternalState:
    """Validated private state prepared for a stateful-neuron commit."""

    potential: float
    spike: int


class StatefulNeuron(Neuron):
    """A binary integrate-and-fire neuron with retained potential.

    For each tick, the neuron computes::

        candidate = previous_potential * retention + incoming_signal
        spike = int(candidate >= threshold)

    If no spike occurs, the candidate becomes the next potential. If a spike
    occurs, the configured reset rule determines the next potential. At most
    one spike can be emitted per tick, even when the retained potential remains
    at or above the threshold.

    Args:
        id: Identifier that must be unique within one network.
        threshold: Positive finite firing threshold.
        retention: Finite fraction in ``[0, 1]`` applied to prior potential.
        reset: One explicit reset-rule object. Defaults to ``HardReset()``.
        potential: Initial finite potential restored by ``reset``.
        spike: Initial committed binary output restored by ``reset``. It is
            intentionally independent of the initial potential and threshold.

    Configuration can be edited through validated property setters between
    ticks. ``set_state`` edits current committed state, while
    ``set_initial_state`` edits the independent state restored by ``reset``.
    """

    __slots__ = (
        "_threshold",
        "_retention",
        "_reset_rule",
        "_potential",
        "_spike",
        "_initial_potential",
        "_initial_spike",
    )

    def __init__(
        self,
        *,
        id: str,
        threshold: float = 1.0,
        retention: float = 1.0,
        reset: ResetRule | None = None,
        potential: float = 0.0,
        spike: int = 0,
    ) -> None:
        converted_threshold = finite_float(threshold, "threshold")
        if converted_threshold <= 0.0:
            raise ValueError("threshold must be positive")
        converted_retention = finite_float(retention, "retention")
        if not 0.0 <= converted_retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")
        converted_potential = finite_float(potential, "potential")
        converted_spike = binary_spike(spike)
        reset_rule = HardReset() if reset is None else reset
        if not isinstance(reset_rule, ResetRule):
            raise TypeError("reset must implement ResetRule")

        super().__init__(id=id, output=float(converted_spike))
        self._threshold = converted_threshold
        self._retention = converted_retention
        self._reset_rule = reset_rule
        self._potential = converted_potential
        self._spike = converted_spike
        self._initial_potential = converted_potential
        self._initial_spike = converted_spike

    @property
    def threshold(self) -> float:
        """Return the positive firing threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the positive finite threshold used by subsequent ticks.

        Changing this configuration preserves current and initial state.
        """
        converted = finite_float(value, "threshold")
        if converted <= 0.0:
            raise ValueError("threshold must be positive")
        self._threshold = converted

    @property
    def retention(self) -> float:
        """Return the fraction of potential retained between ticks."""
        return self._retention

    @retention.setter
    def retention(self, value: float) -> None:
        """Set the finite retention fraction used by subsequent ticks.

        Changing this configuration does not modify accumulated potential.
        """
        converted = finite_float(value, "retention")
        if not 0.0 <= converted <= 1.0:
            raise ValueError("retention must be in [0, 1]")
        self._retention = converted

    @property
    def reset_rule(self) -> ResetRule:
        """Return the configured post-spike potential reset rule."""
        return self._reset_rule

    @reset_rule.setter
    def reset_rule(self, value: ResetRule) -> None:
        """Set the reset rule used by subsequent threshold crossings."""
        if not isinstance(value, ResetRule):
            raise TypeError("reset_rule must implement ResetRule")
        self._reset_rule = value

    @property
    def potential(self) -> float:
        """Return the currently committed internal potential."""
        return self._potential

    @property
    def spike(self) -> int:
        """Return the currently committed binary output as integer 0 or 1."""
        return self._spike

    @property
    def initial_potential(self) -> float:
        """Return the potential restored by ``reset``."""
        return self._initial_potential

    @property
    def initial_spike(self) -> int:
        """Return the binary spike restored by ``reset``."""
        return self._initial_spike

    def set_state(
        self,
        *,
        potential: float | None = None,
        spike: int | None = None,
    ) -> None:
        """Atomically edit only the current committed dynamic state.

        Omitted values preserve their current counterparts. The reset target is
        not changed. Potential and spike remain independent values.

        Raises:
            ValueError: If no value is supplied or a value is invalid.
        """
        if potential is None and spike is None:
            raise ValueError("set_state requires potential or spike")
        next_potential = (
            self._potential
            if potential is None
            else finite_float(potential, "potential")
        )
        next_spike = self._spike if spike is None else binary_spike(spike)

        self._potential = next_potential
        self._spike = next_spike
        self._output = float(next_spike)

    def set_initial_state(
        self,
        *,
        potential: float | None = None,
        spike: int | None = None,
    ) -> None:
        """Atomically edit only the state restored by ``reset``.

        Omitted values preserve their existing reset targets. Current dynamic
        state is not changed until ``reset`` is called.
        """
        if potential is None and spike is None:
            raise ValueError("set_initial_state requires potential or spike")
        next_potential = (
            self._initial_potential
            if potential is None
            else finite_float(potential, "potential")
        )
        next_spike = (
            self._initial_spike if spike is None else binary_spike(spike)
        )

        self._initial_potential = next_potential
        self._initial_spike = next_spike
        self._initial_output = float(next_spike)

    def set_initial_state_from_current(self) -> None:
        """Use the current potential and spike as the future reset target."""
        self._initial_potential = self._potential
        self._initial_spike = self._spike
        self._initial_output = float(self._spike)

    @property
    def role(self) -> NeuronRole:
        """Return the ``hidden`` role."""
        return "hidden"

    @property
    def accepts_incoming(self) -> bool:
        """Return ``True`` because stateful neurons integrate network signals."""
        return True

    @property
    def emits_outgoing(self) -> bool:
        """Return ``True`` because their binary spikes feed the graph."""
        return True

    def prepare_update(
        self,
        *,
        weighted_input: float = 0.0,
        external_input: float | None = None,
    ) -> NeuronUpdate:
        """Prepare one integrate-and-fire transition without mutation."""
        if external_input is not None:
            raise ValueError("StatefulNeuron cannot receive external input")

        incoming_signal = finite_float(weighted_input, "weighted_input")
        candidate = finite_float(
            self._potential * self._retention + incoming_signal,
            "candidate",
        )
        spike = int(candidate >= self._threshold)
        if spike:
            next_potential = finite_float(
                self._reset_rule.potential_after_spike(
                    candidate=candidate,
                    threshold=self._threshold,
                ),
                "reset potential",
            )
        else:
            next_potential = candidate

        internal_state = StatefulInternalState(
            potential=next_potential,
            spike=spike,
        )
        step_state = StatefulStepState(
            potential=next_potential,
            spike=spike,
            incoming_signal=incoming_signal,
            candidate=candidate,
        )
        return NeuronUpdate(
            output=float(spike),
            internal_state=internal_state,
            step_state=step_state,
        )

    def apply_update(self, update: NeuronUpdate) -> None:
        """Commit a state that was fully validated by ``prepare_update``.

        This method deliberately performs no runtime validation. The
        prepare/apply contract guarantees that an update returned by this
        neuron's ``prepare_update`` is safe to commit.
        """
        state = cast(StatefulInternalState, update.internal_state)
        super().apply_update(update)
        self._potential = state.potential
        self._spike = state.spike

    def reset(self) -> None:
        """Restore the currently configured initial potential and spike."""
        super().reset()
        self._potential = self._initial_potential
        self._spike = self._initial_spike
