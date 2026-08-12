"""Integrate-and-fire neuron used for stateful network dynamics."""

from dataclasses import dataclass

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
        retention: float = 0.0,
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

    @property
    def retention(self) -> float:
        """Return the fraction of potential retained between ticks."""
        return self._retention

    @property
    def reset_rule(self) -> ResetRule:
        """Return the configured post-spike potential reset rule."""
        return self._reset_rule

    @property
    def potential(self) -> float:
        """Return the currently committed internal potential."""
        return self._potential

    @property
    def spike(self) -> int:
        """Return the currently committed binary output as integer 0 or 1."""
        return self._spike

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
        """Commit a state validated by ``prepare_update``."""
        state = update.internal_state
        if not isinstance(state, StatefulInternalState):
            raise TypeError("StatefulNeuron requires StatefulInternalState")
        super().apply_update(update)
        self._potential = state.potential
        self._spike = state.spike

    def reset(self) -> None:
        """Restore the constructor-provided potential and binary spike."""
        super().reset()
        self._potential = self._initial_potential
        self._spike = self._initial_spike
