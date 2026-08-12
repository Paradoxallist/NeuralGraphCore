"""Immutable public result of one completed network tick."""

from collections.abc import Mapping
from dataclasses import dataclass

from .step_state import NeuronStepState, StatefulStepState


@dataclass(frozen=True, slots=True)
class StepResult:
    """Typed diagnostic snapshot returned after a successful tick.

    Attributes:
        tick: Number of successfully completed ticks.
        states: Snapshot for every neuron indexed by ID. Input, stateful, and
            pulsating neurons expose different immutable snapshot types.
        outputs: Full integrate-and-fire snapshots for output-role neurons.

    The mappings are detached and read-only. Later ticks cannot change a
    previously returned result.
    """

    tick: int
    states: Mapping[str, NeuronStepState]
    outputs: Mapping[str, StatefulStepState]
