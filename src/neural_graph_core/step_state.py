"""Immutable per-neuron snapshots produced by a completed network tick."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class InputStepState:
    """External value committed by an ``InputNeuron`` during a tick."""

    value: float


@dataclass(frozen=True, slots=True)
class StatefulStepState:
    """Complete integrate-and-fire calculation for one completed tick.

    Attributes:
        potential: Potential retained after the threshold decision and reset.
        spike: Binary output committed for the tick.
        incoming_signal: Sum of weighted source signals used in the calculation.
        candidate: Potential before the threshold decision and reset.
    """

    potential: float
    spike: int
    incoming_signal: float
    candidate: float


@dataclass(frozen=True, slots=True)
class PulsatingStepState:
    """Binary pulse and local timer committed by a ``PulsatingNeuron``."""

    spike: int
    ticks_since_spike: int


NeuronStepState: TypeAlias = (
    InputStepState | StatefulStepState | PulsatingStepState
)
