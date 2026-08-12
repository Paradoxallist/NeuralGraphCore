"""Immutable public result of one completed network tick."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepResult:
    """Snapshot returned after a synchronous network update.

    Args:
        tick: Number of completed ticks. The first call to ``step`` returns 1.
        states: State of every neuron after the update, indexed by neuron ID.
        outputs: State of every neuron whose role is ``output``.

    Both mappings are read-only snapshots. Later network updates do not change
    a previously returned result.

    Example:
        Read a network output after a step::

            result = runner.step(inputs={"sensor": 1.0})
            action = result.outputs["action"]
    """

    tick: int
    states: Mapping[str, float]
    outputs: Mapping[str, float]
