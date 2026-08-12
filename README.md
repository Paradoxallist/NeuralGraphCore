# NeuralGraphCore

Current package version: **0.0.2** (release/tag form: `v0.0.2`).

```python
import neural_graph_core

print(neural_graph_core.__version__)  # 0.0.2
```

NeuralGraphCore is a small, deterministic, pure-Python simulation core for
stateful graph-based and recurrent neural networks. It provides integrate-and-
fire neurons, analog environment inputs, binary internal signaling, directed
weighted synapses, validated topology, and synchronous tick execution.

Read [NETWORK_SPEC.md](NETWORK_SPEC.md) for the exact network physics. That
document is authoritative for tick timing, signal propagation, potential
integration, thresholding, reset rules, and atomic commit behavior.

## Scope

The library contains:

- analog `InputNeuron` sources;
- binary integrate-and-fire `StatefulNeuron` nodes;
- sink-only integrate-and-fire `OutputNeuron` readouts;
- autonomous binary `PulsatingNeuron` sources;
- mutable weighted synapses with immutable endpoints;
- validated graph topology;
- synchronous `step`, sequential `run`, diagnostic snapshots, and reset.

It intentionally excludes learning, plasticity algorithms, DNA/genomes, growth,
visualization, serialization, environment adapters, experiment frameworks,
multiprocessing, training, and vector-valued state.

## Installation

NeuralGraphCore requires Python 3.11 or newer and has no runtime dependencies
outside the standard library.

```bash
python -m pip install -e /path/to/NeuralGraphCore
```

Run the regression suite with:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Basic network

```python
from neural_graph_core import (
    HardReset,
    InputNeuron,
    Network,
    NetworkRunner,
    OutputNeuron,
    StatefulNeuron,
)

network = Network()
sensor = network.add_neuron(InputNeuron(id="sensor"))
hidden = network.add_neuron(
    StatefulNeuron(
        id="hidden",
        threshold=1.0,
        retention=0.5,
        reset=HardReset(),
    )
)
output = network.add_neuron(
    OutputNeuron(id="action", threshold=1.0, retention=0.0)
)

network.connect(sensor, hidden, weight=1.0)
network.connect(hidden, output, weight=1.0)

runner = NetworkRunner(network)

tick_1 = runner.step({"sensor": 1.0})
assert tick_1.states["hidden"].spike == 1
assert tick_1.outputs["action"].spike == 0

tick_2 = runner.step({})
assert tick_2.outputs["action"].spike == 1
```

External input reaches a directly connected neuron during the current tick. A
new internal spike is transmitted only during the following tick.

## Integrate-and-fire behavior

`StatefulNeuron` and `OutputNeuron` use:

```text
candidate = previous_potential * retention + incoming_signal
spike = 1 if candidate >= threshold else 0
```

The default retention is `1.0`, so potential accumulates without implicit
decay. Set `retention < 1.0` explicitly when decay is desired.

When no spike occurs, `potential` becomes `candidate`. After a spike, the
configured reset rule selects the retained potential. A neuron emits at most
one binary spike per tick.

```python
neuron = StatefulNeuron(
    id="memory",
    threshold=1.0,
    retention=0.8,
    potential=0.25,
    spike=0,
)

print(neuron.potential)
print(neuron.spike)
print(neuron.output)  # float(neuron.spike)
```

Initial `potential` and `spike` are independent committed values. They are not
required to satisfy the threshold relation.

## Reset rules

Exactly one explicit reset rule is configured per stateful/output neuron:

```python
from neural_graph_core import (
    FixedResidualReset,
    HardReset,
    PercentageReset,
    SubtractiveReset,
)

HardReset()                       # potential = 0.0
SubtractiveReset()                # potential = candidate - threshold
FixedResidualReset(value=0.2)     # potential = 0.2
PercentageReset(fraction=0.25)    # potential = candidate * 0.25
```

`FixedResidualReset.value` must be finite. `PercentageReset.fraction` must be
between 0 and 1 inclusive.

## Signals and synapses

Every neuron has a read-only `output` used by synapses:

```text
contribution = source.output * synapse.weight
```

- `InputNeuron.output` is an analog float supplied by the environment.
- `StatefulNeuron.output` and `OutputNeuron.output` are binary `0.0` or `1.0`.
- `PulsatingNeuron.output` is a binary `0.0` or `1.0`.

`Neuron.output` means “the value transmitted through outgoing synapses.” It is
available on every neuron type and is unrelated to the sink-only class named
`OutputNeuron`. A stateful neuron's `potential` is internal and is never
transmitted directly.

Synapse endpoints never change because `(source_id, target_id)` is its network
key. Weight and enabled state can be edited between ticks:

```python
synapse = network.connect("sensor", "hidden", weight=0.5)
synapse.weight = -1.25
synapse.enabled = False
```

Weights may be any finite float. A disabled synapse contributes zero. Only one
synapse may exist for an ordered endpoint pair; a stateful self-loop is valid.

## Pulsating neuron

```python
from neural_graph_core import PulsatingNeuron

clock = PulsatingNeuron(
    id="clock",
    period_ticks=3,
    ticks_since_spike=0,
    spike=0,
)
```

The timer is local to the neuron and independent of `NetworkRunner.tick`.
`PulsatingNeuron` emits a binary spike when its timer reaches the period. It
accepts neither incoming synapses nor external input.

## StepResult diagnostics

`step()` returns immutable typed snapshots:

```python
result = runner.step({"sensor": 0.75})

print(result.states["sensor"].value)

hidden_state = result.states["hidden"]
print(hidden_state.incoming_signal)
print(hidden_state.candidate)
print(hidden_state.potential)
print(hidden_state.spike)

output_state = result.outputs["action"]
print(output_state.potential, output_state.spike)
```

The fields have distinct meanings:

- `candidate` is potential before threshold evaluation and reset;
- `spike` is the binary threshold-decision result;
- `potential` is the retained value after any reset.

For example, subtractive reset with candidate `1.4` and threshold `1.0`
produces `spike == 1` and `potential == 0.4`.

Snapshot types are `InputStepState`, `StatefulStepState`, and
`PulsatingStepState`. `StepResult.outputs` contains full stateful snapshots for
output-role neurons, not only their spike values.

## run and reset

```python
results = runner.run([
    {"sensor": 1.0},
    {"sensor": 0.5},
    None,
])

runner.reset()
```

`run(sequence)` is equivalent to successive `step` calls. Missing known input
values become `0.0`; unknown IDs are errors.

`reset()` restores the runner tick counter and each neuron's configured initial
dynamic state, including potential, spike, input value, and pulsating timer
phase. Initial state starts from constructor values but can be changed through
the explicit APIs described below. Reset does not change topology, weights,
enabled flags, thresholds, retention values, reset rules, or periods.

## Live mutation

Public mutation APIs are intended to be called between ticks. A successful
change is used by the next `step()` and never triggers an implicit reset.

Three concepts remain separate:

- **configuration** controls future calculations;
- **current dynamic state** is the state committed now;
- **initial state** is the target restored by `reset()`.

### Stateful and output neurons

```python
neuron.threshold = 1.5
neuron.retention = 0.8
neuron.reset_rule = PercentageReset(0.5)

neuron.set_state(potential=0.75, spike=1)
neuron.set_initial_state(potential=0.3, spike=0)
neuron.set_initial_state_from_current()
```

Configuration setters preserve both current and initial state. `set_state`
changes only current committed state and keeps `spike` synchronized with
`output`. `set_initial_state` changes only the future reset target.
`OutputNeuron` inherits this complete API.

### Pulsating neurons

Timer configuration is atomic because period and phase are related:

```python
clock.configure_timer(
    period_ticks=5,
    ticks_since_spike=2,
    initial_ticks_since_spike=1,
)

clock.set_state(spike=1, ticks_since_spike=3)
clock.set_initial_state(spike=0, ticks_since_spike=1)
clock.set_initial_state_from_current()
```

Both current and initial phases must be valid for the resulting period. An
invalid combination raises an error without changing anything; phases are
never silently clamped or reduced modulo the period.

### Editing a running network

```python
neuron = StatefulNeuron(id="a", threshold=1.0, retention=1.0)

runner.step(...)

neuron.threshold = 1.5
neuron.set_state(potential=0.7)

runner.step(...)

neuron.set_initial_state_from_current()
```

The same between-tick rule applies to mutable `Synapse.weight` and
`Synapse.enabled`. IDs, synapse endpoints, and `NetworkRunner.tick` remain
read-only.

## Graph API

```python
network.add_neuron(neuron)
network.remove_neuron(neuron_or_id)
network.get_neuron("hidden")

network.add_synapse(synapse)
network.connect(source_or_id, target_or_id, weight=1.0)
network.disconnect(source_or_id, target_or_id)
network.get_synapse(source_or_id, target_or_id)

network.incoming(neuron_or_id)
network.outgoing(neuron_or_id)
```

Neuron IDs are unique within a network. Removing a neuron automatically removes
all connected synapses. `InputNeuron` and `PulsatingNeuron` cannot receive
synapses; `OutputNeuron` cannot emit them.

## Public imports

The main API is available from the package root:

```python
from neural_graph_core import (
    FixedResidualReset,
    HardReset,
    InputNeuron,
    InputStepState,
    Network,
    NetworkRunner,
    Neuron,
    NeuronRole,
    NeuronStepState,
    OutputNeuron,
    PercentageReset,
    PulsatingNeuron,
    PulsatingStepState,
    ResetRule,
    StatefulNeuron,
    StatefulStepState,
    StepResult,
    SubtractiveReset,
    Synapse,
)
```

Before depending on timing or recurrence behavior, read
[NETWORK_SPEC.md](NETWORK_SPEC.md). Dependent projects should use
`NetworkRunner` rather than reimplementing tick semantics.
