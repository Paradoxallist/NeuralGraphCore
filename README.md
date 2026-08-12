# NeuralGraphCore

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
python -m unittest discover -s tests -v
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

`reset()` restores the runner tick counter and each neuron's constructor-
provided dynamic state, including potential, spike, input value, and pulsating
timer phase. It does not change topology, weights, enabled flags, thresholds,
retention values, reset rules, or periods.

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
