# NeuralGraphCore

NeuralGraphCore is a small, pure-Python foundation for graph-based and
recurrent neural systems. It provides reusable neuron types, directed weighted
synapses, graph validation, and deterministic synchronous execution.

The library is intended to be shared by multiple projects so that each project
can focus on its own learning rules, growth logic, environment integration, and
experiments instead of rebuilding network mechanics.

For the exact tick-by-tick behavior, read [NETWORK_SPEC.md](NETWORK_SPEC.md).
That document is the source of truth for network semantics.

## Scope

NeuralGraphCore currently contains:

- input, hidden/stateful, output, and autonomous pulsating neurons;
- directed weighted synapses;
- a validated network container;
- synchronous single-step and multi-step execution;
- immutable step-result snapshots;
- reset support for dynamic neuron state and runner time.

NeuralGraphCore intentionally does not contain:

- training algorithms or optimizers;
- backpropagation or automatic differentiation;
- datasets, loss functions, or task-specific metrics;
- DNA, genome, mutation, crossover, or growth systems;
- visualization or user-interface code;
- environment-specific input/output adapters;
- experiment-specific behavior.

Those features belong in projects that depend on this library.

## Requirements

- Python 3.11 or newer
- no runtime dependencies outside the Python standard library

## Local installation

Install the repository in editable mode while developing both the library and
a dependent project:

```bash
python -m pip install -e /path/to/NeuralGraphCore
```

On Windows, for example:

```powershell
python -m pip install -e C:\Projects\NeuralGraphCore
```

After installation, changes made in the NeuralGraphCore source directory are
immediately visible to the dependent environment.

## Public API

The primary classes can be imported from the package root:

```python
from neural_graph_core import (
    InputNeuron,
    Network,
    NetworkRunner,
    Neuron,
    NeuronRole,
    OutputNeuron,
    PulsatingNeuron,
    StatefulNeuron,
    StepResult,
    Synapse,
)
```

### Neuron implementations

`InputNeuron`
: Receives a value from the external environment. It cannot have incoming
  synapses. A missing value on a step is treated as `0.0`.

`StatefulNeuron`
: General-purpose hidden neuron. It supports incoming links, outgoing links,
  recurrent loops, and self-loops. Its update is
  `activation(weighted_input + bias)`.

`OutputNeuron`
: Computes values like a `StatefulNeuron`, appears in `StepResult.outputs`, and
  cannot have outgoing synapses.

`PulsatingNeuron`
: Autonomous periodic source with a private local timer. It accepts neither
  synaptic nor external input.

## Creating and running a network

```python
from neural_graph_core import (
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
        bias=0.1,
        activation=lambda value: max(0.0, value),
    )
)
output = network.add_neuron(OutputNeuron(id="output"))

network.connect(sensor, hidden, weight=0.5)
network.connect(hidden, output, weight=1.0)

runner = NetworkRunner(network)
result = runner.step(inputs={"sensor": 1.0})

print(result.tick)
print(result.states)
print(result.outputs)
```

External input is visible to directly connected neurons during the current
tick. Newly calculated internal neuron states are transmitted only on the next
tick. Consequently, in the example above `hidden` reacts during the first
step, while `output` sees that new hidden state during the second step.

## Adding synapses

`connect` accepts either registered neuron instances or their identifiers:

```python
network.connect(sensor, hidden, weight=0.5)
network.connect("hidden", "output", weight=1.0)
```

A pre-built `Synapse` can also be added instead of calling `connect`:

```python
from neural_graph_core import Synapse

network.add_synapse(
    Synapse(
        source_id="sensor",
        target_id="hidden",
        weight=-0.25,
        enabled=True,
    )
)
```

Only one synapse may exist for an ordered `(source_id, target_id)` pair.
Positive and negative weights are both supported. A disabled synapse remains
in the topology but transmits no signal.

## Running a sequence

Pass one input mapping per tick to `run`:

```python
results = runner.run([
    {"sensor": 1.0},
    {"sensor": 0.5},
    {},
])

for result in results:
    print(result.tick, result.outputs)
```

`runner.run(sequence)` is equivalent to calling `runner.step(inputs)` for each
item in the same order. `None` and `{}` both represent a step on which every
known input defaults to zero.

## Pulsating neuron

```python
from neural_graph_core import Network, PulsatingNeuron

network = Network()
clock = network.add_neuron(
    PulsatingNeuron(
        id="clock",
        period_ticks=3,
        ticks_since_spike=0,
        spike_value=1.0,
        resting_value=0.0,
    )
)
```

The timer is local to `clock`; it does not depend on the runner's absolute tick
number. `ticks_since_spike` can set an initial phase. A pulse committed during
one tick is available to downstream neurons during the following tick.

## Resetting execution

```python
runner.reset()
```

Resetting:

- sets `runner.tick` back to `0`;
- restores each neuron's constructor-provided initial state;
- restores each `PulsatingNeuron` timer to its constructor-provided phase;
- preserves all neurons, synapses, weights, enabled flags, biases, activation
  functions, and other static parameters.

## Inspecting the graph

```python
neuron = network.get_neuron("hidden")
synapse = network.get_synapse("sensor", "hidden")

incoming = network.incoming("hidden")
outgoing = network.outgoing("hidden")

all_neurons = network.neurons
all_synapses = network.synapses
```

`network.neurons` and `network.synapses` are read-only mapping views. Structural
changes must go through `Network` methods so the graph invariants remain valid.

Removing a neuron automatically removes every synapse connected to it:

```python
network.remove_neuron("hidden")
```

## Error behavior

The library rejects, among other invalid operations:

- duplicate neuron identifiers in one network;
- duplicate synapse endpoint pairs;
- synapses referencing unknown neurons;
- incoming synapses targeting an `InputNeuron` or `PulsatingNeuron`;
- outgoing synapses originating from an `OutputNeuron`;
- external input for an unknown or non-input neuron;
- NaN and infinite state, input, bias, activation-result, or weight values.

## Behavioral specification

Read [NETWORK_SPEC.md](NETWORK_SPEC.md) before relying on timing, recurrence,
self-loops, atomicity, or reset behavior. Projects using NeuralGraphCore should
not reimplement or reinterpret those rules.
