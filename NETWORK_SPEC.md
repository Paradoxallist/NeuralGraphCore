# NeuralGraphCore Network Semantics

This document is the source of truth for the current behavior of
NeuralGraphCore. It answers the question:

> What exactly happens on the next tick?

The specification describes the implemented model. Features not listed here,
including threshold firing, membrane-potential retention, learning, and
multiple spike-reset policies, are not currently part of NeuralGraphCore.

## Values and time

Every neuron exposes one committed `float` state. Network execution advances in
discrete synchronous ticks through `NetworkRunner.step()`.

The runner has a completed-tick counter:

- a new runner starts at `tick == 0`;
- the first successful `step()` produces `StepResult.tick == 1`;
- the counter advances only after a successful commit;
- `reset()` returns the counter to `0`.

## Neuron roles

The built-in `NeuronRole` type contains:

```python
Literal["input", "hidden", "output", "pulsating"]
```

### InputNeuron

An `InputNeuron` is the strict `environment -> network` boundary.

- It receives its value only from `NetworkRunner.step(inputs=...)`.
- It cannot be the target of a synapse.
- It may be the source of a synapse.
- A missing known input value is treated as `0.0`.
- A supplied unknown ID raises `KeyError`.
- A supplied ID belonging to a non-input neuron raises `ValueError`.
- Its current external value is available to downstream neurons during the
  same tick.
- Its committed state after the tick equals the supplied value or default zero.

### StatefulNeuron

A `StatefulNeuron` is the general-purpose hidden and recurrent node.

- It may have incoming and outgoing synapses.
- It may participate in recurrent loops.
- It may have a self-loop.
- It does not accept external input.
- It calculates its next state as:

```text
total = weighted_input + bias
next_state = activation(total)
```

`activation` is a callable receiving one `float` and returning a value
convertible to a finite `float`. The default activation is the identity
function.

A stateful neuron's previous state does not implicitly enter its calculation.
Memory is expressed structurally through a self-loop or another recurrent path.

### OutputNeuron

An `OutputNeuron` is the strict `network -> environment` boundary.

- It inherits the calculation performed by `StatefulNeuron`.
- It may be the target of synapses.
- It cannot be the source of a synapse.
- Its committed state appears in both `StepResult.states` and
  `StepResult.outputs`.

This sink-only rule deliberately separates external readout from recurrent
internal nodes. Use a `StatefulNeuron` when an internally propagated value is
also required.

### PulsatingNeuron

A `PulsatingNeuron` is an autonomous periodic source.

- It cannot be the target of a synapse.
- It does not accept external input.
- It may be the source of synapses.
- It uses its own `ticks_since_spike` timer, not `NetworkRunner.tick`.
- `period_ticks` must be a positive integer.
- The timer phase is local and can be selected at construction time.

For each update:

```text
next_ticks = ticks_since_spike + 1

if next_ticks >= period_ticks:
    next_state = spike_value
    next_ticks = 0
else:
    next_state = resting_value
```

Both the new public state and new timer are committed together. A pulse newly
created on tick N becomes an internal source signal on tick N+1.

## Synapses

A `Synapse` is a directed connection with:

- `source_id`;
- `target_id`;
- a finite `weight`;
- an `enabled` flag.

For an enabled synapse, its contribution is:

```text
source_signal * weight
```

Positive weights increase the target's aggregated input. Negative weights
decrease it and may make the aggregate negative. A disabled synapse contributes
exactly zero and otherwise remains part of the topology.

Synapses are immutable. Their endpoint pair is the network key, and at most one
synapse may exist for an ordered `(source_id, target_id)` pair.

A self-loop is an ordinary synapse whose source and target IDs are equal. It is
valid only if that neuron permits both incoming and outgoing connections. Among
the built-in implementations, this means a `StatefulNeuron` can have a
self-loop, while input, output, and pulsating neurons cannot.

## Network structure

`Network` owns neuron and synapse registration.

- A neuron ID is unique within one `Network` instance.
- The same ID may appear in a different network.
- Both endpoints must be registered before a synapse is added.
- Passing a neuron object to `connect` requires that exact object to be
  registered, not merely another object with the same ID.
- Removing a neuron automatically removes every connected synapse.
- `neurons` and `synapses` expose read-only mapping views.

Structural changes are made between calls to `step()`. The current
implementation executes `step()` synchronously and does not provide mutation
operations during a partially evaluated tick.

## Tick semantics

`NetworkRunner.step(inputs)` performs the following logical phases.

### 1. Copy and validate external inputs

The supplied mapping is copied. Every supplied ID must identify an existing
`InputNeuron`. Missing registered inputs receive `0.0`.

No neuron state has changed at this point.

### 2. Prepare current-tick input states

An update is prepared for every `InputNeuron`. Its prepared value is also used
as that input neuron's source signal during this tick.

No update has been committed yet.

### 3. Aggregate enabled synaptic signals

Each enabled synapse contributes `source_signal * weight` to its target.

The selected source signal depends on the source role:

- `InputNeuron`: the external value prepared for the current tick;
- every other neuron: its previously committed `state`.

This is the central timing rule of the library:

```text
external input transmission = current tick value
internal neuron transmission = previous committed state
```

### 4. Prepare all non-input updates

Every non-input neuron calculates its next state from its complete aggregated
input. Calculation order cannot affect the result because no newly calculated
state is visible during this phase.

### 5. Commit all updates

Only after every update has been prepared successfully are the updates applied
to the neurons. The runner then increments its tick counter and constructs a
`StepResult` snapshot.

## One-synapse-per-tick propagation

Consider:

```text
Input --1.0--> A --1.0--> B
```

where A and B are linear `StatefulNeuron` instances initialized to zero.

| Completed tick | External Input | A committed state | B committed state |
|---:|---:|---:|---:|
| 0 | - | 0.0 | 0.0 |
| 1 | 1.0 | 1.0 | 0.0 |
| 2 | 0.0 | 0.0 | 1.0 |
| 3 | 0.0 | 0.0 | 0.0 |

Input reaches A on tick 1. A's newly calculated state does not reach B until
tick 2. Internal activity therefore crosses at most one synapse per tick.

## Self-loop example

Consider a linear neuron A with initial state `1.0`, zero bias, and a self-loop
of weight `0.5`:

```text
     0.5
    +---+
    |   v
    +-- A
```

| Completed tick | Previous A | Self-loop input | New A |
|---:|---:|---:|---:|
| 0 | - | - | 1.0 |
| 1 | 1.0 | 0.5 | 0.5 |
| 2 | 0.5 | 0.25 | 0.25 |
| 3 | 0.25 | 0.125 | 0.125 |

The value calculated for A never feeds back into the calculation of the same
tick. Only the previously committed state crosses the self-loop.

## Recurrent-loop example

Consider two linear stateful neurons with A initialized to `1.0`, B initialized
to `0.0`, and unit-weight connections in both directions:

```text
A --1.0--> B
A <--1.0-- B
```

| Completed tick | A | B |
|---:|---:|
| 0 | 1.0 | 0.0 |
| 1 | 0.0 | 1.0 |
| 2 | 1.0 | 0.0 |
| 3 | 0.0 | 1.0 |
| 4 | 1.0 | 0.0 |

Both next states are calculated from the same previous committed snapshot, so
the alternating sequence is independent of neuron insertion order.

## Atomicity

The runner prepares all neuron updates before applying any of them. If input
validation, signal aggregation, or any neuron's `prepare_update` fails:

- no prepared state is committed;
- the runner tick counter does not advance;
- the network remains in its state from before `step()`.

Built-in neuron implementations validate their complete prepared update before
commit. Custom neuron implementations must preserve the contract that
`prepare_update` performs all potentially failing calculation and validation,
while `apply_update` only commits an already valid update.

## StepResult

After a successful tick, `step()` returns a frozen `StepResult` containing:

- `tick`: completed tick number;
- `states`: every neuron state indexed by ID;
- `outputs`: only states whose neuron role is `output`.

The mappings are detached, read-only snapshots. Later steps and structural
changes do not modify earlier results.

## run()

For an input sequence `sequence`:

```python
results = runner.run(sequence)
```

is equivalent to:

```python
results = tuple(runner.step(inputs) for inputs in sequence)
```

Execution stops immediately if one step raises an exception. Earlier successful
steps remain committed; the failing step itself follows the atomicity rule.

## Reset semantics

`NetworkRunner.reset()` resets dynamic execution state:

- the runner tick becomes `0`;
- every neuron state returns to the initial `state` supplied to its constructor;
- every `PulsatingNeuron` timer returns to its initial `ticks_since_spike`.

Reset does not change static structure or configuration:

- neurons are not added or removed;
- synapses are not added or removed;
- synapse weights and enabled flags do not change;
- neuron IDs, roles, biases, activations, periods, spike values, and resting
  values do not change.

Reset restores constructor-provided initial values, which are not necessarily
zero.
