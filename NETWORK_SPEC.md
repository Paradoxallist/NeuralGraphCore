# NeuralGraphCore Network Semantics

This document is the authoritative specification of NeuralGraphCore network
physics. It answers:

> What exactly happens on the next tick?

## State and signals

Each neuron exposes a read-only `output: float`, which is the only value a
synapse transmits.

`Neuron.output` means the value transmitted through outgoing synapses. The
property exists on every neuron type and is not synonymous with the
`OutputNeuron` class. `OutputNeuron` names a sink-only network role; `output`
names the common transmission value.

| Neuron | Internal/public state | Synaptic output |
|---|---|---|
| `InputNeuron` | analog `value: float` | the analog value |
| `StatefulNeuron` | `potential: float`, `spike: int` | binary spike as `0.0` or `1.0` |
| `OutputNeuron` | same as `StatefulNeuron` | binary spike; outgoing links are forbidden |
| `PulsatingNeuron` | local timer and `spike: int` | binary spike as `0.0` or `1.0` |

Only input neurons are analog signal sources. Every built-in internal source is
binary.

## StatefulNeuron

`StatefulNeuron` is a discrete integrate-and-fire node. It supports incoming
and outgoing synapses, recurrent loops, and self-loops.

Its configuration is:

- positive finite `threshold`;
- finite `retention` in `[0, 1]`;
- exactly one `ResetRule`;
- initial finite `potential`;
- initial binary integer `spike` in `{0, 1}`.

The default retention is `1.0`. This is the neutral accumulation behavior:
potential does not decay unless the user explicitly selects `retention < 1.0`.

Initial potential and spike are independent committed values. An initial spike
of 1 does not require the initial potential to equal or exceed the threshold.

For one tick:

```text
incoming_signal = sum(source_output * synapse.weight)
candidate = previous_potential * retention + incoming_signal
```

If `candidate < threshold`:

```text
next_spike = 0
next_potential = candidate
```

If `candidate >= threshold`:

```text
next_spike = 1
next_potential = reset_rule(candidate, threshold)
```

At most one spike occurs per tick. A reset may leave potential at or above the
threshold, but that residual can produce another spike only on a later tick.
Potential is private neuron state and is never transmitted through a synapse.

## Reset rules

Four immutable reset configurations are built in.

### HardReset

```text
next_potential = 0.0
```

### SubtractiveReset

```text
next_potential = candidate - threshold
```

### FixedResidualReset

```text
next_potential = configured finite value
```

### PercentageReset

```text
next_potential = candidate * fraction
```

`fraction` is finite and belongs to `[0, 1]`. A neuron owns exactly one reset-
rule object; reset modes are not combined.

## InputNeuron

`InputNeuron` is a strict analog `environment -> network` source.

- Values arrive only through `NetworkRunner.step(inputs=...)`.
- It cannot receive synapses and may emit them.
- A missing registered input value becomes `0.0`.
- An unknown input ID or a non-input ID raises an error.
- A finite float is accepted; input is not restricted to binary values.
- Its current-tick value is immediately available to downstream neurons.

For input `0.5` and synaptic weight `0.8`, the current-tick contribution is
`0.4`.

## OutputNeuron

`OutputNeuron` inherits the complete integrate-and-fire model from
`StatefulNeuron`, including potential, threshold, retention, reset, and binary
spike. It may receive synapses, cannot emit them, and its full
`StatefulStepState` appears in `StepResult.outputs`. It is not a linear readout.

## PulsatingNeuron

`PulsatingNeuron` is an autonomous binary source with a local timer.

- It cannot receive synapses or external input and may emit synapses.
- It does not consult the absolute runner tick.
- `period_ticks` is a positive integer.
- `ticks_since_spike` selects a local initial phase.
- `spike` is always integer 0 or 1.

For each update:

```text
next_ticks = ticks_since_spike + 1

if next_ticks >= period_ticks:
    next_spike = 1
    next_ticks = 0
else:
    next_spike = 0
```

A spike generated on tick N is visible to downstream neurons on tick N+1.

## Synapse

A synapse is a directed connection. Its contribution when enabled is:

```text
source.output * weight
```

- `source_id` and `target_id` are immutable.
- Their ordered pair uniquely identifies the synapse within one network.
- `weight` is mutable between ticks and may be any finite float.
- `enabled` is mutable between ticks and must be a boolean.
- A disabled synapse contributes exactly zero.
- Positive and negative weights are supported.
- A self-loop is an ordinary synapse.
- Parallel synapses for the same ordered pair are forbidden.

## Live mutation semantics

Public mutation operations are intended to run between ticks. A successful
change affects the next tick and never implicitly calls `reset()` on a neuron
or runner.

The API distinguishes:

- **configuration**, which controls future calculations;
- **current dynamic state**, which is committed now;
- **initial/reset state**, which is restored by `reset()`.

### StatefulNeuron and OutputNeuron

The validated configuration properties `threshold`, `retention`, and
`reset_rule` are mutable. Changing one preserves current potential/spike and
the initial reset target.

`set_state(potential=..., spike=...)` atomically changes only current state.
Either argument may be omitted. Potential must be finite; spike must be integer
0 or 1. Changing spike also sets `output = float(spike)` in the same operation.
Potential and spike remain independent.

`set_initial_state(potential=..., spike=...)` atomically changes only the reset
target. `set_initial_state_from_current()` copies current potential and spike
to that target. Neither operation implicitly resets current state.

`OutputNeuron` inherits these APIs without changing its sink-only topology
rules.

### PulsatingNeuron

`configure_timer(period_ticks=..., ticks_since_spike=...,
initial_ticks_since_spike=...)` atomically validates the resulting period,
current phase, and initial phase before assigning any of them. Omitted values
are preserved. Both phases must satisfy:

```text
0 <= phase < resulting period_ticks
```

An incompatible period change raises an error and leaves all three values
unchanged. No hidden clamp or modulo is performed.

`set_state(spike=..., ticks_since_spike=...)` changes only current state and
synchronizes `output` with spike. `set_initial_state(...)` changes only the
reset target. `set_initial_state_from_current()` copies the current binary
spike and timer phase to the reset target.

### Immutable fields

Neuron IDs, synapse endpoints, and `NetworkRunner.tick` remain read-only.
Changing a synapse endpoint continues to mean disconnecting the old synapse and
connecting a new one.

## Synchronous tick phases

`NetworkRunner.step(inputs)` follows these phases.

### 1. Snapshot topology and validate inputs

The runner copies the external input mapping and snapshots current neuron and
synapse iteration collections. Every supplied ID must belong to a registered
input neuron. Missing input values default to zero.

### 2. Prepare current-tick input updates

Every `InputNeuron` prepares its analog value. That prepared value is used as
the source signal during this same tick. Nothing is committed yet.

### 3. Aggregate synaptic contributions

For each enabled synapse, the runner chooses:

- the prepared current-tick value for an `InputNeuron` source;
- the previous committed binary `output` for any internal source.

It multiplies that signal by the current synaptic weight and adds the result to
the target's incoming signal.

```text
external input transmission = current tick analog value
internal transmission = previous committed binary spike
```

### 4. Prepare every internal update

Every stateful/output neuron calculates incoming signal, candidate, next spike,
and next potential. Every pulsating neuron calculates its next timer and spike.
No neuron is mutated during preparation.

### 5. Commit

Only after all preparations succeed are all updates committed. The runner then
increments its completed-tick counter and returns a `StepResult`. Neuron
insertion and traversal order cannot affect the result because all internal
signals come from the same previous committed state.

## Propagation timing example

For unit weights, threshold 1, retention 0, and hard reset:

```text
Input -> A -> B
```

| Completed tick | Input | A spike | B spike |
|---:|---:|---:|---:|
| 0 | - | 0 | 0 |
| 1 | 1.0 | 1 | 0 |
| 2 | 0.0 | 0 | 1 |
| 3 | 0.0 | 0 | 0 |

Current external input reaches A immediately. A's newly generated spike crosses
the A-to-B synapse only on the next tick.

## Retention example

With threshold 1, retention 0.5, no incoming signal, and initial potential 0.8:

| Completed tick | Candidate | Spike | Retained potential |
|---:|---:|---:|---:|
| 1 | 0.4 | 0 | 0.4 |
| 2 | 0.2 | 0 | 0.2 |
| 3 | 0.1 | 0 | 0.1 |

## Self-loop

A self-loop uses the previous committed spike. A spike newly generated by A on
tick N cannot affect A's candidate on tick N; it becomes a self-loop source
only on tick N+1.

## Recurrent loop

For two hard-reset neurons with threshold 1, retention 0, unit bidirectional
weights, and initial committed spikes A=1 and B=0:

| Completed tick | A spike | B spike |
|---:|---:|---:|
| 0 | 1 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |

## Atomicity contract

All calculations and validation belong in `prepare_update`. The contract is:

> If `prepare_update()` successfully returns its valid `NeuronUpdate`, the
> corresponding `apply_update()` must not unexpectedly fail.

Consequently, preparation performs all reasonable validation of future state,
while application performs only simple assignments of already validated data.
The library does not implement a complex transactional rollback system.

If input validation, aggregation, or any preparation fails, no neuron update is
committed, the runner tick does not advance, and the network retains its pre-
step dynamic state. Custom neuron implementations must follow this contract.

## StepResult

After commit, the runner returns detached read-only mappings of immutable
snapshots:

- `InputStepState(value)`;
- `StatefulStepState(potential, spike, incoming_signal, candidate)`;
- `PulsatingStepState(spike, ticks_since_spike)`.

`StepResult.states` contains every neuron. `StepResult.outputs` contains the
full `StatefulStepState` for each output-role neuron. Later ticks cannot change
an earlier result.

For `StatefulStepState`:

- `candidate` is potential before threshold evaluation and reset;
- `spike` is the result of the threshold decision;
- `potential` is the retained potential after any reset.

With subtractive reset, candidate `1.4`, and threshold `1.0`, the resulting
snapshot has spike `1` and potential `0.4`.

## run

`runner.run(sequence)` is exactly successive `runner.step(inputs)` calls for
the same sequence. Earlier successful ticks remain committed if a later item
fails; the failing tick itself is atomic.

## Reset

`NetworkRunner.reset()` restores configured initial dynamic state:

- runner tick becomes 0;
- input values return to their constructor-provided initial values;
- stateful/output potential and spike return to their current initial targets;
- pulsating spike and local timer return to their current initial targets.

It preserves neuron identities, topology, current synapse weights and enabled
flags, thresholds, retention values, reset rules, and periods. Constructor-
configured initial dynamic state need not be zero.

## Network invariants

- Neuron IDs are unique within one network.
- Synapse endpoint pairs are unique within one network.
- Both endpoints must exist before connection.
- Input and pulsating neurons cannot receive synapses.
- Output neurons cannot emit synapses.
- Removing a neuron removes all connected synapses.
- Structural changes, neuron mutations, and synapse edits occur between ticks.
