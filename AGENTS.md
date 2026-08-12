# NeuralGraphCore Contributor Guidance

NeuralGraphCore is a shared core library for graph-based and recurrent neural
projects.

Before changing the package:

1. Read `README.md` for installation and public API usage.
2. Read `NETWORK_SPEC.md` for the authoritative network and tick semantics.
3. Treat regression tests as executable definitions of fundamental invariants.

Keep project-specific concerns outside this repository. Do not add learning
algorithms, DNA or genome systems, growth rules, visualization, environment
logic, task adapters, or experimental behavior to the shared core unless their
general-purpose inclusion has been explicitly agreed.

Do not reimplement or reinterpret tick semantics in dependent projects. Use
`NetworkRunner` and the public API provided by this package.

If a proposed change modifies network semantics, discuss and update
`NETWORK_SPEC.md` and the corresponding regression tests before treating the
change as complete. Tests must document the intended behavior rather than
changing the behavior to make testing easier.

Keep public docstrings and repository documentation in English. Preserve the
separation between dynamic state and static topology/configuration, and keep
new public fields encapsulated behind read-only properties unless mutability is
an explicit part of the API design.
