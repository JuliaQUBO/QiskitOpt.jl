# QAOA Best Practices

This guide maps the QAOA workflow guidance from
[qopt-best-practices](https://github.com/qiskit-community/qopt-best-practices),
[qaoa_training_pipeline](https://github.com/qiskit-community/qaoa_training_pipeline),
and [arXiv:2606.05311](https://arxiv.org/abs/2606.05311) onto the
QiskitOpt.jl attributes that users should set explicitly.

The short version is: start locally, make the run reproducible, record where
the initial angles came from, and move to IBM hardware only after the local
configuration is stable.

## Recommended Workflow

1. Build and debug the JuMP/MOI model with the default local Aer backend.
2. Fix the QAOA depth, shot counts, simulator seed, transpiler seed, and initial
   parameters before comparing runs.
3. Validate several initial-parameter strategies locally before spending IBM
   hardware time.
4. If hardware topology matters, do the hardware-aware transpilation and mapping
   work in an upstream workflow, then pass only the selected backend, seeds, and
   angles into QiskitOpt.jl.
5. Treat hardware execution as final validation and sampling, not as the first
   place to tune angles or train top-tail sample objectives.

## QiskitOpt.jl Attributes To Set

| Purpose | QAOA attribute or helper |
| --- | --- |
| QAOA depth | `QiskitOpt.QAOA.NumberOfLayers()` |
| Optimizer shots | `QiskitOpt.QAOA.NumberOfReads()` |
| Final sampling shots | `QiskitOpt.QUBODrivers.FinalNumberOfReads()` |
| Optimizer iterations | `QiskitOpt.QAOA.MaximumIterations()` |
| Initial angle values | `QiskitOpt.QAOA.InitialParameters()` |
| Initial angle provenance | `QiskitOpt.QAOA.InitialParameterSource()` |
| Initial metadata recording | `QiskitOpt.QAOA.RecordInitialParameters()` |
| Standard benchmark seed | `QiskitOpt.QUBODrivers.RandomSeed()` |
| Local Aer method | `QiskitOpt.QAOA.AerBackendMethod()` |
| Local Aer precision | `QiskitOpt.QAOA.AerPrecision()` |
| Local Aer simulator seed | `QiskitOpt.QAOA.AerSeedSimulator()` |
| Qiskit transpiler seed | `QiskitOpt.QAOA.TranspilerSeed()` |
| Aer MPS controls | `QiskitOpt.QAOA.AerMPSOmpThreads()`, `QiskitOpt.QAOA.AerMPSTruncationThreshold()`, `QiskitOpt.QAOA.AerMPSMaxBondDimension()`, `QiskitOpt.QAOA.AerMPSSampleMeasureAlgorithm()` |
| Named IBM backend | `QiskitOpt.QAOA.IBMBackend()` |
| Local emulation of an IBM backend | `QiskitOpt.QAOA.IBMBackend()` plus `QiskitOpt.QAOA.IsLocal()` |
| IBM Runtime account selection | `QiskitOpt.QAOA.Channel()` and `QiskitOpt.QAOA.Instance()` |
| Custom local backend factory | `QiskitOpt.QAOA.IBMFakeBackend()` |
| Custom QAOA pass-manager factory | `QiskitOpt.QAOA.PassManagerFactory()` |

QiskitOpt.jl defaults to local Aer execution when `IBMBackend()` is unset. The
default Aer method is `matrix_product_state`, so a first QAOA run does not need
an IBM token.

```julia
using JuMP
using QiskitOpt

model = Model(QiskitOpt.QAOA.Optimizer)

set_attribute(model, QiskitOpt.QAOA.NumberOfLayers(), 2)
set_attribute(model, QiskitOpt.QAOA.NumberOfReads(), 1024)
set_attribute(model, QiskitOpt.QUBODrivers.FinalNumberOfReads(), 8192)
set_attribute(model, QiskitOpt.QAOA.MaximumIterations(), 100)

set_attribute(model, QiskitOpt.QUBODrivers.RandomSeed(), 73001)
set_attribute(model, QiskitOpt.QAOA.AerBackendMethod(), "matrix_product_state")
set_attribute(model, QiskitOpt.QAOA.AerSeedSimulator(), 73001)
set_attribute(model, QiskitOpt.QAOA.TranspilerSeed(), 73001)
```

`QUBODrivers.RandomSeed()` is the benchmark-facing seed. If
`AerSeedSimulator()` or `TranspilerSeed()` is unset, QiskitOpt derives that
Qiskit-specific seed deterministically from the standard seed. Explicit
QiskitOpt seed attributes still take precedence.

## Initial Parameters

QiskitOpt.jl follows Qiskit's QAOA parameter order: all beta angles followed by
all gamma angles. Use `QiskitOpt.QAOA.parameter_names(...)` before setting
initial values when you want to audit the order.

The available choices are:

- Default zero vector. Leave `InitialParameters()` unset. Metadata records the
  source as `default_zero`.
- Seeded random starts. Generate values with
  `QiskitOpt.QAOA.random_initial_parameters(number_of_layers=p, seed=seed)` and
  store a matching `InitialParameterSource()`.
- Fixed Wurtz-Lykov angles. Use
  `QiskitOpt.QAOA.fixed_angle_initial_parameters(...)` for the built-in
  3-regular tree table, and record `QiskitOpt.QAOA.FIXED_ANGLE_SOURCE`.
- Schedule-based starts. Use
  `QiskitOpt.QAOA.linear_ramp_initial_parameters(...)`,
  `QiskitOpt.QAOA.tqa_initial_parameters(...)`, or
  `QiskitOpt.QAOA.interpolated_initial_parameters(...)`, then record a
  descriptive `InitialParameterSource()`.

```julia
p = 2
seed = 73001

random_start = QiskitOpt.QAOA.random_initial_parameters(
    number_of_layers=p,
    seed=seed,
)
set_attribute(model, QiskitOpt.QAOA.NumberOfLayers(), p)
set_attribute(model, QiskitOpt.QAOA.InitialParameters(), random_start)
set_attribute(model, QiskitOpt.QAOA.InitialParameterSource(), "random_seed_73001")
```

```julia
fixed_start = QiskitOpt.QAOA.fixed_angle_initial_parameters(
    number_of_layers=2,
    gamma_sign=-1,
)
set_attribute(model, QiskitOpt.QAOA.InitialParameters(), fixed_start)
set_attribute(
    model,
    QiskitOpt.QAOA.InitialParameterSource(),
    QiskitOpt.QAOA.FIXED_ANGLE_SOURCE,
)
```

```julia
schedule_start = QiskitOpt.QAOA.linear_ramp_initial_parameters(
    number_of_layers=p,
    delta_beta=0.35,
    delta_gamma=0.75,
    gamma_sign=-1,
)
set_attribute(model, QiskitOpt.QAOA.NumberOfLayers(), p)
set_attribute(model, QiskitOpt.QAOA.InitialParameters(), schedule_start)
set_attribute(model, QiskitOpt.QAOA.InitialParameterSource(), "linear_ramp_arxiv_2606_05311")
```

The returned `SampleSet` metadata records `metadata["initial_parameters"]` with
the source, parameter names, and values when `RecordInitialParameters()` is
true. Keep that enabled for experiments where angles are compared across
simulators, seeds, or hardware backends.

## Exporting Fixed-Parameter Circuits

Use `QiskitOpt.QAOA.fixed_parameter_circuit` after parameter search is complete
and the next step is submitting or inspecting a fixed QAOA circuit. It uses the
same QUBO-to-Qiskit cost-operator path and Qiskit `QAOAAnsatz` parameter order
as `QAOA.Optimizer`, but it does not run the optimizer loop, select a backend,
invoke Runtime primitives, transpile, or sample.

```julia
circuit, metadata = QiskitOpt.QAOA.fixed_parameter_circuit(
    JuMP.unsafe_backend(model);
    parameters=trained_parameters,
    reps=p,
    parameter_order=:beta_then_gamma,
    measure=true,
)
```

The metadata is intended to travel with exported circuits. It records variable
order, Qiskit count-key order, QAOA parameter order, objective scale/offset,
objective sign convention, and backend-independent circuit size information.
Parameter values stored in metadata always align with Qiskit parameter names,
regardless of the caller's input order. Store caller-specific angle provenance,
such as source path, training target, seed, or expected probabilities, alongside
the returned metadata when you need it. Use
`QiskitOpt.QAOA.count_key_bits(key)` to convert Qiskit count keys back to
`[x1, x2, ...]` before scoring with `QUBOTools.value`.

## IBM Runtime Handoff For Fixed Circuits

Use `QiskitOpt.QAOA.ibm_runtime_handoff` when QiskitOpt has built a measured,
fixed-parameter QAOA circuit and another workflow is responsible for IBM
Runtime account configuration and job monitoring. The helper is dry-runable by
default, so it can be exercised in CI without credentials:

```julia
handoff = QiskitOpt.QAOA.ibm_runtime_handoff(
    circuit;
    fixed_metadata=metadata,
    backend="ibm_fez",
    shots=8192,
    transpiler_seed=73001,
)
```

The dry-run metadata records the intended backend, shot budget, transpiler
seed, fixed QAOA parameter metadata, package versions, and count-key scoring
assumptions. It records whether an instance selector is configured, but it does
not record token values, instance/CRN values, account file paths, or other
credential material.

Live submission requires an explicit opt-in:

```julia
live_handoff = QiskitOpt.QAOA.ibm_runtime_handoff(
    circuit;
    fixed_metadata=metadata,
    backend="ibm_fez",
    shots=8192,
    transpiler_seed=73001,
    dry_run=false,
)

job = live_handoff.job
job_id = live_handoff.metadata["runtime_handoff"]["job"]["id"]
```

The live path resolves the named backend with `QiskitRuntimeService`,
transpiles the measured circuit with the requested seed and optimization level,
then submits it through `qiskit_ibm_runtime.SamplerV2`. Credentials should come
from normal Qiskit Runtime account storage or environment variables such as
`QISKIT_IBM_TOKEN`; QiskitOpt does not persist them. Set
`QISKIT_IBM_INSTANCE` or pass `instance=...` when Runtime cannot auto-resolve an
account instance. In practice this may be required even when a token is present;
the Runtime failure can look like:

```text
IBMInputValueError: No matching instances found for the following filters: .
```

Treat returned counts as Qiskit count keys. Convert keys with
`QiskitOpt.QAOA.count_key_bits(key)`, then score them with the linear,
quadratic, scale, and offset values from the original QUBO.

## Moving From Aer To IBM Hardware

Use three stages when hardware execution is the goal:

1. Local Aer without `IBMBackend()` for fast model and parameter checks.
2. Local Aer emulation of a named IBM backend by setting `IBMBackend()` and
   `IsLocal()` to `true`.
3. Real IBM hardware by setting the same `IBMBackend()` and `IsLocal()` to
   `false`.

```julia
# Stage 2: emulate a named IBM backend locally through Aer.
set_attribute(model, QiskitOpt.QAOA.IBMBackend(), "ibm_fez")
set_attribute(model, QiskitOpt.QAOA.IsLocal(), true)

# Stage 3: submit to the selected IBM backend.
set_attribute(model, QiskitOpt.QAOA.IsLocal(), false)
```

For hardware runs, configure IBM Runtime credentials through the documented
environment variables or attributes:

- `QISKIT_IBM_TOKEN`
- `QISKIT_IBM_INSTANCE`, or `QiskitOpt.QAOA.Instance()`
- `QISKIT_IBM_CHANNEL`, or `QiskitOpt.QAOA.Channel()`

Keep `AerSeedSimulator()` and `TranspilerSeed()` set while transitioning through
these stages so metadata can explain differences between local and hardware
runs.

## Hardware-Aware Pass Managers

For advanced QAOA workflows, `QAOA.PassManagerFactory()` lets you override only
the QAOA transpilation path while keeping backend selection, parameter
optimization, sampling, and metadata inside QiskitOpt.jl. This is intended for
pass-manager recipes developed with upstream tooling such as
`qiskit-community/qopt-best-practices`; QiskitOpt.jl does not add those tools as
dependencies.

```julia
function hardware_aware_qaoa_pass_manager(
    backend;
    optimization_level=3,
    seed_transpiler=nothing,
)
    return QiskitOpt.preset_pass_manager(
        backend;
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
end

set_attribute(
    model,
    QiskitOpt.QAOA.PassManagerFactory(),
    hardware_aware_qaoa_pass_manager,
)
```

The factory receives the selected backend and may accept `optimization_level`
and `seed_transpiler` keywords. `TranspilerSeed()` is forwarded as
`seed_transpiler` when the callable supports it. QiskitOpt.jl uses the returned
pass manager for both the optimization ansatz and the final measured circuit.
Result metadata records whether QAOA used the default preset pass manager or a
custom factory. VQE does not expose this QAOA-specific hook.

## Upstream Boundary

QiskitOpt.jl should remain the JuMP/MOI adapter that sends a QUBO to Qiskit,
selects the execution backend, records seeds and angle metadata, and returns
samples. Broader QAOA research workflows belong upstream:

| Belongs in QiskitOpt.jl | Belongs upstream |
| --- | --- |
| `IBMBackend()`, `IsLocal()`, `IBMFakeBackend()`, `PassManagerFactory()`, Aer attributes, and seeds | Hardware-aware transpilation recipe development from `qopt-best-practices` |
| `InitialParameters()` and `InitialParameterSource()` | MPS, Pauli propagation, parameter-transfer, and schedule-training pipelines |
| Result metadata for backend configuration, seeds, and starting angles | SAT mapping, SWAP strategies, qubit selection, and CVaR/top-tail sample-objective training loops |

The adapter boundary is intentional. If a workflow trains angles with
`qaoa_training_pipeline`, Pauli propagation, MPS, fixed-angle transfer, or a
hardware-aware mapping strategy, QiskitOpt.jl should receive the final QAOA
depth, ordered angle vector, backend choice, and reproducibility metadata. It
should not reimplement those generic QAOA capabilities inside the optimizer.

CVaR and top-tail sample objectives follow the same boundary. The current
`QAOA.Optimizer` trains parameters with an expected-energy estimator objective,
then samples the optimized circuit. A CVaR/top-tail training loop would need to
sample inside the parameter optimizer, choose and document the tail fraction,
and record training metadata. That is generic QAOA training infrastructure, so
QiskitOpt.jl documents the recommendation but does not add a package-local
objective switch or post-processing helper.

The paper behind arXiv:2606.05311 makes the same operational point: approximate
energy evaluators such as MPS and Pauli propagation can train utility-scale
angles, but the selected angles still need careful local and hardware
validation. Dense instances, topology-sensitive MPS runs, and hardware-near
circuits require workflow-level benchmarking rather than package-specific
shortcuts.
