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
| Local Aer method | `QiskitOpt.QAOA.AerBackendMethod()` |
| Local Aer precision | `QiskitOpt.QAOA.AerPrecision()` |
| Local Aer simulator seed | `QiskitOpt.QAOA.AerSeedSimulator()` |
| Qiskit transpiler seed | `QiskitOpt.QAOA.TranspilerSeed()` |
| Aer MPS controls | `QiskitOpt.QAOA.AerMPSOmpThreads()`, `QiskitOpt.QAOA.AerMPSTruncationThreshold()`, `QiskitOpt.QAOA.AerMPSMaxBondDimension()`, `QiskitOpt.QAOA.AerMPSSampleMeasureAlgorithm()` |
| Named IBM backend | `QiskitOpt.QAOA.IBMBackend()` |
| Local emulation of an IBM backend | `QiskitOpt.QAOA.IBMBackend()` plus `QiskitOpt.QAOA.IsLocal()` |
| IBM Runtime account selection | `QiskitOpt.QAOA.Channel()` and `QiskitOpt.QAOA.Instance()` |
| Custom local backend factory | `QiskitOpt.QAOA.IBMFakeBackend()` |

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

set_attribute(model, QiskitOpt.QAOA.AerBackendMethod(), "matrix_product_state")
set_attribute(model, QiskitOpt.QAOA.AerSeedSimulator(), 73001)
set_attribute(model, QiskitOpt.QAOA.TranspilerSeed(), 73001)
```

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

## Upstream Boundary

QiskitOpt.jl should remain the JuMP/MOI adapter that sends a QUBO to Qiskit,
selects the execution backend, records seeds and angle metadata, and returns
samples. Broader QAOA research workflows belong upstream:

| Belongs in QiskitOpt.jl | Belongs upstream |
| --- | --- |
| `IBMBackend()`, `IsLocal()`, `IBMFakeBackend()`, Aer attributes, and seeds | Hardware-aware transpilation recipes from `qopt-best-practices` |
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
