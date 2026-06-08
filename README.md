# QiskitOpt.jl
[![DOI](https://zenodo.org/badge/587349377.svg)](https://zenodo.org/badge/latestdoi/587349377)
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)

IBM Qiskit Optimization Wrapper for JuMP

## Installation
```julia
julia> import Pkg

julia> Pkg.add("QiskitOpt")
```

`Pkg.add("QiskitOpt")` installs the registered package from Julia's General registry. Use `QiskitOpt` as the package name; the `.jl` suffix only appears in the GitHub repository name.

`QiskitOpt` provides MOI-compatible optimizers and does not depend on JuMP directly. Install JuMP separately if you want to run the examples below:

```julia
julia> Pkg.add("JuMP")
```

If you want a development checkout from source instead of the registered package, use:

```julia
julia> import Pkg

julia> Pkg.develop(url="https://github.com/JuliaQUBO/QiskitOpt.jl")
```

## Local Quickstart
`QiskitOpt.jl` now defaults to credential-free local execution. If you do not pick a real IBM backend explicitly, both `QAOA.Optimizer` and `VQE.Optimizer` run locally with Aer using the matrix-product-state simulator.

```julia
using JuMP
using QiskitOpt

# Using QAOA
model = Model(QiskitOpt.QAOA.Optimizer)

# Using VQE
model = Model(QiskitOpt.VQE.Optimizer)

Q = [
   -1  2  2
    2 -1  2
    2  2 -1
]

@variable(model, x[1:3], Bin)
@objective(model, Min, x' * Q * x)

# No IBM token is required for default local Aer execution.
optimize!(model)

for i = 1:result_count(model)
    xi = value.(x; result=i)
    yi = objective_value(model; result=i)

    println("f($xi) = $yi")
end
```

## Maintained Examples

The `examples/ds_mfg_qubo` tutorial is a compact DS-MFG-style workflow that
loads scalar, linear, and quadratic QUBO CSV data, builds the MOI model, compares
cached QAOA/VQE sample distributions against a known classical solution pool,
and writes a distribution SVG without running expensive simulations by default.

```shell
julia --project=. examples/ds_mfg_qubo/ds_mfg_qubo_qiskitopt.jl
```

Set `QISKITOPT_DSMFG_RERUN=true` to regenerate the sample distributions through
local Aer.

## Runtime Diagnostics
Use `check_runtime` to verify the Julia/Python bridge and local Qiskit stack before running an optimizer:

```julia
using QiskitOpt

diagnostics = QiskitOpt.check_runtime(; local_backend=true, ibm=false)
diagnostics.ok
```

Julia reserves `local` as syntax, so the Julia-callable keyword is `local_backend`.
With `ibm=false`, diagnostics do not import `qiskit_ibm_runtime`. Set `ibm=true` when you want IBM Runtime availability reported separately.
The local backend check verifies the default Aer backend. Current QAOA/VQE local solves still use `qiskit_ibm_runtime` `EstimatorV2` and `SamplerV2` primitives, so run with `ibm=true` when you want a full solver-readiness check.

## Updating optimization parameters

```julia
# Number of shots
set_attribute(model, VQE.NumberOfReads(), 1000) # or QAOA.NumberOfReads

# Optional final sampling shots, separate from optimizer/estimator shots.
# If unset, final sampling uses NumberOfReads().
set_attribute(model, QiskitOpt.QUBODrivers.FinalNumberOfReads(), 8192)

# Maximum optimizer iterations
set_attribute(model, VQE.MaximumIterations(), 100) # or QAOA.MaximumIterations

# Ansatz
set_attribute(model, VQE.Ansatz(), QiskitOpt.qiskit.circuit.library.EfficientSU2)

# Number of QAOA ansatz repetitions (for QAOA only)
set_attribute(model, QAOA.NumberOfLayers(), 5)
```

## Initial Parameters
Both optimizers expose helpers for reproducible initial parameters and validate
the vector length before Qiskit execution starts.

```julia
# QAOA parameter order is beta angles followed by gamma angles.
QiskitOpt.QAOA.parameter_names(36; number_of_layers=2)

qaoa_start = QiskitOpt.QAOA.fixed_angle_initial_parameters(
    number_of_layers=2,
    gamma_sign=-1,
)
qaoa_guarantee = QiskitOpt.QAOA.fixed_angle_guarantee(number_of_layers=2)
set_attribute(model, QiskitOpt.QAOA.NumberOfLayers(), 2)
set_attribute(model, QiskitOpt.QAOA.InitialParameters(), qaoa_start)
set_attribute(model, QiskitOpt.QAOA.InitialParameterSource(), QiskitOpt.QAOA.FIXED_ANGLE_SOURCE)
```

```julia
# VQE parameter order comes from the selected Qiskit ansatz.
vqe_names = QiskitOpt.VQE.parameter_names(n_variables=36)
vqe_start = QiskitOpt.VQE.random_initial_parameters(n_variables=36, seed=73001)
set_attribute(model, QiskitOpt.VQE.InitialParameters(), vqe_start)
set_attribute(model, QiskitOpt.VQE.InitialParameterSource(), "random_seed_73001")
```

Returned `SampleSet` metadata records the source, parameter names, and actual
initial vector under `metadata["initial_parameters"]`. Disable this with
`RecordInitialParameters()` when you do not want parameters retained in metadata:

```julia
set_attribute(model, QiskitOpt.QAOA.RecordInitialParameters(), false)
```

## Configuring Local Aer
Both `QAOA.Optimizer` and `VQE.Optimizer` expose the same Aer local-simulation attributes. They apply when the optimizer builds the default local backend, and also when `IBMBackend()` is combined with `IsLocal() == true` to emulate a named IBM backend through Aer.

```julia
set_attribute(model, QAOA.AerBackendMethod(), "matrix_product_state") # or VQE.AerBackendMethod
set_attribute(model, QAOA.AerPrecision(), "single")
set_attribute(model, QAOA.AerMaxParallelThreads(), 8)
set_attribute(model, QAOA.AerMPSOmpThreads(), 4)
set_attribute(model, QAOA.AerMPSTruncationThreshold(), 1.0e-8)
set_attribute(model, QAOA.AerMPSMaxBondDimension(), 128)
set_attribute(model, QAOA.AerMPSSampleMeasureAlgorithm(), "mps_apply_measure")
set_attribute(model, QAOA.AerSeedSimulator(), 73001)
set_attribute(model, QAOA.TranspilerSeed(), 73001)
```

The returned `SampleSet` metadata records the selected backend configuration and simulator/transpiler seeds. If you need complete control, the existing backend-factory override remains available:

```julia
set_attribute(
    model,
    QAOA.IBMFakeBackend(),
    QiskitOpt.local_aer_backend_factory(
        method="statevector",
        precision="single",
        seed_simulator=73001,
    ),
)
```

You can still provide any zero-argument function that returns a Qiskit backend through `IBMFakeBackend()`.
When `IBMFakeBackend()` is set to an explicit factory, that factory is used as-is and the Aer backend-construction attributes above do not modify it. Use `local_aer_backend_factory(...)` when you want QiskitOpt to retain the factory's Aer configuration in result metadata.

## Choosing a Local Backend

```julia
# Use a fake IBM backend for local testing
set_attribute(
    model,
    VQE.IBMFakeBackend(),
    QiskitOpt.qiskit_ibm_runtime.fake_provider.FakeManilaV2,
)

# Or emulate the noise model of a named IBM backend locally through Aer
set_attribute(model, VQE.IBMBackend(), "ibm_fez")
set_attribute(model, VQE.IsLocal(), true)
```

List of fake backends available: [Qiskit Documentation](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider#fake-backends)

## IBM Quantum Platform
To run on a real IBM backend, set `IBMBackend()` explicitly and authenticate with the current IBM Quantum Platform flow. The package does not call `save_account()` on import and does not write credentials to disk for you.

Execution mode is selected from the backend attributes:

| Configuration | Behavior |
| --- | --- |
| No `IBMBackend()` | Use the configured local backend, which defaults to `AerSimulator(method="matrix_product_state")`. No IBM credentials or runtime service are required. |
| `IBMBackend()` and `IsLocal() == true` | Fetch the named IBM backend configuration, then run locally with `AerSimulator.from_backend(...)`. IBM credentials are required for backend lookup. |
| `IBMBackend()` and `IsLocal() == false` | Submit `EstimatorV2` and `SamplerV2` jobs to the selected IBM backend. IBM credentials are required. |

The supported environment variables are:

- `QISKIT_IBM_TOKEN`
- `QISKIT_IBM_INSTANCE`
- `QISKIT_IBM_CHANNEL` (optional, defaults to `ibm_quantum_platform`)

Legacy `IBMQ_API_TOKEN` and `IBMQ_INSTANCE` names are still accepted as fallbacks, but they are no longer the documented primary path.

If you do not call `set_attribute(model, QAOA.Channel(), ...)` or `set_attribute(model, VQE.Channel(), ...)`, the package will read `QISKIT_IBM_CHANNEL` and otherwise fall back to `ibm_quantum_platform`.

IBM documents the current account construction in the [`QiskitRuntimeService` API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit-runtime-service). The IBM docs describe the API key as the minimum required authentication input for non-local channels, but they recommend providing the instance or CRN as well to avoid account discovery and ambiguity. The local emulation behavior follows IBM's [Qiskit Runtime local testing mode](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode).

For the non-blocking GitHub Actions hardware smoke workflow, configure these repository settings under `Settings` -> `Secrets and variables` -> `Actions`:

- Secret `QISKIT_IBM_TOKEN`: IBM Quantum Platform API key.
- Secret `QISKIT_IBM_INSTANCE`: IBM Cloud CRN or runtime instance.
- Variable `QISKIT_IBM_BACKEND`: backend name, for example `ibm_fez`.
- Variable `QISKIT_IBM_CHANNEL`: optional, defaults to `ibm_quantum_platform` when omitted.

After those values are set, run the `IBM Hardware Smoke` workflow manually from the `Actions` tab, or let the scheduled monthly run exercise it.

Example setup:

```shell
export QISKIT_IBM_TOKEN=YOUR_API_KEY
export QISKIT_IBM_INSTANCE=YOUR_IBM_CLOUD_CRN
export QISKIT_IBM_CHANNEL=ibm_quantum_platform
```

```julia
using JuMP
using QiskitOpt

model = Model(QiskitOpt.QAOA.Optimizer)

@variable(model, x[1:3], Bin)
@objective(model, Min, x[1] + x[2] + x[3] - 2x[1]x[2] - 2x[2]x[3])

set_attribute(model, QiskitOpt.QAOA.IBMBackend(), "ibm_fez")

optimize!(model)
```

If you prefer a saved Qiskit account, the current Python-side command is:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_API_KEY",
    instance="YOUR_IBM_CLOUD_CRN",
    set_as_default=True,
)
```

**Disclaimer:** _The IBM Qiskit Optimization Wrapper for Julia is not officially supported by IBM. If you are a commercial customer interested in official support for Julia from IBM, let them know!_

**Note**: _If you are using [QiskitOpt.jl](https://github.com/JuliaQUBO/QiskitOpt.jl) in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The PythonCall module will place a lot of files in this folder when building its Python environment._
