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

## Updating optimization parameters

```julia
# Number of shots
set_attribute(model, VQE.NumberOfReads(), 1000) # or QAOA.NumberOfReads

# Maximum optimizer iterations
set_attribute(model, VQE.MaximumIterations(), 100) # or QAOA.MaximumIterations

# Ansatz
set_attribute(model, VQE.Ansatz(), QiskitOpt.qiskit.circuit.library.EfficientSU2)

# Number of QAOA ansatz repetitions (for QAOA only)
set_attribute(model, QAOA.NumberOfLayers(), 5)
```

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
