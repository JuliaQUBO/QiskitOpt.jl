# QiskitOpt.jl
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7535883.svg)](https://doi.org/10.5281/zenodo.7535883)
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
`QiskitOpt.jl` now defaults to credential-free local execution. If you do not pick a real IBM backend explicitly, `QAOA.Optimizer`, `VQE.Optimizer`, and `GAS.Optimizer` run locally with Aer using the matrix-product-state simulator.

```julia
using JuMP
using QiskitOpt

# Using QAOA
model = Model(QiskitOpt.QAOA.Optimizer)

# Using VQE
model = Model(QiskitOpt.VQE.Optimizer)

# Using Grover Adaptive Search (GAS)
model = Model(QiskitOpt.GAS.Optimizer)

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

## Guides

- [QAOA best practices](docs/qaoa_best_practices.md) explains how to start
  locally with Aer, choose reproducible QAOA initial parameters, record metadata,
  and move selected runs to IBM hardware.
- [Pauli Correlation Encoding (PCE) evaluation](docs/pce_evaluation.md) documents
  the qubit compression, observable count, `EstimatorV2` call cost, `tanh`
  surrogate vs true QUBO energy separation, opt-in classical 1-flip polish
  attribution, and experimental submodule boundary for `QiskitOpt.PCE`.

## Runtime Diagnostics
Use `check_runtime` to verify the native `Qiskit.jl` + `QiskitIBMRuntime.jl` stack and local simulator before running an optimizer:

```julia
using QiskitOpt

diagnostics = QiskitOpt.check_runtime(; local_backend=true, ibm=false)
diagnostics.ok
```

Julia reserves `local` as syntax, so the Julia-callable keyword is `local_backend`.
With `ibm=false`, diagnostics verify the native `Qiskit.jl` circuit/transpiler engine and pure-Julia local simulator without querying `QiskitIBMRuntime.jl` credentials. Set `ibm=true` when you want IBM Runtime C-API availability reported separately, or `iqm=true` (after `using PythonCall`) when testing optional `iqm.qiskit_iqm` Python integration.

## Updating optimization parameters

```julia
# Number of shots
set_attribute(model, VQE.NumberOfReads(), 1000) # or QAOA.NumberOfReads

# Optional final sampling shots, separate from optimizer/estimator shots.
# If unset, final sampling uses NumberOfReads().
set_attribute(model, QiskitOpt.QUBODrivers.FinalNumberOfReads(), 8192)

# Maximum optimizer iterations
set_attribute(model, VQE.MaximumIterations(), 100) # or QAOA.MaximumIterations

# Standard benchmark seed. When algorithm-specific seeds are unset, QiskitOpt
# deterministically derives the local Aer and transpiler seeds from this value.
set_attribute(model, QiskitOpt.QUBODrivers.RandomSeed(), 73001)

# Ansatz (native descriptor or legacy QiskitOpt.qiskit.circuit.library.EfficientSU2)
set_attribute(model, VQE.Ansatz(), QiskitOpt.NativeAnsatz(:efficient_su2, 3; reps=3))

# Number of QAOA ansatz repetitions (for QAOA only)
set_attribute(model, QAOA.NumberOfLayers(), 5)

# Grover Adaptive Search (GAS) options (for GAS only)
set_attribute(model, GAS.NumValueQubits(), 5)        # `nothing` (default) auto-sizes from posiform/negaform bounds
set_attribute(model, GAS.NumIterations(), 3)         # iterations without improvement before stopping
set_attribute(model, GAS.CoefficientPrecision(), 2)  # decimal precision when integerizing non-integer coefficients
```

## Grover Adaptive Search (`GAS.Optimizer`)
`QiskitOpt.GAS.Optimizer` implements Grover Adaptive Search (Gilliam et al., 2021) natively in Julia for small binary QUBO instances:
- **Auto value-register sizing**: When `GAS.NumValueQubits()` (or alias `GAS.NumQubits()`) is `nothing` (default), `GAS.estimate_num_value_qubits(L, Q)` computes the minimum two's-complement bit width `max(2, ceil(Int, log2(max(pos_sum, -neg_sum) + 1)) + 1)` from the posiform upper bound `pos_sum` and negaform lower bound `neg_sum`.
- **Coefficient integerization**: Because the GAS value register uses two's-complement integer arithmetic, `GAS.integerize_coefficients(L, Q; precision=1)` scales and rounds non-integer QUBO coefficients by `10^precision` for the phase oracle while evaluating returned states against the **original unrounded QUBO** (`QUBOTools.value(state, L, Q, α, β)`) so reported objective values remain exact in the model's units.
- **Backend & seed attributes**: `GAS.Optimizer` accepts the same local simulator, fake-backend, and IBM Runtime attributes (`IBMBackend`, `IBMFakeBackend`, `IsLocal`, `Aer*`, `TranspilerSeed`, `Channel`, `Instance`, `RandomSeed`, `NumberOfReads`, and `FinalNumberOfReads`) as `QAOA.Optimizer` and `VQE.Optimizer`.

## Experimental Pauli Correlation Encoding (`PCE.Optimizer`)
`QiskitOpt.PCE` implements QSplit's Pauli Correlation Encoding (`solver_pce.py`) natively in Julia as an experimental submodule (see [`docs/pce_evaluation.md`](docs/pce_evaluation.md)):
- **Qubit compression**: Maps $N$ binary variables onto $m$ qubits ($N \le 3\binom{m}{k}$, so $m = O(\sqrt{N})$ for default `ParityWeight() == 2`) via weight-$k$ $X$, $Y$, and $Z$ Pauli strings (`PCE.pauli_correlation_observables`, `PCE.minimum_pce_qubits`).
- **True QUBO scoring**: Trains an $m$-qubit `n_local` ansatz with native statevector or 3-basis-circuit ($X^{\otimes m}, Y^{\otimes m}, Z^{\otimes m}$) shot parity evaluation on the smooth `tanh` surrogate (`PCE.surrogate_loss`), decodes expectation signs via `PCE.decode_expectations`, and **always** scores the returned binary assignment against the original QUBO (`QUBOTools.value(state, L, Q, α, β)`), recording the surrogate loss separately in `metadata["pce"]["surrogate_loss"]`.
- **Opt-in classical polish**: Classical 1-bit-flip local search (`PCE.bitflip_polish`) is **disabled by default** (`PCE.Polish() == false`) so quantum expectation decoding is not conflated with classical hill-climbing. When `MOI.set(model, PCE.Polish(), true)` is set, `metadata["pce"]` records both `raw_quantum_state` / `raw_quantum_value` and `polished_value` / `polish_flips` / `polish_passes`.

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

schedule_start = QiskitOpt.QAOA.linear_ramp_initial_parameters(number_of_layers=2)
set_attribute(model, QiskitOpt.QAOA.InitialParameters(), schedule_start)
set_attribute(model, QiskitOpt.QAOA.InitialParameterSource(), "linear_ramp_arxiv_2606_05311")
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

After a QAOA or VQE solve, `metadata["optimized_parameters"]` records the final
optimizer vector in the same order as `metadata["optimized_parameters"]["parameter_names"]`.
For QAOA this is the Qiskit order used by `QAOAAnsatz` and
`fixed_parameter_circuit`: beta angles followed by gamma angles. For VQE it is
the selected Qiskit ansatz parameter order. Use the top-level
`metadata["optimizer"]` for run-level optimizer accounting, and
`metadata["optimized_parameters"]["optimizer"]` for optimizer result fields tied
to the recorded trained vector.

```julia
MOI = QiskitOpt.QUBODrivers.MOI
raw_solver = MOI.get(JuMP.unsafe_backend(model), MOI.RawSolver())
sampleset = QiskitOpt.QUBOTools.solution(raw_solver)
metadata = QiskitOpt.QUBOTools.metadata(sampleset)

trained = metadata["optimized_parameters"]
trained["parameter_names"]
trained["values"]
trained["optimizer"]["objective_value"]

qaoa_reps = div(length(trained["values"]), 2)
qaoa_circuit, qaoa_metadata = QiskitOpt.QAOA.fixed_parameter_circuit(
    JuMP.unsafe_backend(model);
    parameters=trained["values"],
    reps=qaoa_reps,
    parameter_order=:qiskit,
    measure=true,
)
```

## Sample Postprocessing
Raw QUBO samples are kept as the solver returned them. When an application needs
projected variables, feasibility flags, repaired objectives, or labels from the
original model, attach those derived values with `postprocess_samples`:

```julia
processed = QiskitOpt.postprocess_samples(
    sampleset;
    name="toy_projection",
    version="1.0.0",
    source_problem="two-bit original model",
    scoring_convention="sum projected decision bits",
) do sample, context
    bits = QiskitOpt.QUBOTools.state(sample)
    projected = bits[1:2]

    return Dict(
        "original_objective" => sum(projected),
        "feasible" => sum(projected) <= 1,
        "projected_variables" => projected,
    )
end

postprocessing = QiskitOpt.QUBOTools.metadata(processed)["postprocessing"]
first_row = postprocessing["rows"][1]
first_row["raw_value"]                  # unchanged QUBO objective
first_row["reads"]                      # raw sample count
first_row["probability"]                # reads / total reads
first_row["derived"]["original_objective"]
```

The helper returns a new `SampleSet`; raw states, reads, probabilities, and QUBO
objective values are copied unchanged, while derived fields are stored under
`metadata["postprocessing"]["rows"][i]["derived"]`. The callback context includes
a shared read-only snapshot of the original `SampleSet` metadata as
`context.metadata`. To skip postprocessing, do not call the helper, or pass
`enabled=false` when keeping one code path for processed and unprocessed runs.
Callback failures throw
`QiskitOpt.SamplePostprocessingError` instead of returning a partially annotated
result.

## Fixed-Parameter QAOA Circuits
Use `QiskitOpt.QAOA.fixed_parameter_circuit` when parameters were trained
outside QiskitOpt and you need the Qiskit circuit that QiskitOpt would bind for
that QUBO. This path does not run `QAOA.Optimizer`, choose a backend, transpile,
sample, or contact IBM Runtime.

```julia
qaoa_parameters = QiskitOpt.QAOA.fixed_angle_initial_parameters(number_of_layers=2)

circuit, metadata = QiskitOpt.QAOA.fixed_parameter_circuit(
    JuMP.unsafe_backend(model);
    parameters=qaoa_parameters,
    reps=2,
    parameter_order=:beta_then_gamma,
    measure=true,
)
```

The metadata records QAOA depth, Qiskit parameter names and binding order,
variable order, Qiskit count-key bit order, objective scale/offset, objective
sign convention, and backend-independent circuit properties. Parameter values
stored in metadata always align with the Qiskit parameter names, regardless of
the input order. Store caller-specific angle provenance, such as training seed
or source path, alongside the returned metadata when you need it.

Use `QiskitOpt.QAOA.resource_audit` before a noisy simulator or hardware run to
record circuit resources without submitting a job or reading IBM credentials.
When `backend` is omitted, the audit transpiles against QiskitOpt's
credential-free local Aer backend. Pass a Qiskit fake backend, local
`AerSimulator.from_backend(...)`, or IBM backend object when you need a specific
target.

```julia
audit = QiskitOpt.QAOA.resource_audit(
    JuMP.unsafe_backend(model);
    parameters=qaoa_parameters,
    reps=2,
    optimization_level=3,
    transpiler_seed=73001,
    measure=true,
)

audit.metadata["untranspiled_circuit"]
audit.metadata["transpilation"]["status"] # "success", "skipped", or "failed"
audit.metadata["transpiled_circuit"]
```

The returned metadata includes QAOA layer and parameter details, variable and
measurement order, untranspiled depth/size/operation counts, backend target
summary, transpiler seed and optimization level, transpiled depth/size/operation
counts, two-qubit operation counts, measured-bit count, layout summary when
available, sanitized failure metadata, and QiskitOpt/Qiskit package versions.
Set `transpile=false` to record only the untranspiled circuit. The default Aer
backend is useful for credential-free transpilation checks; pass a fake or real
backend target when layout and routing metadata are part of the audit decision.

Use `QiskitOpt.QAOA.ibm_runtime_handoff` when the fixed circuit is ready for an
IBM Runtime `SamplerV2` handoff. The default mode is a credential-free dry run:
it records the intended backend, shots, transpiler seed, fixed-parameter
metadata, package versions, and count-scoring convention without contacting IBM
or writing token, instance/CRN, or account-file values into the metadata.

```julia
handoff = QiskitOpt.QAOA.ibm_runtime_handoff(
    circuit;
    fixed_metadata=metadata,
    backend="ibm_fez",
    shots=8192,
    transpiler_seed=73001,
)
```

Live submission is opt-in:

```julia
live_handoff = QiskitOpt.QAOA.ibm_runtime_handoff(
    circuit;
    fixed_metadata=metadata,
    backend="ibm_fez",
    shots=8192,
    transpiler_seed=73001,
    dry_run=false,
)
```

Runtime credentials remain outside QiskitOpt artifacts and are read through the
normal Qiskit Runtime account flow or environment variables. Set
`QISKIT_IBM_INSTANCE` or pass `instance=...` when Runtime cannot auto-resolve an
account instance; this may be required even when a token is present. If live
setup, transpilation, or submission fails before a job is returned, catch
`QiskitOpt.QAOA.RuntimeHandoffError` and persist `err.metadata`; `err.cause`
retains the original exception. Failure-message redaction removes
QiskitOpt-resolved token and instance values plus obvious `token=...`,
`instance=...`, `account_file=...`, and `crn:...` fragments, but callers should
still keep Runtime credentials outside project artifacts. Convert a Qiskit count key back to QUBO variable order with:

```julia
bits = QiskitOpt.QAOA.count_key_bits("0101")
```

## Reusable QAOA Execution Hooks and Optional IQM Backend Path
`QiskitOpt.QAOA` exposes a provider-neutral execution and count-decoding seam so fixed-parameter QAOA circuits and non-IBM Qiskit `BackendV2` targets (such as IQM Qiskit backends via `iqm.qiskit_iqm`) can run and produce scored `QUBOTools.SampleSet` results without requiring `QiskitIBMRuntime.jl` credentials or mandatory Python dependencies (`PythonCall` is an optional `[weakdeps]` extension `QiskitOptPythonCallExt`).

### Decoding Measurement Counts into a `SampleSet`
Use `QAOA.decode_counts_sampleset` when a circuit was executed on an external backend or custom runner and you want a `SampleSet` scored against the original QUBO (preserving variable order, transpiler layout/classical-bit permutation, objective scale/offset, minimization/maximization sense, and read multiplicities). Both Qiskit MSB-first bitstrings (`"100"`) and native `QiskitIBMRuntime.Samples` / LSB-first `BitVector` shot vectors are supported directly:

```julia
# Optional 1-based classical-bit permutation when physical/measured bit order
# differs from logical QUBO variable order [x1, x2, x3]:
sampleset = QiskitOpt.QAOA.decode_counts_sampleset(
    JuMP.unsafe_backend(model),
    Dict("100" => 40, "011" => 24);
    layout_permutation=[3, 1, 2],
    backend_name="iqm_fake_adonis",
    execution_mode="local",
)
```

When a transpiled `Qiskit.QuantumCircuit` (or `QiskitOpt.TranspileResult`) is available, pass `transpiled_circuit=transpiled_qc` (or call `QAOA.extract_layout_permutation(transpiled_qc; num_variables=n)`) to derive the virtual-variable-to-classical-bit permutation automatically from the transpiler layout and its `measure` instructions.

### Provider-Neutral Fixed-Parameter Execution and IQM Handoff
To use live IQM Qiskit backends (`iqm.qiskit_iqm`) via the optional `QiskitOptPythonCallExt` extension, load `PythonCall` in Julia and install `iqm-client[qiskit]` into the active Python environment:

```shell
pip install "iqm-client[qiskit]"
```

Check optional IQM Python availability with:

```julia
using PythonCall, QiskitOpt
diagnostics = QiskitOpt.check_runtime(; local_backend=true, ibm=false, iqm=true)
diagnostics.iqm_runtime
```

For remote IQM Resonance / IQM Server hardware, configure credentials through standard IQM environment variables (or keyword arguments) rather than storing secrets in project files:
- `IQM_SERVER_URL` (for example `https://cocos.resonance.meetiqm.com/garnet`)
- `IQM_TOKEN` or `IQM_TOKENS_FILE`

Prepare or execute a fixed-parameter QAOA circuit for IQM with `QAOA.iqm_runtime_handoff` (dry-run by default, redacting `IQM_TOKEN`, `IQM_TOKENS_FILE`, and `IQM_SERVER_URL`) or run end-to-end with `QAOA.execute_fixed_parameter_circuit`:

```julia
# Credential-free dry-run metadata for an IQM handoff:
iqm_dry_run = QiskitOpt.QAOA.iqm_runtime_handoff(
    circuit;
    fixed_metadata=metadata,
    server_url="https://cocos.resonance.meetiqm.com/garnet",
    quantum_computer="garnet",
    shots=4096,
    transpiler_seed=73001,
)

# Execute a fixed-parameter QAOA circuit on an IQM fake/real backend or custom BackendV2:
exec_result = QiskitOpt.QAOA.execute_fixed_parameter_circuit(
    JuMP.unsafe_backend(model);
    parameters=qaoa_parameters,
    reps=2,
    backend="IQMFakeAdonis", # or an IQMBackend / BackendV2 object
    shots=1024,
    transpiler_seed=73001,
)
exec_result.sampleset
```

You can also override final-circuit sampling in `QAOA.Optimizer` with `QAOA.CircuitExecutor()` (for example `QAOA.backend_run_executor` or a custom `(circuit, backend; shots) -> counts` callback).

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

The returned `SampleSet` metadata records the selected backend configuration,
standard sampler seed, derived or explicit simulator/transpiler seeds, final
read count, optimizer iteration/evaluation counts, and structured backend
name/version under `metadata["backend"]`. A scalar `metadata["backend_name"]`
alias is also recorded for scripts that only need the backend label. If you
need complete control, the existing backend-factory override remains
available:

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

## Custom QAOA Pass Managers
`QAOA.Optimizer` also accepts an optional pass-manager factory for users who
want to plug in hardware-aware QAOA transpilation flows from tooling such as
`qiskit-community/qopt-best-practices` without adding those tools as
QiskitOpt.jl dependencies. The hook is QAOA-only; VQE continues to use the
built-in level-3 preset pass manager.

```julia
function qaoa_pass_manager(backend; optimization_level=3, seed_transpiler=nothing)
    # Replace this body with a hardware-aware pass-manager recipe from your
    # Python/Qiskit workflow.
    return QiskitOpt.preset_pass_manager(
        backend;
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
end

set_attribute(model, QiskitOpt.QAOA.PassManagerFactory(), qaoa_pass_manager)
```

The factory receives the selected backend. QiskitOpt.jl passes
`optimization_level=3` and `TranspilerSeed()` as `seed_transpiler` when the
callable accepts those keywords. The returned pass manager is used for both the
optimization ansatz and the final measured circuit, and result metadata records
whether QAOA used the default preset path or a custom factory.

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

## Citation

For work that uses this Julia wrapper, cite `QiskitOpt.jl` with its
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.7535883). The concept DOI is
the evergreen software identifier and resolves to the latest archived wrapper
release. Citation metadata is available in [`CITATION.cff`](CITATION.cff).

For exact-version reproducibility, cite the Zenodo version DOI recorded in that
GitHub release and Zenodo record. For example, the latest archived Zenodo
record under this concept is
[10.5281/zenodo.23648754](https://doi.org/10.5281/zenodo.23648754) (`v0.7.2`).

This package builds on the native Julia C-API bindings [`Qiskit.jl`](https://github.com/Qiskit/Qiskit.jl) and [`QiskitIBMRuntime.jl`](https://github.com/Qiskit/QiskitIBMRuntime.jl) for upstream
[Qiskit](https://github.com/Qiskit/qiskit). When using Qiskit quantum circuits, transpilers, or optimization
algorithms, also cite Javadi-Abhari et al.,
["Quantum computing with Qiskit"](https://arxiv.org/abs/2405.08810), and the
upstream [Qiskit citation guidance](https://github.com/Qiskit/qiskit/blob/main/CITATION.bib).
The `QiskitOpt.jl` DOI identifies this Julia package; it does not identify or
archive upstream Qiskit components.

For general discussion or use of the JuliaQUBO ecosystem, cite the
[QUBO.jl ecosystem article](https://doi.org/10.1080/10556788.2026.2702926).

Release-time archive verification steps are documented in
[`.github/RELEASE_CHECKLIST.md`](.github/RELEASE_CHECKLIST.md).

**Disclaimer:** _The IBM Qiskit Optimization Wrapper for Julia is not officially supported by IBM. If you are a commercial customer interested in official support for Julia from IBM, let them know!_

**Note**: _`QiskitOpt.jl` runs natively in Julia via `Qiskit.jl` and `QiskitIBMRuntime.jl` without requiring a Python environment or `.CondaPkg`. `PythonCall` is an optional `[weakdeps]` extension (`QiskitOptPythonCallExt`) loaded only if you explicitly import `PythonCall` (for example, to drive third-party Python `iqm.qiskit_iqm` backends)._
