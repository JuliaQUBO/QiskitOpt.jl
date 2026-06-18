# DS-MFG QAOA/VQE Example

This maintained example is a compact DS-MFG-style QUBO workflow for `QiskitOpt.jl`.
It keeps the expensive quantum simulation optional and uses cached sample
distributions by default.

The example data lives in `data/`:

- `scalars.csv`: QUBO variable count, scale, and offset.
- `linear.csv`: linear QUBO coefficients plus flow/auxiliary bookkeeping.
- `quadratic.csv`: sparse upper-triangular quadratic QUBO coefficients.
- `solution_pool.csv`: known classical optimum and nearby solutions.
- `cached_distributions.csv`: cached QAOA/VQE sample counts.

The first three variables are projected flow decisions. The final two variables
are auxiliary bits that duplicate selected flow bits. Raw QUBO energy evaluates
the full bitstring with penalties and offset; projected objective evaluates only
the flow decision; repaired raw energy replaces auxiliary bits with their
flow-consistent values before evaluating the QUBO.

## Run From A Checkout

Instantiate the package environment once:

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the cached-data path:

```shell
julia --project=. examples/ds_mfg_qubo/ds_mfg_qubo_qiskitopt.jl
```

Outputs are written under `examples/ds_mfg_qubo/output/`:

- `distribution_summary.csv`
- `distribution_comparison.svg`
- `sample_distributions.csv` when rerun is enabled

## QAOA Configuration

This tutorial is intentionally local-first. The default command reads
`data/cached_distributions.csv` so documentation builds, CI, and quick local
checks can inspect the DS-MFG bookkeeping without spending time in a quantum
optimizer. Regenerate the samples only when you want to inspect the local
QAOA/VQE path directly. The cached rows are recorded outputs from this
lightweight local configuration, not hardware benchmark data.

The QAOA rerun settings follow the local validation workflow in the
[QAOA best-practices guide](../../docs/qaoa_best_practices.md):

- `QAOA.IBMBackend()` is left unset, so QiskitOpt.jl uses local Aer instead of
  IBM Runtime hardware.
- `QAOA.NumberOfLayers() = 1` keeps the five-variable tutorial cheap to rerun;
  deeper or hardware-aware angle studies belong in a separate training
  workflow.
- `QAOA.NumberOfReads()` controls the optimizer sampling shot budget,
  `QUBODrivers.FinalNumberOfReads()` controls final sampling shots, and
  `QAOA.MaximumIterations()` controls the classical optimizer iteration budget.
  This example sets both read counts to `128` and caps optimization at `10`
  iterations.
- `QAOA.InitialParameters()` uses
  `QAOA.fixed_angle_initial_parameters(number_of_layers = 1)` and
  `QAOA.InitialParameterSource()` records `QAOA.FIXED_ANGLE_SOURCE`, so the
  starting angles remain auditable in metadata.
- Local Aer uses the default `matrix_product_state` method with
  `QAOA.AerPrecision() = "single"` and `QAOA.AerMaxParallelThreads() = 1`.
- `QAOA.AerSeedSimulator() = 73001` and `QAOA.TranspilerSeed() = 73001` pin the
  simulator and transpiler paths used when distributions are regenerated.

For a different QAOA study, choose and record the initializer deliberately.
Leave `InitialParameters()` unset for the documented zero default, use
`QAOA.random_initial_parameters(...; seed = seed)` with a source such as
`"random_seed_73001"`, keep the fixed-angle helper with
`QAOA.FIXED_ANGLE_SOURCE`, or use a schedule helper such as
`QAOA.linear_ramp_initial_parameters(...)` with a descriptive
`InitialParameterSource()`.

To move beyond this tutorial, first validate the QUBO, ordered angles, and
local Aer results, then run hardware-aware mapping, transpilation, and angle
training in an upstream workflow such as
[`qopt-best-practices`](https://github.com/qiskit-community/qopt-best-practices).
QiskitOpt.jl can export a fixed-parameter QAOA circuit and prepare a dry-run IBM
Runtime handoff, but this example intentionally avoids embedding generic
hardware transpilation, qubit selection, SWAP strategies, or angle-training
infrastructure.

## Rerun Local Aer

The cached path is intended for normal documentation and CI use. To regenerate
the sample distribution with local Aer, opt in explicitly:

```shell
QISKITOPT_DSMFG_RERUN=true julia --project=. examples/ds_mfg_qubo/ds_mfg_qubo_qiskitopt.jl
```

The rerun path uses local Aer defaults, fixed QAOA angles, seeded VQE initial
parameters, and deterministic simulator/transpiler seeds where supported:

- `QAOA.NumberOfLayers() = 1`
- `NumberOfReads() = 128`
- `QUBODrivers.FinalNumberOfReads() = 128`
- `MaximumIterations() = 10`
- `AerPrecision() = "single"`
- `AerMaxParallelThreads() = 1`
- `AerSeedSimulator() = 73001`
- `TranspilerSeed() = 73001`

You can redirect generated files with `QISKITOPT_DSMFG_OUTPUT_DIR`.

## CI Smoke Coverage

The package test suite includes this module and validates the loader,
classical objective bookkeeping, cached distributions, and SVG rendering path.
It does not run the Aer simulation in CI.

Reusable export, objective-bookkeeping, final-read, and reformulation-metadata
APIs live in lower-level packages such as `QUBOTools.jl`, `QUBODrivers.jl`, and
`ToQUBO.jl`. This example keeps only tutorial-specific CSV parsing, comparison,
and SVG rendering glue locally.
