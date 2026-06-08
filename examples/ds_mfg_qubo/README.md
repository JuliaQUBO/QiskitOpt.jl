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
- `MaximumIterations() = 10`
- `AerPrecision() = "single"`
- `AerMaxParallelThreads() = 1`

You can redirect generated files with `QISKITOPT_DSMFG_OUTPUT_DIR`.

## CI Smoke Coverage

The package test suite includes this module and validates the loader,
classical objective bookkeeping, cached distributions, and SVG rendering path.
It does not run the Aer simulation in CI.

Reusable export or objective-bookkeeping APIs should live in lower-level
packages such as `QUBOTools.jl`, `QUBODrivers.jl`, or `ToQUBO.jl` once those
interfaces exist. This example keeps only tutorial-specific glue locally.
