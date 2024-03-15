# QiskitOpt.jl
[![DOI](https://zenodo.org/badge/587349377.svg)](https://zenodo.org/badge/latestdoi/587349377)

IBM Qiskit Optimization Wrapper for JuMP (ft. [Anneal.jl](https://github.com/psrenergy/Anneal.jl))

## Installation
```julia
julia> import Pkg

julia> Pkg.add("QiskitOpt.jl")
```

## Basic Usage
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

optimize!(model)

for i = 1:result_count(model)
    xi = value.(x; result=i)
    yi = objective_value(model; result=i)

    println("f($xi) = $yi")
end
```

## API Token
To access IBM's Quantum Computers, it is necessary to create an account at [IBM Q](https://quantum-computing.ibm.com/) to obtain an API Token and run the following python code:

```python
from qiskit import IBMQ

IBMQ.save_account("YOUR_TOKEN_HERE")
```

Another option is to set the `IBMQ_API_TOKEN` environment variable before loading `QiskitOpt.jl`:
```shell
$ export IBQM_API_TOKEN=YOUR_TOKEN_HERE

$ julia

julia> using QiskitOpt
```

**Disclaimer:** _The IBM Qiskit Optimization Wrapper for Julia is not officially supported by IBM. If you are a commercial customer interested in official support for Julia from IBM, let them know!_

**Note**: _If you are using [QiskitOpt.jl](https://github.com/psrenergy/QiskitOpt.jl) in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The PythonCall module will place a lot of files in this folder when building its Python environment._

