module QiskitOpt

using Anneal
using PythonCall

# -*- :: Python Qiskit Modules :: -*- #
const qiskit                         = PythonCall.pynew()
const qiskit_algorithms              = PythonCall.pynew()
const qiskit_optimization            = PythonCall.pynew()
const qiskit_optimization_algorithms = PythonCall.pynew()
const qiskit_optimization_runtime    = PythonCall.pynew()
const qiskit_utils                   = PythonCall.pynew()

function __init__()
    # -*- Load Python Packages -*- #
    PythonCall.pycopy!(qiskit, pyimport("qiskit"))
    PythonCall.pycopy!(qiskit_algorithms, pyimport("qiskit.algorithms"))
    PythonCall.pycopy!(qiskit_optimization, pyimport("qiskit_optimization"))
    PythonCall.pycopy!(
        qiskit_optimization_algorithms,
        pyimport("qiskit_optimization.algorithms"),
    )
    PythonCall.pycopy!(qiskit_optimization_runtime, pyimport("qiskit_optimization.runtime"))
    PythonCall.pycopy!(qiskit_utils, pyimport("qiskit.utils"))

    # -*- IBMQ Credentials -*- #
    IBMQ_API_TOKEN = get(ENV, "IBMQ_API_TOKEN", nothing)

    if !isnothing(IBMQ_API_TOKEN)
        qiskit.IBMQ.save_account(IBMQ_API_TOKEN)
    end
end

function quadratic_program(sampler::Anneal.AbstractSampler{T}) where {T}
    # -*- Retrieve Model -*- #
    Q, α, β = Anneal.qubo(sampler, Dict)

    # -*- Build Qiskit Model -*- #
    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()

    for ((i, j), q) in Q
        if i == j
            linear[string(i)] = q
        else
            quadratic[string(i), string(j)] = q
        end
    end

    qp = qiskit_optimization.QuadraticProgram()

    for v in string.(Anneal.indices(sampler))
        qp.binary_var(v)
    end

    qp.minimize(linear = linear, quadratic = quadratic)

    return (qp, α, β)
end

export QAOA, VQE

include("QAOA.jl")
include("VQE.jl")

end # module QiskitOpt
