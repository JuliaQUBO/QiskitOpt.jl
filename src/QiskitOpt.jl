module QiskitOpt

using PythonCall
import QUBODrivers: MOI, QUBODrivers, QUBOTools

# :: Python Qiskit Modules ::
const qiskit                         = PythonCall.pynew()
const qiskit_algorithms              = PythonCall.pynew()
const qiskit_optimization            = PythonCall.pynew()
const qiskit_optimization_algorithms = PythonCall.pynew()
const qiskit_ibm_runtime             = PythonCall.pynew()
const qiskit_utils                   = PythonCall.pynew()
const qiskit_minimum_eigensolvers                            = PythonCall.pynew()

function __init__()
    # Load Python Packages
    PythonCall.pycopy!(qiskit, pyimport("qiskit"))
    PythonCall.pycopy!(qiskit_algorithms, pyimport("qiskit.algorithms"))
    PythonCall.pycopy!(qiskit_optimization, pyimport("qiskit_optimization"))
    PythonCall.pycopy!(
        qiskit_optimization_algorithms,
        pyimport("qiskit_optimization.algorithms"),
    )
    PythonCall.pycopy!(qiskit_ibm_runtime, pyimport("qiskit_ibm_runtime"))
    PythonCall.pycopy!(qiskit_utils, pyimport("qiskit.utils"))
    PythonCall.pycopy!(qiskit_minimum_eigensolvers, pyimport("qiskit.algorithms.minimum_eigensolvers"))

    # IBMQ Credentials
    IBMQ_API_TOKEN = get(ENV, "IBMQ_API_TOKEN", nothing)

    if !isnothing(IBMQ_API_TOKEN)
        qiskit.IBMQ.save_account(IBMQ_API_TOKEN)
    end
end

function quadratic_program(sampler::QUBODrivers.AbstractSampler{T}) where {T}
    # Retrieve Model
    n, h, J, α, β = QUBOTools.qubo(sampler, :dict; sense = :min)

    # Build Qiskit Model
    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()

    for (i, val) in h
        linear[string(i)] = val
    end
    for ((i, j), val) in J
        quadratic[string(i), string(j)] = val
    end

    qp = qiskit_optimization.QuadraticProgram()

    for v in string.(QUBOTools.indices(sampler))
        qp.binary_var(v)
    end

    qp.minimize(linear = linear, quadratic = quadratic)

    return (qp, α, β)
end

export  QAOA, VQE

include("QAOA.jl")
include("VQE.jl")

end # module QiskitOpt
