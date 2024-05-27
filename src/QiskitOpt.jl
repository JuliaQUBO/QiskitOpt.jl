module QiskitOpt

using PythonCall
using QUBO
MOI = QUBODrivers.MOI

# :: Python Qiskit Modules ::
const qiskit              = PythonCall.pynew()
const qiskit_optimization = PythonCall.pynew()
const qiskit_ibm_runtime  = PythonCall.pynew()
const qiskit_algorithms   = PythonCall.pynew()
const qiskit_aer          = PythonCall.pynew()  
const scipy               = PythonCall.pynew()
const numpy               = PythonCall.pynew()

function __init__()
    # Load Python Packages
    PythonCall.pycopy!(qiskit, pyimport("qiskit"))
    PythonCall.pycopy!(qiskit_optimization, pyimport("qiskit_optimization"))
    PythonCall.pycopy!(qiskit_ibm_runtime, pyimport("qiskit_ibm_runtime"))
    PythonCall.pycopy!(qiskit_algorithms, pyimport("qiskit_algorithms"))
    PythonCall.pycopy!(qiskit_aer, pyimport("qiskit_aer"))
    PythonCall.pycopy!(scipy, pyimport("scipy"))
    PythonCall.pycopy!(numpy, pyimport("numpy"))

    # IBMQ Credentials
    IBMQ_API_TOKEN = get(ENV, "IBMQ_API_TOKEN", nothing)
    IBMQ_INSTANCE = get(ENV, "IBMQ_INSTANCE", "ibm-q/open/main")

    if !isnothing(IBMQ_API_TOKEN)
        qiskit_ibm_runtime.QiskitRuntimeService.save_account(channel=pystr("ibm_quantum"), instance = pystr(IBMQ_INSTANCE), token=pystr(IBMQ_API_TOKEN))
    end
end

function quadratic_program(sampler::QUBODrivers.AbstractSampler{T}) where {T}
    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)

    # Build Qiskit Model
    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()

    for i in 1:n
        linear[string(i)] = L[i]
    end
    for i in 1:n, j in 1:n
        quadratic[string(i), string(j)] = Q[i,j]
    end

    qp = qiskit_optimization.QuadraticProgram()

    for v in string.(QUBOTools.indices(sampler))
        qp.binary_var(v)
    end

    qp.minimize(linear = linear, quadratic = quadratic)

    return return qp.to_ising()[0]
end

export  VQE, QAOA

include("QAOA.jl")
include("VQE.jl")

end # module QiskitOpt
