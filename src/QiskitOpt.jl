module QiskitOpt

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

include("QAOA.jl")
include("VQE.jl")

end # module QiskitOpt
