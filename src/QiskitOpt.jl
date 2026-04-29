module QiskitOpt

using PythonCall
using QUBO
MOI = QUBODrivers.MOI

const DEFAULT_RUNTIME_CHANNEL = "ibm_quantum_platform"

# :: Python Qiskit Modules ::
const qiskit              = PythonCall.pynew()
const qiskit_optimization = PythonCall.pynew()
const qiskit_ibm_runtime  = PythonCall.pynew()
const qiskit_aer          = PythonCall.pynew()  
const scipy               = PythonCall.pynew()
const numpy               = PythonCall.pynew()
const _runtime_service_builder = Ref{Function}()

function __init__()
    # Load Python Packages
    PythonCall.pycopy!(qiskit, pyimport("qiskit"))
    PythonCall.pycopy!(qiskit_optimization, pyimport("qiskit_optimization"))
    PythonCall.pycopy!(qiskit_ibm_runtime, pyimport("qiskit_ibm_runtime"))
    PythonCall.pycopy!(qiskit_aer, pyimport("qiskit_aer"))
    PythonCall.pycopy!(scipy, pyimport("scipy"))
    PythonCall.pycopy!(numpy, pyimport("numpy"))
    _runtime_service_builder[] = default_runtime_service
end

function nonempty_or_nothing(value)
    if isnothing(value)
        return nothing
    end

    text = strip(String(value))
    return isempty(text) ? nothing : text
end

function runtime_channel(channel::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(channel)
    if !isnothing(provided)
        return provided == "ibm_quantum" ? DEFAULT_RUNTIME_CHANNEL : provided
    end

    env_channel = nonempty_or_nothing(get(ENV, "QISKIT_IBM_CHANNEL", nothing))
    if isnothing(env_channel) || env_channel == "ibm_quantum"
        return DEFAULT_RUNTIME_CHANNEL
    end

    return env_channel
end

function runtime_token()
    for name in ("QISKIT_IBM_TOKEN", "IBMQ_API_TOKEN")
        token = nonempty_or_nothing(get(ENV, name, nothing))
        !isnothing(token) && return token
    end

    return nothing
end

function runtime_instance(instance::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(instance)
    !isnothing(provided) && return provided

    for name in ("QISKIT_IBM_INSTANCE", "IBMQ_INSTANCE")
        resolved = nonempty_or_nothing(get(ENV, name, nothing))
        !isnothing(resolved) && return resolved
    end

    return nothing
end

function default_runtime_service(;
    channel::Union{Nothing,AbstractString} = nothing,
    token::Union{Nothing,AbstractString} = nothing,
    instance::Union{Nothing,AbstractString} = nothing,
)
    resolved_channel = runtime_channel(channel)
    resolved_token = nonempty_or_nothing(token)
    resolved_instance = runtime_instance(instance)

    if isnothing(resolved_token) && isnothing(resolved_instance)
        return qiskit_ibm_runtime.QiskitRuntimeService(channel=resolved_channel)
    elseif isnothing(resolved_token)
        return qiskit_ibm_runtime.QiskitRuntimeService(
            channel=resolved_channel,
            instance=resolved_instance,
        )
    elseif isnothing(resolved_instance)
        return qiskit_ibm_runtime.QiskitRuntimeService(
            channel=resolved_channel,
            token=resolved_token,
        )
    end

    return qiskit_ibm_runtime.QiskitRuntimeService(
        channel=resolved_channel,
        token=resolved_token,
        instance=resolved_instance,
    )
end

function runtime_service(;
    channel::Union{Nothing,AbstractString} = nothing,
    instance::Union{Nothing,AbstractString} = nothing,
)
    return _runtime_service_builder[](
        channel=runtime_channel(channel),
        token=runtime_token(),
        instance=runtime_instance(instance),
    )
end

function default_local_backend()
    return qiskit_aer.AerSimulator(method="matrix_product_state")
end

function backend_name(backend)
    try
        return PythonCall.pyconvert(String, backend.name)
    catch
        return PythonCall.pyconvert(String, backend.backend_name)
    end
end

function sample_bits(key)
    bitstring = replace(PythonCall.pyconvert(String, key), " " => "")
    return reverse(Int[(digit == '1') for digit in bitstring])
end

function empty_metadata(algorithm::AbstractString, backend::AbstractString, execution_mode::AbstractString)
    return Dict{String,Any}(
        "origin" => "$(algorithm) @ $(backend)",
        "backend" => backend,
        "execution_mode" => execution_mode,
        "time" => Dict{String,Any}(),
        "evals" => Float64[],
    )
end

function quadratic_program(sampler::QUBODrivers.AbstractSampler{T}) where {T}
    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)

    # Build Qiskit Model
    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()
    variable_names = string.(1:n)

    sense = MOI.get(sampler, MOI.ObjectiveSense())

    for i in 1:n
        linear[variable_names[i]] = L[i] * (sense == MOI.MIN_SENSE ? 1 : -1)
    end
    for i in 1:n, j in 1:n
        quadratic[variable_names[i], variable_names[j]] = Q[i,j] * (sense == MOI.MIN_SENSE ? 1 : -1)
    end

    qp = qiskit_optimization.QuadraticProgram()

    for v in variable_names
        qp.binary_var(v)
    end

    qp.minimize(linear = linear, quadratic = quadratic)
    
    return qp.to_ising()
end

export  VQE, QAOA

include("QAOA.jl")
include("VQE.jl")

end # module QiskitOpt
