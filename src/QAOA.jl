module QAOA

using Anneal
using PythonCall

Anneal.@anew Optimizer begin
    name    = "IBM Qiskit QAOA"
    sense   = :min
    domain  = :bool
    version = v"0.4.0"
    attributes = begin
        NumberOfReads["num_reads"]::Integer        = 1_000
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
        IBMBackend["ibm_backend"]::String          = "ibmq_qasm_simulator"
    end
end

# -*- :: Python Qiskit Modules :: -*- #
const qiskit                         = PythonCall.pynew()
const qiskit_algorithms              = PythonCall.pynew()
const qiskit_optimization            = PythonCall.pynew()
const qiskit_optimization_algorithms = PythonCall.pynew()
const qiskit_optimization_runtime    = PythonCall.pynew()
const qiskit_utils                   = PythonCall.pynew()

function __init__()
    # -*- Load Python Packages -*- #
    PythonCall.pycopy!(qiskit                        , pyimport("qiskit"))
    PythonCall.pycopy!(qiskit_algorithms             , pyimport("qiskit.algorithms"))
    PythonCall.pycopy!(qiskit_optimization           , pyimport("qiskit_optimization"))
    PythonCall.pycopy!(qiskit_optimization_algorithms, pyimport("qiskit_optimization.algorithms"))
    PythonCall.pycopy!(qiskit_optimization_runtime   , pyimport("qiskit_optimization.runtime"))
    PythonCall.pycopy!(qiskit_utils                  , pyimport("qiskit.utils"))

    # -*- IBMQ Credentials -*- #
    IBMQ_API_TOKEN = get(ENV, "IBMQ_API_TOKEN", nothing)

    if !isnothing(IBMQ_API_TOKEN)
        qiskit.IBMQ.save_account(IBMQ_API_TOKEN)
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # -*- Retrieve Attributes - *-
    seed        = MOI.get(sampler, QAOA.RandomSeed())
    num_reads   = MOI.get(sampler, QAOA.NumberOfReads())
    ibm_backend = MOI.get(sampler, QAOA.IBMBackend())

    # -*- Retrieve Model -*- #
    Q, α, β = Anneal.qubo(sampler, Dict)

    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()

    for ((i, j), q) in Q
        if i == j
            linear[string(i)] = q
        else
            quadratic[string(i), string(j)] = q
        end
    end

    # -*- Build Qiskit Model -*- #
    qp = qiskit_optimization.QuadraticProgram()

    for v in string.(Anneal.indices(sampler))
        qp.binary_var(v)
    end
    
    qp.minimize(linear = linear, quadratic = quadratic)

    # Results vector
    samples = Vector{Anneal.Sample{T,Int}}(undef, num_reads)

    # Timing Information 
    time_data = Dict{String,Any}()

    # Connect to IBMQ and get backend
    connect(ibm_backend) do client
        qaoa    = qiskit_optimization_algorithms.MinimumEigenOptimizer(client)
        results = qaoa.solve(qp)

        Ψ = Vector{Int}[]
        ρ = Float64[]
        Λ = T[]

        for sample in results.samples
            # state:
            push!(Ψ, pyconvert.(Int, sample.x))
            # reads:
            push!(ρ, pyconvert(Float64, sample.probability))
            # value: 
            push!(Λ, α * (pyconvert(T, sample.fval) + β))
        end

        P = cumsum(ρ)

        for i = 1:num_reads
            p = rand()
            j = first(searchsorted(P, p))

            samples[i] = Sample{T}(Ψ[j], Λ[j])
        end

        time_data["effective"] = pyconvert(
            Float64,
            results.min_eigen_solver_result.optimizer_time
        )

        return nothing
    end
    
    metadata = Dict{String,Any}(
        "time"   => time_data,
        "origin" => "IBMQ @ $(ibm_backend)",
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

function connect(callback::Function, ibm_backend::String)
    qiskit.IBMQ.load_account()

    provider = qiskit.IBMQ.get_provider()
    backend  = provider.get_backend(ibm_backend)

    client = qiskit_optimization_runtime.QAOAClient(
        provider=provider,
        backend=backend,
    )

    callback(client)

    return nothing
end

end # module QAOA
