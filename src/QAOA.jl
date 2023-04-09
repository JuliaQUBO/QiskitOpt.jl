module QAOA

using Random
using PythonCall: pyconvert
using ..QiskitOpt:
    qiskit,
    qiskit_optimization_algorithms,
    qiskit_optimization_runtime,
    quadratic_program
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet,
    @setup,
    sample

@setup Optimizer begin
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

function sample(sampler::Optimizer{T}) where {T}
    # -*- Retrieve Attributes - *-
    seed        = MOI.get(sampler, QAOA.RandomSeed())
    num_reads   = MOI.get(sampler, QAOA.NumberOfReads())
    ibm_backend = MOI.get(sampler, QAOA.IBMBackend())

    # -*- Retrieve Model -*- #
    qp, α, β = quadratic_program(sampler)

    # -*- Instantiate Random Generator -*- #
    rng = Random.Xoshiro(seed)

    # Results vector
    samples = Vector{Sample{T,Int}}(undef, num_reads)

    # Timing Information 
    metadata = Dict{String,Any}(
        "origin" => "IBMQ QAOA @ $(ibm_backend)",
        "time"   => Dict{String,Any}(), 
    )

    # Connect to IBMQ and get backend
    connect(sampler) do client
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
            p = rand(rng)
            j = first(searchsorted(P, p))

            samples[i] = Sample{T}(Ψ[j], Λ[j])
        end

        metadata["time"]["effective"] = pyconvert(
            Float64,
            results.min_eigen_solver_result.optimizer_time,
        )

        return nothing
    end

    return SampleSet{T}(samples, metadata)
end

function connect(
    callback::Function,
    sampler::Optimizer,
)
    # -*- Retrieve Attributes -*- #
    ibm_backend = MOI.get(sampler, QAOA.IBMBackend())

    # -*- Load Credentials -*- #
    qiskit.IBMQ.load_account()

    # -*- Connect to provider -*- #
    provider = qiskit.IBMQ.get_provider()
    backend  = provider.get_backend(ibm_backend)

    # -*- Setup QAOA Client -*- #
    client = qiskit_optimization_runtime.QAOAClient(
        provider = provider,
        backend  = backend,
    )

    callback(client)

    return nothing
end

end # module QAOA
