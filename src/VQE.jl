module VQE

using Random
using Anneal
using PythonCall: pyconvert
using ..QiskitOpt:
    qiskit,
    qiskit_optimization_algorithms,
    qiskit_algorithms,
    qiskit_optimization_runtime,
    quadratic_program

Anneal.@anew Optimizer begin
    name    = "VQE @ IBMQ"
    sense   = :min
    domain  = :bool
    version = v"0.4.0"
    attributes = begin
        NumberOfReads["num_reads"]::Integer        = 1_000
        NumberOfShots["num_shots"]::Integer        = 1_024
        MaximumIterations["max_iter"]::Integer     = 50
        NumberOfRepetitions["num_reps"]::Integer   = 1
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
        IBMBackend["ibm_backend"]::String          = "ibmq_qasm_simulator"
        Entanglement["entanglement"]::String       = "linear"
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # -*- Retrieve Attributes - *-
    seed        = MOI.get(sampler, VQE.RandomSeed())
    num_reads   = MOI.get(sampler, VQE.NumberOfReads())
    ibm_backend = MOI.get(sampler, VQE.IBMBackend())

    # -*- Instantiate Random Generator -*- #
    rng = Random.Xoshiro(seed)

    # -*- Retrieve Model -*- #
    qp, α, β = quadratic_program(sampler)

    # Results vector
    samples = Vector{Anneal.Sample{T,Int}}(undef, num_reads)

    # Timing Information 
    time_data = Dict{String,Any}()

    # Connect to IBMQ and get backend
    connect(sampler) do client
        vqe     = qiskit_optimization_algorithms.MinimumEigenOptimizer(client)
        results = vqe.solve(qp)

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

        time_data["effective"] = pyconvert(
            Float64,
            results.min_eigen_solver_result.optimizer_time
        )

        return nothing
    end

    metadata = Dict{String,Any}(
        "origin" => "IBMQ @ $(ibm_backend)",
        "time"   => time_data, 
    )


    return Anneal.SampleSet{T}(samples, metadata)
end

function connect(
    callback::Function,
    sampler::Optimizer,
)
    # -*- Retrieve Attributes -*- #
    num_shots    = MOI.get(sampler, VQE.NumberOfShots())
    max_iter     = MOI.get(sampler, VQE.MaximumIterations())
    num_reps     = MOI.get(sampler, VQE.NumberOfRepetitions())
    num_qubits   = MOI.get(sampler, MOI.NumberOfVariables())
    ibm_backend  = MOI.get(sampler, VQE.IBMBackend())
    entanglement = MOI.get(sampler, VQE.Entanglement())

    # -*- Load Credentials -*- #
    qiskit.IBMQ.load_account()

    # -*- Connect to provider -*- #
    provider = qiskit.IBMQ.get_provider()
    backend  = provider.get_backend(ibm_backend)
    SPSA     = qiskit_algorithms.optimizers.SPSA(maxiter = max_iter)

    # -*- Setup Ansatz -*- #
    ansatz = qiskit.circuit.library.EfficientSU2(
        num_qubits   = num_qubits,
        reps         = num_reps,
        entanglement = entanglement,
    )

    # -*- Setup VQE Client -*- #
    client = qiskit_optimization_runtime.VQEClient(
        provider  = provider,
        backend   = backend,
        ansatz    = ansatz,
        optimizer = SPSA,
        shots     = num_shots,
    )

    callback(client)

    return nothing
end

end # module VQE