module QAOA

using Random
using PythonCall: pyconvert, pylist
using ..QiskitOpt:
    qiskit,
    qiskit_optimization_algorithms,
    qiskit_ibm_runtime,
    quadratic_program
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet

QUBODrivers.@setup Optimizer begin
    name    = "QAOA @ IBMQ"
    attributes = begin
        NumberOfReads["num_reads"]::Integer        = 1_000
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfRepetitions["num_reps"]::Integer   = 1
        RandomSeed["seed"]::Union{Integer, Nothing}                = nothing
        InitialParameters["initial_parameters"]::Union{Vector{Float64}, Nothing} = nothing 
        IBMBackend["ibm_backend"]::String          = "ibmq_qasm_simulator"
        Entanglement["entanglement"]::String       = "linear"
        Channel["channel"]::String                 = "ibm_quantum"
        Instance["instance"]::String               = "ibm-q/open/main"
        ClassicalOptimizer["optimizer"]            = qiskit.algorithms.optimizers.COBYLA
        Ansatz["ansatz"]                           = qiskit.circuit.library.QAOAAnsatz
        IterationCallback["iteration_callback"]::Vector{Int}    = []
        ValueCallback["value_callback"]::Vector{Float64}        = []
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
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
        "evals"  => Vector{Float64}(),
    )

    # Connect to IBMQ and get backend
    retrieve(sampler) do job_results
        results = job_results
        
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

function retrieve(
    callback::Function,
    sampler::Optimizer{T},
) where {T}
    # -*- Retrieve Attributes -*- #
    max_iter        = MOI.get(sampler, QAOA.MaximumIterations())
    num_reps        = MOI.get(sampler, QAOA.NumberOfRepetitions())
    num_qubits      = MOI.get(sampler, MOI.NumberOfVariables())
    ibm_backend     = MOI.get(sampler, QAOA.IBMBackend())
    entanglement    = MOI.get(sampler, QAOA.Entanglement())
    classical_opt   = MOI.get(sampler, QAOA.ClassicalOptimizer())
    channel         = MOI.get(sampler, QAOA.Channel())
    instance        = MOI.get(sampler, QAOA.Instance())
    # initial_parameters   = MOI.get(sampler, QAOA.InitialParameters())
    reps            = MOI.get(sampler, QAOA.NumberOfRepetitions())
    
    # Set Optimizer
    optimizer = classical_opt(maxiter = max_iter)

    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel  = channel,
        instance = instance,
    )   

    session   = qiskit_ibm_runtime.Session(service=service, backend=ibm_backend)
    qiskit_sampler   = qiskit_ibm_runtime.Sampler(session=session)

    counts = pylist()
    values = pylist()

    function _store_intermediate_results(eval_count, parameters, mean, std)
        counts.append(eval_count)
        values.append(mean)
    end

    # Setup QAOA
    qaoa = qiskit.algorithms.QAOA(
        sampler = qiskit_sampler,
        optimizer=optimizer,
        reps = reps,
        callback = _store_intermediate_results
    )

    qp, α, β = quadratic_program(sampler)

    quantum_optimizer = qiskit_optimization_algorithms.MinimumEigenOptimizer(qaoa)

    results = quantum_optimizer.solve(qp)

    MOI.set(sampler, QAOA.ValueCallback(), α *(pyconvert(Vector{Float64}, values) .+ β))
    MOI.set(sampler, QAOA.IterationCallback(), pyconvert(Vector{Int}, counts))

    callback(results)

    return nothing
end

end # module QAOA
