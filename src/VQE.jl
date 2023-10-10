module VQE

using Random
using PythonCall: pyconvert, pylist
using ..QiskitOpt:
    qiskit,
    qiskit_optimization_algorithms,
    qiskit_algorithms,
    qiskit_ibm_runtime,
    quadratic_program,
    qiskit_minimum_eigensolvers
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet

QUBODrivers.@setup Optimizer begin
    name    = "VQE @ IBMQ"
    attributes = begin
        NumberOfReads["num_reads"]::Integer        = 1_000
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfRepetitions["num_reps"]::Integer   = 1
        RandomSeed["seed"]::Union{Integer, Nothing}                = nothing
        InitialPoint["initial_point"]::Union{Vector{Float64}, Nothing} = nothing 
        IBMBackend["ibm_backend"]::String          = "ibmq_qasm_simulator"
        Entanglement["entanglement"]::String       = "linear"
        Channel["channel"]::String                 = "ibm_quantum"
        Instance["instance"]::String               = "ibm-q/open/main"
        ClassicalOptimizer["optimizer"]            = qiskit_algorithms.optimizers.COBYLA
        Ansatz["ansatz"]                           = qiskit.circuit.library.EfficientSU2
        IterationCallback["iteration_callback"]::Vector{Int}    = []
        ValueCallback["value_callback"]::Vector{Float64}        = []
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Retrieve Attribute
    seed        = MOI.get(sampler, VQE.RandomSeed())
    num_reads   = MOI.get(sampler, VQE.NumberOfReads())
    ibm_backend = MOI.get(sampler, VQE.IBMBackend())

    # Instantiate Random Generator
    rng = Random.Xoshiro(seed)

    # Retrieve Model
    qp, α, β = quadratic_program(sampler)

    # Results vector
    samples = Vector{Sample{T,Int}}(undef, num_reads)

    # Extra Information 
    metadata = Dict{String,Any}(
        "origin" => "IBMQ VQE @ $(ibm_backend)",
        "time"   => Dict{String,Any}(),
        "evals"  => Vector{Float64}(),
    )

    # Connect to IBMQ and get backend
    retrieve(sampler) do job_results
        results = client

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
            results.min_eigen_solver_result.optimizer_time
        )

        return nothing
    end

    return SampleSet{T}(samples, metadata)
end

function retieve(
    callback::Function,
    sampler::Optimizer{T},
) where {T}
    # Retrieve Attributes
    max_iter        = MOI.get(sampler, VQE.MaximumIterations())
    num_reps        = MOI.get(sampler, VQE.NumberOfRepetitions())
    num_qubits      = MOI.get(sampler, MOI.NumberOfVariables())
    ibm_backend     = MOI.get(sampler, VQE.IBMBackend())
    entanglement    = MOI.get(sampler, VQE.Entanglement())
    ansatz_instance = MOI.get(sampler, VQE.Ansatz())
    classical_opt   = MOI.get(sampler, VQE.ClassicalOptimizer())
    channel         = MOI.get(sampler, VQE.Channel())
    instance        = MOI.get(sampler, VQE.Instance())
    initial_point   = MOI.get(sampler, VQE.InitialPoint())
    
    # Set Optimizer
    optimizer = classical_opt(maxiter = max_iter)
    
    # Setup Ansatz
    ansatz = ansatz_instance(
        num_qubits   = num_qubits,
        reps         = num_reps,
        entanglement = entanglement,
        )
        
    if isnothing(initial_point)
        initial_point = pylist(rand(pyconvert(Int, ansatz.num_parameters)))
    end

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

    # Setup VQE
    vqe = qiskit_minimum_eigensolvers.SamplingVQE(
        sampler = qiskit_sampler,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback = _store_intermediate_results
    )

    qp, α, β = quadratic_program(sampler)

    quantum_optimizer = qiskit_optimization_algorithms.MinimumEigenOptimizer(vqe)

    results = quantum_optimizer.solve(qp)

    MOI.set(sampler, VQE.ValueCallback(), α *(pyconvert(Vector{Float64}, values) .+ β))
    MOI.set(sampler, VQE.IterationCallback(), pyconvert(Vector{Int}, counts))

    callback(results)

    return nothing
end

end # module VQE