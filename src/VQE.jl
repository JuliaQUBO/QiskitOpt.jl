module VQE

using Random
using LinearAlgebra
using PythonCall: pyconvert, pylist, pydict, pyint, pytuple, @pyexec
using ..QiskitOpt:
    qiskit,
    qiskit_ibm_runtime,
    qiskit_algorithms,
    quadratic_program,
    scipy,
    numpy

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
        CallbackFunction["callback"]::Function     = nothing
        CallbackDict["callback_dict"]::Dict{String,Any} = nothing
    end
end

# VQE cost function: ⟨Ψ(Θ)|H|Ψ(Θ)⟩ 
# @pyexec """
# def cost_function(params, ansatz, hamiltonian, estimator):
#     pub = (ansatz, [hamiltonian], [params])
#     result = estimator.run(pubs=[pub]).result()
#     energy = result[0].data.evs[0]
#     return energy
# """ => cost_function

function cost_function(params, ansatz, hamiltonian, estimator)
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    return energy
end


function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Retrieve Attribute
    seed        = MOI.get(sampler, VQE.RandomSeed())
    num_reads   = MOI.get(sampler, VQE.NumberOfReads())

    # Instantiate Random Generator
    rng = Random.Xoshiro(seed)

    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)

    # Results vector
    samples = Vector{Sample{T,Int}}(undef, num_reads)

    # Extra Information 
    # metadata = Dict{String,Any}(
    #     "origin" => "IBMQ VQE @ $(ibm_backend)",
    #     "time"   => Dict{String,Any}(),
    #     "evals"  => Vector{Float64}(),
    # )

    # Connect to IBMQ and get backend
    retrieve(sampler) do result, samples
        
        # eigenvalue = pyconvert(Float64, result.eigenvalue)

        Ψ = Vector{Int}[]
        ρ = Float64[]
        Λ = T[]

        for key in samples.keys()
            # state:
            state = pyconvert.(Int, key)
            push!(Ψ, state)
            # reads:
            push!(ρ, pyconvert(Float64, results.eigenstate[key]))
            # value: 
            push!(Λ, α * (state'(Q+Diagonal(L))*state) + β)
        end

        P = cumsum(ρ)

        for i = 1:num_reads
            p = rand(rng)
            j = first(searchsorted(P, p))

            samples[i] = Sample{T}(Ψ[j], Λ[j])
        end

        # metadata["time"]["effective"] = pyconvert(
        #     Float64,
        #     results.min_eigen_solver_result.optimizer_time
        # )

        return nothing
    end

    return SampleSet{T}(samples, metadata)
end

function retrieve(
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
    classical_optimizer   = MOI.get(sampler, VQE.ClassicalOptimizer())
    channel         = MOI.get(sampler, VQE.Channel())
    instance        = MOI.get(sampler, VQE.Instance())
    initial_point   = MOI.get(sampler, VQE.InitialPoint())
    is_local = ibm_backend == "" || ibm_backend == "ibmq_qasm_simulator"

    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel  = channel,
        instance = instance,
    )   

    backend = service.get_backend(ibm_backend)

    ising_hamiltonian = quadratic_program(sampler)
    ansatz = ansatz_instance(num_qubits = num_qubits)

    # pass manager for the quantum circuit (optimize the circuit for the target device)
    pass_manager = qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager(
        target = backend.target,
        optimization_level = 3
        )
    
    
    # Ansatz and Hamiltonian to ISA (Instruction Set Architecture)
    if !is_local
        ansatz = pass_manager.run(ansatz)
        ising_hamiltonian = ising_hamiltonian.apply_layout(layout = ansatz.layout)
    end

    if isnothing(initial_point)
        initial_point = pyint(2) * numpy.pi * numpy.random.random(ansatz.num_parameters)
    end

    session = if !is_local
        qiskit_ibm_runtime.Session(service=service, backend=backend)
    else
        nothing
    end

    # set Estimator primitive
    estimator = if !is_local
        qiskit_ibm_runtime.EstimatorV2(session=session)
    else
        qiskit.primitives.EstimatorV2()
    end
    estimator.options.default_shots = num_reps

    # result = vqe.compute_minimum_eigenvalue(ising_hamiltonian)

    println("Sending QUBO to IBMQ VQE...")
    println("Number of Qubits: ", ansatz.num_qubits)
    println("Initial Point: ", initial_point)
    println("Hamiltonian: ", ising_hamiltonian)
    result = scipy.optimize.minimize(
        cost_function,
        initial_point,
        args = pytuple(ansatz, ising_hamiltonian, estimator),
        method = "cobyla"
    )

    println("Status: ", result.message)

    qc = ansatz.assign_parameters(result.x)
    qc.measure_all()
    optimized_qc = if is_local
        qc
    else
        pass_manager.run(qc)
    end

    qiskit_sampler = Sampler(default_shots = pyint(num_reps))
    sampling_result = qiskit_sampler.run(pylist([optimized_qc])).result()[0]
    samples = sampling_result.data.meas.get_counts()

    callback(result, samples)

    return nothing
end

end # module VQE
