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

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

QUBODrivers.@setup Optimizer begin
    name    = "VQE @ IBMQ"
    attributes = begin
        NumberOfReads["num_reads"]::Integer        = 1_000
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfShots["num_shots"]::Integer   = 1000
        RandomSeed["seed"]::Union{Integer, Nothing}                = nothing
        InitialParameters["initial_parameters"]::Union{Vector{Float64}, Nothing} = nothing 
        IBMBackend["ibm_backend"]                  = qiskit_ibm_runtime.fake_provider.FakeManilaV2
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
# function cost_function(params, ansatz, hamiltonian, estimator)
#     pub = (ansatz, [hamiltonian], [params])
#     result = estimator.run(pubs=[pub]).result()
#     energy = result[0].data.evs[0]
#     return energy
# end


function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Retrieve Attribute
    seed        = MOI.get(sampler, VQE.RandomSeed())

    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)

    # Results vector
    samples = QUBOTools.Sample{T,Int}[]

    # Extra Information 
    metadata = Dict{String,Any}(
        "origin" => "IBMQ VQE (Qiskit)",
        "time"   => Dict{String,Any}(),
        "evals"  => Vector{Float64}(),
    )

    # Connect to IBMQ and get backend
    retrieve(sampler) do result, sample_results

        @show result, sample_results

        for key in sample_results.keys()
            state = parse.(Int,split(pyconvert.(String, key),""))
            sample = QUBOTools.Sample{T,Int}(
            # state:
            state,
            # energy:
            α * (state'* (Q+Diagonal(L)) * state) + β,
            # reads:
            pyconvert(Int, sample_results[key])
            )   
            push!(samples, sample)
        end

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
    num_shots        = MOI.get(sampler, VQE.NumberOfShots())
    num_qubits      = MOI.get(sampler, MOI.NumberOfVariables())
    ibm_backend     = MOI.get(sampler, VQE.IBMBackend())
    entanglement    = MOI.get(sampler, VQE.Entanglement())
    ansatz_instance = MOI.get(sampler, VQE.Ansatz())
    classical_optimizer   = MOI.get(sampler, VQE.ClassicalOptimizer())
    channel         = MOI.get(sampler, VQE.Channel())
    instance        = MOI.get(sampler, VQE.Instance())
    initial_parameters   = MOI.get(sampler, VQE.InitialParameters())

    @pyexec """
    def cost_function(params, ansatz, hamiltonian, estimator):
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]
        return energy
    """ => cost_function

    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel  = channel,
        instance = instance,
    )   

    backend = if typeof(ibm_backend) == String
         service.get_backend(ibm_backend)
    else
        ibm_backend()
    end

    ising_hamiltonian = quadratic_program(sampler)
    ansatz = ansatz_instance(num_qubits = num_qubits)

    # pass manager for the quantum circuit (optimize the circuit for the target device)
    pass_manager = qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager(
        target = backend.target,
        optimization_level = 3
        )
    
    
    # Ansatz and Hamiltonian to ISA (Instruction Set Architecture)
    ansatz_isa = pass_manager.run(ansatz)
    ising_hamiltonian = ising_hamiltonian.apply_layout(layout = ansatz_isa.layout)


    if isnothing(initial_parameters)
        initial_parameters = numpy.empty([ansatz_isa.num_parameters])
        for i in 1:pyconvert(Int, ansatz_isa.num_parameters)
            initial_parameters[i-1] = numpy.random.rand()
        end
    end

    session = qiskit_ibm_runtime.Session(service=service, backend=backend)
    estimator = qiskit_ibm_runtime.EstimatorV2(session=session)
    estimator.options.default_shots = num_shots

    # result = vqe.compute_minimum_eigenvalue(ising_hamiltonian)

    println("Sending QUBO to IBMQ VQE...")
    result = scipy.optimize.minimize(
        cost_function,
        initial_parameters,
        args = (ansatz_isa, ising_hamiltonian, estimator),
        method = "cobyla"
    )

    println("Status: ", result.message)

    qc = ansatz.assign_parameters(result.x)
    qc.measure_all()
    optimized_qc = pass_manager.run(qc)

    qiskit_sampler = qiskit.primitives.StatevectorSampler(default_shots = pyint(num_shots))
    sampling_result = qiskit_sampler.run(pylist([optimized_qc])).result()[0]
    samples = sampling_result.data.meas.get_counts()

    callback(result, samples)

    return nothing
end

end # module VQE
