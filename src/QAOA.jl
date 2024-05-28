module QAOA

using LinearAlgebra
using PythonCall: pyconvert, pylist, pydict, pyint, pytuple, pybool, @pyexec
using ..QiskitOpt:
    qiskit,
    qiskit_ibm_runtime,
    qiskit_algorithms,
    qiskit_aer,
    quadratic_program,
    scipy,
    numpy

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

QUBODrivers.@setup Optimizer begin
    name    = "QAOA @ IBMQ"
    attributes = begin
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfReads["num_reads"]::Integer        = 100
        NumberOfLayers["num_layers"]::Integer   = 1
        InitialParameters["initial_parameters"]::Union{Vector{Float64}, Nothing} = nothing 
        IBMFakeBackend["ibm_fake_backend"]   = qiskit_ibm_runtime.fake_provider.FakeAlgiers
        IBMBackend["ibm_backend"]::Union{String, Nothing}  = nothing
        IsLocal["is_local"]::Bool          = false
        Entanglement["entanglement"]::String       = "linear"
        Channel["channel"]::String                 = "ibm_quantum"
        Instance["instance"]::String               = "ibm-q/open/main"
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # -*- Retrieve Attributes - *-
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    ibm_backend = MOI.get(sampler, QAOA.IBMBackend())

    # Results vector
    samples = QUBOTools.Sample{T,Int}[]

    # Extra Information 
    metadata = Dict{String,Any}(
        "origin" => "IBMQ QAOA @ $(ibm_backend)",
        "time"   => Dict{String,Any}(),
        "evals"  => Vector{Float64}(),
    )

    retrieve(sampler) do result, sample_results
        if MOI.get(sampler, MOI.ObjectiveSense()) == MOI.MAX_SENSE
            α = -α
        end

        for key in sample_results.keys()
            state = reverse(parse.(Int,split(pyconvert.(String, key),"")))
            sample = QUBOTools.Sample{T,Int}(
            # state:
            state,
            # energy:
            α * (state'* (Q+Diagonal(L)) * state + β),
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
    # -*- Retrieve Attributes -*- #
    max_iter        = MOI.get(sampler, QAOA.MaximumIterations())
    num_reads        = MOI.get(sampler, QAOA.NumberOfReads())
    num_layers      = MOI.get(sampler, QAOA.NumberOfLayers())
    ibm_backend     = MOI.get(sampler, QAOA.IBMBackend())
    ibm_fake_backend = MOI.get(sampler, QAOA.IBMFakeBackend())
    channel         = MOI.get(sampler, QAOA.Channel())
    instance        = MOI.get(sampler, QAOA.Instance())
    initial_parameters   = MOI.get(sampler, QAOA.InitialParameters())
    is_local    = MOI.get(sampler, QAOA.IsLocal())

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

    backend = if !isnothing(ibm_backend)
        _backend = service.get_backend(ibm_backend)
        if is_local && ibm_backend != "ibmq_qasm_simulator"
            qiskit_aer.AerSimulator.from_backend(_backend)
        else
            _backend
        end
    else
        _backend = ibm_fake_backend()
        is_local = true
        ibm_backend = _backend.backend_name
        _backend
    end

    if is_local && ibm_backend != "ibmq_qasm_simulator"
        backend = qiskit_aer.AerSimulator.from_backend(backend)
    end

    ising_qp = quadratic_program(sampler)
    ising_hamiltonian = ising_qp[0]
    ansatz = qiskit.circuit.library.QAOAAnsatz(
        ising_hamiltonian,
        reps = num_layers
    )

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

    estimator = if is_local || ibm_backend == "ibmq_qasm_simulator"
        qiskit_ibm_runtime.EstimatorV2(backend = backend)
    else
        session = qiskit_ibm_runtime.Session(service=service, backend=backend)
        qiskit_ibm_runtime.EstimatorV2(session=session)
    end
    if !is_local
        estimator.options.default_shots = num_reads
    end

    println("Running QAOA on $(ibm_backend)...")
    scipy_options = pydict()
    scipy_options["maxiter"] = max_iter
    result = scipy.optimize.minimize(
        cost_function,
        initial_parameters,
        args = (ansatz_isa, ising_hamiltonian, estimator),
        method = "cobyla",
        options = scipy_options
    )

    qc = ansatz.assign_parameters(result.x)
    qc.measure_all()
    optimized_qc = pass_manager.run(qc)

    qiskit_sampler = qiskit.primitives.StatevectorSampler(default_shots = pyint(num_reads))
    sampling_result = qiskit_sampler.run(pylist([optimized_qc])).result()[0]
    samples = sampling_result.data.meas.get_counts()

    callback(result, samples)

    return nothing
end

end # module QAOA
