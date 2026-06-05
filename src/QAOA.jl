module QAOA

using PythonCall: pyconvert, pylist, pydict, pyint, @pyexec
using ..QiskitOpt:
    AerBackendConfig,
    backend_name,
    configured_execution_backend,
    default_local_backend,
    empty_metadata,
    preset_pass_manager,
    qiskit,
    qiskit_ibm_runtime,
    quadratic_program,
    runtime_channel,
    runtime_instance,
    sample_bits,
    scipy,
    numpy

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

QUBODrivers.@setup Optimizer begin
    name    = "QAOA @ IBM Quantum"
    attributes = begin
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfReads["num_reads"]::Integer        = 100
        NumberOfLayers["num_layers"]::Integer      = 1
        InitialParameters["initial_parameters"]::Union{Vector{Float64}, Nothing} = nothing 
        IBMFakeBackend["ibm_fake_backend"]         = default_local_backend
        IBMBackend["ibm_backend"]::Union{String, Nothing} = nothing
        IsLocal["is_local"]::Bool                  = false
        AerBackendMethod["aer_backend_method"]::Union{String, Nothing} = "matrix_product_state"
        AerPrecision["aer_precision"]::Union{String, Nothing} = nothing
        AerMaxParallelThreads["aer_max_parallel_threads"]::Union{Integer, Nothing} = nothing
        AerMPSOmpThreads["aer_mps_omp_threads"]::Union{Integer, Nothing} = nothing
        AerMPSTruncationThreshold["aer_mps_truncation_threshold"]::Union{Real, Nothing} = nothing
        AerMPSMaxBondDimension["aer_mps_max_bond_dimension"]::Union{Integer, Nothing} = nothing
        AerMPSSampleMeasureAlgorithm["aer_mps_sample_measure_algorithm"]::Union{String, Nothing} = nothing
        AerSeedSimulator["aer_seed_simulator"]::Union{Integer, Nothing} = nothing
        TranspilerSeed["transpiler_seed"]::Union{Integer, Nothing} = nothing
        Channel["channel"]::Union{String, Nothing} = nothing
        Instance["instance"]::Union{String, Nothing} = nothing
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Retrieve Model
    _, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())

    # Results vector
    samples = QUBOTools.Sample{T,Int}[]

    metadata = retrieve(sampler) do _, sample_results
        for key in sample_results.keys()
            state = sample_bits(key)
            objective_value = QUBOTools.value(state, L, Q, α, β)
            sample = QUBOTools.Sample{T,Int}(
                state,
                sense == MOI.MAX_SENSE ? -objective_value : objective_value,
                pyconvert(Int, sample_results[key]),
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
    max_iter        = MOI.get(sampler, QAOA.MaximumIterations())
    num_reads       = MOI.get(sampler, QAOA.NumberOfReads())
    num_layers      = MOI.get(sampler, QAOA.NumberOfLayers())
    ibm_backend     = MOI.get(sampler, QAOA.IBMBackend())
    ibm_fake_backend = MOI.get(sampler, QAOA.IBMFakeBackend())
    channel         = runtime_channel(MOI.get(sampler, QAOA.Channel()))
    instance        = runtime_instance(MOI.get(sampler, QAOA.Instance()))
    initial_parameters   = MOI.get(sampler, QAOA.InitialParameters())
    is_local         = MOI.get(sampler, QAOA.IsLocal())
    aer_config = AerBackendConfig(
        method=MOI.get(sampler, QAOA.AerBackendMethod()),
        precision=MOI.get(sampler, QAOA.AerPrecision()),
        max_parallel_threads=MOI.get(sampler, QAOA.AerMaxParallelThreads()),
        mps_omp_threads=MOI.get(sampler, QAOA.AerMPSOmpThreads()),
        mps_truncation_threshold=MOI.get(sampler, QAOA.AerMPSTruncationThreshold()),
        mps_max_bond_dimension=MOI.get(sampler, QAOA.AerMPSMaxBondDimension()),
        mps_sample_measure_algorithm=MOI.get(sampler, QAOA.AerMPSSampleMeasureAlgorithm()),
        seed_simulator=MOI.get(sampler, QAOA.AerSeedSimulator()),
        seed_transpiler=MOI.get(sampler, QAOA.TranspilerSeed()),
    )

    @pyexec """
    def cost_function(params, ansatz, hamiltonian, estimator):
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]
        return energy
    """ => cost_function

    backend_selection = configured_execution_backend(
        ibm_backend=ibm_backend,
        local_backend_factory=ibm_fake_backend,
        is_local=is_local,
        channel=channel,
        instance=instance,
        aer_config=aer_config,
    )
    backend = backend_selection.backend
    execution_mode = backend_selection.execution_mode
    backend_label = backend_name(backend)

    ising_qp = quadratic_program(sampler)
    ising_hamiltonian = ising_qp[0]
    ansatz = qiskit().circuit.library.QAOAAnsatz(
        ising_hamiltonian,
        reps=num_layers,
    )

    pass_manager = preset_pass_manager(
        backend;
        optimization_level=3,
        seed_transpiler=aer_config.seed_transpiler,
    )
    
    ansatz_isa = pass_manager.run(ansatz)
    ising_hamiltonian = ising_hamiltonian.apply_layout(layout=ansatz_isa.layout)


    if isnothing(initial_parameters)
        initial_parameters = numpy().zeros(pyint(pyconvert(Int, ansatz_isa.num_parameters)))
    else
        initial_parameters = numpy().array(initial_parameters)
    end

    estimator = qiskit_ibm_runtime().EstimatorV2(mode=backend)
    estimator.options.default_shots = num_reads
    scipy_options = pydict()
    scipy_options["maxiter"] = max_iter
    result = scipy().optimize.minimize(
        cost_function,
        initial_parameters,
        args=(ansatz_isa, ising_hamiltonian, estimator),
        method="cobyla",
        options=scipy_options,
    )

    qc = ansatz.assign_parameters(result.x)
    qc.measure_all()
    optimized_qc = pass_manager.run(qc)

    qiskit_sampler = qiskit_ibm_runtime().SamplerV2(mode=backend)
    sampling_result = qiskit_sampler.run(pylist([optimized_qc]), shots=pyint(num_reads)).result()[0]
    samples = sampling_result.data.meas.get_counts()

    callback(result, samples)

    return empty_metadata(
        "QAOA",
        backend_label,
        execution_mode;
        backend_config=backend_selection.config,
        backend_config_source=backend_selection.source,
        seed_transpiler=aer_config.seed_transpiler,
    )
end

end # module QAOA
