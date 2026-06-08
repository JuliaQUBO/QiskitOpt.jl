module VQE

using PythonCall: pyconvert, pylist, pydict, pyint, @pyexec
using ..QiskitOpt:
    AerBackendConfig,
    backend_name,
    configured_execution_backend,
    default_local_backend,
    empty_metadata,
    preset_pass_manager,
    qiskit_parameter_names,
    qiskit,
    qiskit_ibm_runtime,
    quadratic_program,
    random_unit_values,
    runtime_channel,
    runtime_instance,
    sample_bits,
    scipy,
    numpy,
    validate_initial_parameters

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

function default_ansatz(; kwargs...)
    return qiskit().circuit.library.EfficientSU2(; kwargs...)
end

QUBODrivers.@setup Optimizer begin
    name    = "VQE @ IBM Quantum"
    attributes = begin
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfReads["num_reads"]::Integer        = 100
        InitialParameters["initial_parameters"]::Union{Vector{Float64}, Nothing} = nothing 
        InitialParameterSource["initial_parameter_source"]::Union{String, Nothing} = nothing
        RecordInitialParameters["record_initial_parameters"]::Bool = true
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
        Ansatz["ansatz"]                           = default_ansatz
    end
end

function _check_n_variables(n_variables::Integer)
    n_variables >= 1 || throw(ArgumentError("n_variables must be at least 1"))
    return Int(n_variables)
end

function _ansatz_parameter_names(n_variables::Integer, ansatz)
    num_qubits = _check_n_variables(n_variables)
    circuit = ansatz(num_qubits=num_qubits)
    return qiskit_parameter_names(circuit)
end

"""
    VQE.parameter_names(; n_variables, ansatz=VQE.default_ansatz)

Return the Qiskit list-binding order for the selected VQE ansatz.
"""
function parameter_names(; n_variables::Integer, ansatz = default_ansatz)
    return _ansatz_parameter_names(n_variables, ansatz)
end

function parameter_names(n_variables::Integer; ansatz = default_ansatz)
    return parameter_names(; n_variables=n_variables, ansatz=ansatz)
end

function parameter_names(sampler::QUBODrivers.AbstractSampler; ansatz = MOI.get(sampler, VQE.Ansatz()))
    ising_hamiltonian = quadratic_program(sampler)[0]
    n_variables = pyconvert(Int, ising_hamiltonian.num_qubits)
    return parameter_names(; n_variables=n_variables, ansatz=ansatz)
end

"""
    VQE.parameter_count(; n_variables, ansatz=VQE.default_ansatz)

Return the expected number of VQE initial parameters for the selected ansatz.
"""
function parameter_count(; n_variables::Integer, ansatz = default_ansatz)
    return length(parameter_names(; n_variables=n_variables, ansatz=ansatz))
end

function parameter_count(n_variables::Integer; ansatz = default_ansatz)
    return parameter_count(; n_variables=n_variables, ansatz=ansatz)
end

function parameter_count(sampler::QUBODrivers.AbstractSampler; ansatz = MOI.get(sampler, VQE.Ansatz()))
    return length(parameter_names(sampler; ansatz=ansatz))
end

"""
    VQE.random_initial_parameters(; n_variables, seed=nothing, rng=nothing, ansatz=VQE.default_ansatz)

Return random VQE angles in Qiskit parameter order. Passing `seed` makes
the returned vector reproducible.
"""
function random_initial_parameters(;
    n_variables::Integer,
    seed = nothing,
    rng = nothing,
    ansatz = default_ansatz,
)
    count = parameter_count(; n_variables=n_variables, ansatz=ansatz)
    return 2π .* random_unit_values(count; seed=seed, rng=rng)
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
    max_iter        = MOI.get(sampler, VQE.MaximumIterations())
    num_reads       = MOI.get(sampler, VQE.NumberOfReads())
    final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    ibm_backend     = MOI.get(sampler, VQE.IBMBackend())
    ibm_fake_backend = MOI.get(sampler, VQE.IBMFakeBackend())
    ansatz_instance = MOI.get(sampler, VQE.Ansatz())
    channel         = runtime_channel(MOI.get(sampler, VQE.Channel()))
    instance        = runtime_instance(MOI.get(sampler, VQE.Instance()))
    initial_parameters   = MOI.get(sampler, VQE.InitialParameters())
    initial_parameter_source = MOI.get(sampler, VQE.InitialParameterSource())
    record_initial_parameters = MOI.get(sampler, VQE.RecordInitialParameters())
    is_local         = MOI.get(sampler, VQE.IsLocal())
    aer_config = AerBackendConfig(
        method=MOI.get(sampler, VQE.AerBackendMethod()),
        precision=MOI.get(sampler, VQE.AerPrecision()),
        max_parallel_threads=MOI.get(sampler, VQE.AerMaxParallelThreads()),
        mps_omp_threads=MOI.get(sampler, VQE.AerMPSOmpThreads()),
        mps_truncation_threshold=MOI.get(sampler, VQE.AerMPSTruncationThreshold()),
        mps_max_bond_dimension=MOI.get(sampler, VQE.AerMPSMaxBondDimension()),
        mps_sample_measure_algorithm=MOI.get(sampler, VQE.AerMPSSampleMeasureAlgorithm()),
        seed_simulator=MOI.get(sampler, VQE.AerSeedSimulator()),
        seed_transpiler=MOI.get(sampler, VQE.TranspilerSeed()),
    )

    ising_qp = quadratic_program(sampler)
    ising_hamiltonian = ising_qp[0]
    num_qubits = pyconvert(Int, ising_hamiltonian.num_qubits)
    ansatz = ansatz_instance(num_qubits=num_qubits)
    parameter_names = qiskit_parameter_names(ansatz)
    expected_parameter_count = length(parameter_names)

    initial_parameter_values = if isnothing(initial_parameters)
        zeros(expected_parameter_count)
    else
        validate_initial_parameters(initial_parameters, expected_parameter_count, "VQE")
        Float64.(initial_parameters)
    end
    initial_parameter_source = if isnothing(initial_parameter_source)
        isnothing(initial_parameters) ? "default_zero" : "user"
    else
        initial_parameter_source
    end

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

    pass_manager = preset_pass_manager(
        backend;
        optimization_level=3,
        seed_transpiler=aer_config.seed_transpiler,
    )
    
    ansatz_isa = pass_manager.run(ansatz)
    ising_hamiltonian = ising_hamiltonian.apply_layout(layout=ansatz_isa.layout)


    initial_parameters = numpy().array(initial_parameter_values)

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
    sampling_result = qiskit_sampler.run(pylist([optimized_qc]), shots=pyint(final_num_reads)).result()[0]
    samples = sampling_result.data.meas.get_counts()

    callback(result, samples)

    return empty_metadata(
        "VQE",
        backend_label,
        execution_mode;
        backend_config=backend_selection.config,
        backend_config_source=backend_selection.source,
        seed_transpiler=aer_config.seed_transpiler,
        initial_parameters=record_initial_parameters ? initial_parameter_values : nothing,
        initial_parameter_names=record_initial_parameters ? parameter_names : nothing,
        initial_parameter_source=record_initial_parameters ? initial_parameter_source : nothing,
    )
end

end # module VQE
