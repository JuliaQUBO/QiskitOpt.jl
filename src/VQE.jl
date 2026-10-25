module VQE

using Qiskit
using ..QiskitOpt:
    AerBackendConfig,
    IsingHamiltonian,
    NativeAnsatz,
    NativeFakeBackend,
    NativeLocalBackend,
    apply_layout_to_hamiltonian,
    backend_name,
    backend_version,
    bind_ansatz_parameters,
    build_n_local_circuit,
    configured_execution_backend,
    default_local_backend,
    effective_seed,
    efficient_su2_ansatz,
    empty_metadata,
    estimate_expectation,
    evaluate_ising_bitstring,
    measure_all!,
    minimize_objective,
    n_local_ansatz,
    preset_pass_manager,
    python_int_property,
    qiskit_parameter_names,
    quadratic_program,
    qubo_to_ising,
    random_unit_values,
    resolve_ansatz,
    runtime_channel,
    runtime_instance,
    sample_bits,
    sample_from_circuit,
    optimizer_result_metadata,
    python_float_vector,
    validate_initial_parameters
using ..QAOA: _decode_counts_entries, _run_circuit_with_hook, extract_layout_permutation

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

function default_ansatz(; num_qubits::Integer, kwargs...)
    return efficient_su2_ansatz(num_qubits; kwargs...)
end

QUBODrivers.@setup Optimizer begin
    name    = "VQE @ IBM Quantum"
    attributes = begin
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfReads["num_reads"]::Integer        = 100
        RandomSeed["seed"]::Union{Integer, Nothing} = nothing
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

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

function _check_n_variables(n_variables::Integer)
    n_variables >= 1 || throw(ArgumentError("n_variables must be at least 1"))
    return Int(n_variables)
end

function _ansatz_parameter_names(n_variables::Integer, ansatz)
    num_qubits = _check_n_variables(n_variables)
    resolved = try
        resolve_ansatz(ansatz, num_qubits)
    catch
        ansatz(num_qubits=num_qubits)
    end
    return qiskit_parameter_names(resolved)
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
    n_variables = first(QUBOTools.qubo(sampler, :dict))
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
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())

    # Results vector
    samples = QUBOTools.Sample{T,Int}[]

    retrieve_results = @timed retrieve(sampler) do _, sample_results, layout_perm...
        perm = isempty(layout_perm) ? nothing : first(layout_perm)
        for (state, reads) in _decode_counts_entries(sample_results, n; layout_permutation = perm)
            objective_value = QUBOTools.value(state, L, Q, α, β)
            sample = QUBOTools.Sample{T,Int}(
                state,
                sense == MOI.MAX_SENSE ? -objective_value : objective_value,
                reads,
            )
            push!(samples, sample)
        end

        return nothing
    end
    metadata = retrieve_results.value
    metadata["time"]["effective"] = retrieve_results.time

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
    sampler_seed    = MOI.get(sampler, QUBODrivers.RandomSeed())
    ibm_backend     = MOI.get(sampler, VQE.IBMBackend())
    ibm_fake_backend = MOI.get(sampler, VQE.IBMFakeBackend())
    ansatz_instance = MOI.get(sampler, VQE.Ansatz())
    channel         = runtime_channel(MOI.get(sampler, VQE.Channel()))
    instance        = runtime_instance(MOI.get(sampler, VQE.Instance()))
    initial_parameters   = MOI.get(sampler, VQE.InitialParameters())
    initial_parameter_source = MOI.get(sampler, VQE.InitialParameterSource())
    record_initial_parameters = MOI.get(sampler, VQE.RecordInitialParameters())
    is_local         = MOI.get(sampler, VQE.IsLocal())
    seed_simulator = effective_seed(MOI.get(sampler, VQE.AerSeedSimulator()), sampler_seed, 1)
    seed_transpiler = effective_seed(MOI.get(sampler, VQE.TranspilerSeed()), sampler_seed, 2)
    aer_config = AerBackendConfig(
        method=MOI.get(sampler, VQE.AerBackendMethod()),
        precision=MOI.get(sampler, VQE.AerPrecision()),
        max_parallel_threads=MOI.get(sampler, VQE.AerMaxParallelThreads()),
        mps_omp_threads=MOI.get(sampler, VQE.AerMPSOmpThreads()),
        mps_truncation_threshold=MOI.get(sampler, VQE.AerMPSTruncationThreshold()),
        mps_max_bond_dimension=MOI.get(sampler, VQE.AerMPSMaxBondDimension()),
        mps_sample_measure_algorithm=MOI.get(sampler, VQE.AerMPSSampleMeasureAlgorithm()),
        seed_simulator=seed_simulator,
        seed_transpiler=seed_transpiler,
    )

    n, L, Q, _, _ = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())
    ising_hamiltonian = qubo_to_ising(n, L, Q; sense = sense)
    num_qubits = ising_hamiltonian.num_qubits
    ansatz = resolve_ansatz(ansatz_instance, num_qubits)
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
    backend_release = backend_version(backend)

    pass_manager = preset_pass_manager(
        backend;
        optimization_level=3,
        seed_transpiler=aer_config.seed_transpiler,
    )

    ansatz_isa = pass_manager.run(ansatz)

    eval_counter = Ref(0)
    function cost_function(params::AbstractVector{<:Real})
        eval_counter[] += 1
        eval_seed = aer_config.seed_simulator === nothing ? nothing : Int(aer_config.seed_simulator) + eval_counter[] - 1
        if !(backend isa NativeLocalBackend || backend isa NativeFakeBackend) && ising_hamiltonian.num_qubits > 0
            meas_qc = bind_ansatz_parameters(ansatz, params)
            measure_all!(meas_qc)
            trans_res = pass_manager.run(meas_qc)
            perm = extract_layout_permutation(trans_res; num_variables = ising_hamiltonian.num_qubits)
            exec_qc = trans_res isa Qiskit.TranspileResult ? trans_res.circuit : trans_res
            raw_counts = _run_circuit_with_hook(
                exec_qc,
                backend;
                shots = num_reads,
            ).counts
            val = 0.0
            total_r = 0
            for (state, reads) in _decode_counts_entries(
                raw_counts,
                ising_hamiltonian.num_qubits;
                layout_permutation = perm,
            )
                val += reads * evaluate_ising_bitstring(ising_hamiltonian, state; include_offset = false)
                total_r += reads
            end
            return total_r > 0 ? val / total_r : 0.0
        end
        bound_qc = bind_ansatz_parameters(ansatz, params)
        return estimate_expectation(
            bound_qc,
            ising_hamiltonian;
            shots = num_reads,
            seed = eval_seed,
            precision = something(aer_config.precision, "double"),
            max_bond_dimension = aer_config.mps_max_bond_dimension,
            truncation_threshold = aer_config.mps_truncation_threshold,
        )
    end

    result = minimize_objective(
        cost_function,
        initial_parameter_values;
        max_iter = max_iter,
    )
    optimizer_iterations = python_int_property(result, :nit)
    optimizer_evaluations = python_int_property(result, :nfev)
    optimized_parameter_values = python_float_vector(result.x)
    optimized_parameter_optimizer = optimizer_result_metadata(
        result;
        iterations=optimizer_iterations,
        function_evaluations=optimizer_evaluations,
    )

    qc = bind_ansatz_parameters(ansatz, result.x)
    measure_all!(qc)
    optimized_qc_res = pass_manager.run(qc)
    optimized_qc = optimized_qc_res isa Qiskit.TranspileResult ? optimized_qc_res.circuit : optimized_qc_res
    final_perm = n > 0 ? extract_layout_permutation(optimized_qc_res; num_variables = n) : nothing

    samples = if backend isa NativeLocalBackend
        sample_from_circuit(
            optimized_qc;
            shots = final_num_reads,
            seed = aer_config.seed_simulator,
            precision = something(aer_config.precision, "double"),
            max_bond_dimension = aer_config.mps_max_bond_dimension,
            truncation_threshold = aer_config.mps_truncation_threshold,
        )
    else
        _run_circuit_with_hook(
            optimized_qc,
            backend;
            shots = final_num_reads,
        ).counts
    end

    if applicable(callback, result, samples, final_perm)
        callback(result, samples, final_perm)
    else
        callback(result, samples)
    end

    return empty_metadata(
        "VQE",
        backend_label,
        execution_mode;
        backend_version=backend_release,
        backend_config=backend_selection.config,
        backend_config_source=backend_selection.source,
        number_of_reads=final_num_reads,
        final_number_of_reads=final_num_reads,
        optimizer_iterations=optimizer_iterations,
        optimizer_evaluations=optimizer_evaluations,
        optimizer_number_of_reads=num_reads,
        seed_sampler=sampler_seed,
        seed_transpiler=aer_config.seed_transpiler,
        seed_optimizer=nothing,
        initial_parameters=record_initial_parameters ? initial_parameter_values : nothing,
        initial_parameter_names=record_initial_parameters ? parameter_names : nothing,
        initial_parameter_source=record_initial_parameters ? initial_parameter_source : nothing,
        optimized_parameters=optimized_parameter_values,
        optimized_parameter_names=parameter_names,
        optimized_parameter_optimizer=optimized_parameter_optimizer,
    )
end

end # module VQE
