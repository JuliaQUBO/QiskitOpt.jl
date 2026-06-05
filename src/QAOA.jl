module QAOA

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

const FIXED_ANGLE_SOURCE = "Wurtz-Lykov 3-regular tree fixed-angle table; arXiv:2107.00677"

const _WURTZ_LYKOV_3REGULAR_TREE_ANGLES = Dict(
    1 => (gamma = [0.616], beta = [0.393], guarantee = 0.6925),
    2 => (gamma = [0.488, 0.898], beta = [0.555, 0.293], guarantee = 0.7559),
    3 => (gamma = [0.422, 0.798, 0.937], beta = [0.609, 0.459, 0.235], guarantee = 0.7924),
    4 => (
        gamma = [0.409, 0.781, 0.988, 1.156],
        beta = [0.600, 0.434, 0.297, 0.159],
        guarantee = 0.8169,
    ),
    5 => (
        gamma = [0.360, 0.707, 0.823, 1.005, 1.154],
        beta = [0.632, 0.523, 0.390, 0.275, 0.149],
        guarantee = 0.8364,
    ),
)

QUBODrivers.@setup Optimizer begin
    name    = "QAOA @ IBM Quantum"
    attributes = begin
        MaximumIterations["max_iter"]::Integer     = 15
        NumberOfReads["num_reads"]::Integer        = 100
        NumberOfLayers["num_layers"]::Integer      = 1
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
    end
end

function _check_number_of_layers(number_of_layers::Integer)
    number_of_layers >= 1 || throw(ArgumentError("number_of_layers must be at least 1"))
    return Int(number_of_layers)
end

function _check_n_variables(n_variables::Integer)
    n_variables >= 1 || throw(ArgumentError("n_variables must be at least 1"))
    return Int(n_variables)
end

function _parameter_names(number_of_layers::Integer)
    p = _check_number_of_layers(number_of_layers)
    beta_names = ["β[$(layer)]" for layer in 0:(p - 1)]
    gamma_names = ["γ[$(layer)]" for layer in 0:(p - 1)]
    return vcat(beta_names, gamma_names)
end

"""
    QAOA.parameter_names([problem_or_nqubits]; number_of_layers)

Return the Qiskit list-binding order for `QAOAAnsatz` parameters.

# Examples
```julia
julia> QAOA.parameter_names(36; number_of_layers=2)
4-element Vector{String}:
 "β[0]"
 "β[1]"
 "γ[0]"
 "γ[1]"
```
"""
function parameter_names(; number_of_layers::Integer)
    return _parameter_names(number_of_layers)
end

function parameter_names(n_variables::Integer; number_of_layers::Integer)
    _check_n_variables(n_variables)
    return _parameter_names(number_of_layers)
end

function parameter_names(sampler::QUBODrivers.AbstractSampler; number_of_layers::Integer = MOI.get(sampler, QAOA.NumberOfLayers()))
    ising_hamiltonian = quadratic_program(sampler)[0]
    return parameter_names(ising_hamiltonian; number_of_layers=number_of_layers)
end

function parameter_names(cost_operator; number_of_layers::Integer)
    p = _check_number_of_layers(number_of_layers)
    ansatz = qiskit().circuit.library.QAOAAnsatz(cost_operator, reps=p)
    return qiskit_parameter_names(ansatz)
end

"""
    QAOA.parameter_count([problem_or_nqubits]; number_of_layers)

Return the expected number of QAOA initial parameters.
"""
function parameter_count(; number_of_layers::Integer)
    return length(parameter_names(; number_of_layers=number_of_layers))
end

function parameter_count(problem_or_nqubits; number_of_layers::Integer)
    return length(parameter_names(problem_or_nqubits; number_of_layers=number_of_layers))
end

"""
    QAOA.random_initial_parameters(; number_of_layers, seed=nothing, rng=nothing)

Return random QAOA angles in Qiskit parameter order. Passing `seed` makes
the returned vector reproducible.
"""
function random_initial_parameters(; number_of_layers::Integer, seed = nothing, rng = nothing)
    count = parameter_count(; number_of_layers=number_of_layers)
    return 2π .* random_unit_values(count; seed=seed, rng=rng)
end

"""
    QAOA.fixed_angle_initial_parameters(; number_of_layers, family=:wurtz_lykov_3regular_tree, gamma_sign=-1)

Return fixed-angle QAOA warm-start parameters in Qiskit's list-binding order.
The built-in Wurtz-Lykov 3-regular tree table supports depths 1 through 5.
"""
function fixed_angle_initial_parameters(;
    number_of_layers::Integer,
    family::Symbol = :wurtz_lykov_3regular_tree,
    gamma_sign::Real = -1,
)
    p = _check_number_of_layers(number_of_layers)
    if family != :wurtz_lykov_3regular_tree
        throw(ArgumentError("unsupported fixed-angle family: $(family)"))
    end
    if !haskey(_WURTZ_LYKOV_3REGULAR_TREE_ANGLES, p)
        throw(ArgumentError("fixed-angle family $(family) supports number_of_layers values 1 through 5"))
    end

    angles = _WURTZ_LYKOV_3REGULAR_TREE_ANGLES[p]
    return Float64[v for v in vcat(angles.beta, gamma_sign .* angles.gamma)]
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
    initial_parameter_source = MOI.get(sampler, QAOA.InitialParameterSource())
    record_initial_parameters = MOI.get(sampler, QAOA.RecordInitialParameters())
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

    ising_qp = quadratic_program(sampler)
    ising_hamiltonian = ising_qp[0]
    ansatz = qiskit().circuit.library.QAOAAnsatz(
        ising_hamiltonian,
        reps=num_layers,
    )
    parameter_names = qiskit_parameter_names(ansatz)
    expected_parameter_count = length(parameter_names)

    initial_parameter_values = if isnothing(initial_parameters)
        zeros(expected_parameter_count)
    else
        validate_initial_parameters(initial_parameters, expected_parameter_count, "QAOA")
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
        initial_parameters=record_initial_parameters ? initial_parameter_values : nothing,
        initial_parameter_names=record_initial_parameters ? parameter_names : nothing,
        initial_parameter_source=record_initial_parameters ? initial_parameter_source : nothing,
    )
end

end # module QAOA
