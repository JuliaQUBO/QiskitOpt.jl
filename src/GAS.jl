module GAS

using ..QiskitOpt:
    AerBackendConfig,
    backend_name,
    backend_version,
    build_gas_circuit,
    build_quadratic_program,
    configured_execution_backend,
    default_local_backend,
    effective_seed,
    empty_metadata,
    gas_grover_search,
    preset_pass_manager,
    python_float_property,
    python_float_vector,
    runtime_channel,
    runtime_instance

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

QUBODrivers.@setup Optimizer begin
    name    = "Grover Adaptive Search @ IBM Quantum"
    attributes = begin
        NumValueQubits["num_value_qubits"]::Union{Integer, Nothing} = nothing
        NumIterations["max_iter"]::Integer                          = 3
        CoefficientPrecision["coeff_precision"]::Integer            = 1
        NumberOfReads["num_reads"]::Integer                         = 100
        RandomSeed["seed"]::Union{Integer, Nothing}                 = nothing
        IBMFakeBackend["ibm_fake_backend"]                          = default_local_backend
        IBMBackend["ibm_backend"]::Union{String, Nothing}           = nothing
        IsLocal["is_local"]::Bool                                   = false
        AerBackendMethod["aer_backend_method"]::Union{String, Nothing} = "matrix_product_state"
        AerPrecision["aer_precision"]::Union{String, Nothing}       = nothing
        AerMaxParallelThreads["aer_max_parallel_threads"]::Union{Integer, Nothing} = nothing
        AerMPSOmpThreads["aer_mps_omp_threads"]::Union{Integer, Nothing} = nothing
        AerMPSTruncationThreshold["aer_mps_truncation_threshold"]::Union{Real, Nothing} = nothing
        AerMPSMaxBondDimension["aer_mps_max_bond_dimension"]::Union{Integer, Nothing} = nothing
        AerMPSSampleMeasureAlgorithm["aer_mps_sample_measure_algorithm"]::Union{String, Nothing} = nothing
        AerSeedSimulator["aer_seed_simulator"]::Union{Integer, Nothing} = nothing
        TranspilerSeed["transpiler_seed"]::Union{Integer, Nothing}  = nothing
        Channel["channel"]::Union{String, Nothing}                  = nothing
        Instance["instance"]::Union{String, Nothing}                = nothing
    end
end

const MaximumIterations = NumIterations
const NumQubits = NumValueQubits

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

function _check_num_value_qubits(num_value_qubits::Integer)
    num_value_qubits >= 2 ||
        throw(ArgumentError("NumValueQubits must be at least 2 for two's-complement encoding"))
    return Int(num_value_qubits)
end

function _check_num_iterations(num_iterations::Integer)
    num_iterations >= 1 || throw(ArgumentError("NumIterations must be at least 1"))
    return Int(num_iterations)
end

function _check_coefficient_precision(precision::Integer)
    precision >= 0 || throw(ArgumentError("CoefficientPrecision must be non-negative"))
    return Int(precision)
end

function _check_number_of_reads(number_of_reads::Integer)
    number_of_reads >= 1 || throw(ArgumentError("NumberOfReads must be at least 1"))
    return Int(number_of_reads)
end

function _validate_qubo_arrays(L::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    n = length(L)
    size(Q) == (n, n) || throw(ArgumentError("Q must have size ($(n), $(n)) to match L of length $(n)"))
    all(isfinite, L) || throw(ArgumentError("L must contain only finite real values"))
    all(isfinite, Q) || throw(ArgumentError("Q must contain only finite real values"))
    return n
end

"""
    GAS.integerize_coefficients(L, Q; precision=1)

Convert QUBO linear and quadratic coefficient arrays `(L, Q)` into integer
arrays required by Qiskit's `GroverOptimizer` value register.

When all entries of `L` and `Q` are already integer-valued, they are rounded to
`Int` without scaling (`scaled = false`, `scale_factor = 1.0`). When any entry
is non-integer, all coefficients are scaled by `10.0^precision` and rounded to
`Int` (`scaled = true`).
"""
function integerize_coefficients(
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real};
    precision::Integer = 1,
)
    _validate_qubo_arrays(L, Q)
    checked_precision = _check_coefficient_precision(precision)

    if all(isinteger, L) && all(isinteger, Q)
        return (
            L = Int[round(Int, c) for c in L],
            Q = Int[round(Int, c) for c in Q],
            scale_factor = 1.0,
            scaled = false,
        )
    end

    scale_factor = 10.0^checked_precision
    return (
        L = Int[round(Int, Float64(c) * scale_factor) for c in L],
        Q = Int[round(Int, Float64(c) * scale_factor) for c in Q],
        scale_factor = scale_factor,
        scaled = true,
    )
end

"""
    GAS.estimate_num_value_qubits(L, Q; precision=nothing)

Estimate the minimum two's-complement value-register qubit count needed to
represent the QUBO objective range `[neg_sum, pos_sum]`:

- `pos_sum = sum(max(c, 0) for c in all_coefficients)`
- `neg_sum = sum(min(c, 0) for c in all_coefficients)`
- `num_value_qubits = max(2, ceil(Int, log2(max(pos_sum, -neg_sum) + 1)) + 1)`

When `precision` is provided, `(L, Q)` are first integerized via
`GAS.integerize_coefficients(L, Q; precision=precision)`.
"""
function estimate_num_value_qubits(
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real};
    precision::Union{Integer, Nothing} = nothing,
)
    _validate_qubo_arrays(L, Q)

    L_use, Q_use = if isnothing(precision)
        L, Q
    else
        integerized = integerize_coefficients(L, Q; precision = precision)
        integerized.L, integerized.Q
    end

    pos_sum =
        sum(max(Float64(c), 0.0) for c in L_use; init = 0.0) +
        sum(max(Float64(c), 0.0) for c in Q_use; init = 0.0)
    neg_sum =
        sum(min(Float64(c), 0.0) for c in L_use; init = 0.0) +
        sum(min(Float64(c), 0.0) for c in Q_use; init = 0.0)

    bound = ceil(Int, max(pos_sum, -neg_sum))
    return max(2, ceil(Int, log2(bound + 1)) + 1)
end

function _operation_evaluations(result)
    try
        if pyhasattr(result, "operation_counts")
            counts = getproperty(result, :operation_counts)
            !pyis(counts, pybuiltins.None) && return Int(pylen(counts))
        end
    catch
    end
    return nothing
end

function _threshold_value(result)
    return python_float_property(result, :threshold)
end

function _extract_state_counts(result, n::Integer, total_reads::Integer)
    ordered_states = Vector{Int}[]
    state_probs = Dict{Vector{Int}, Float64}()

    if pyhasattr(result, "samples")
        py_samples = getproperty(result, :samples)
        if !pyis(py_samples, pybuiltins.None)
            for sample in py_samples
                state = Int[round(Int, v) for v in python_float_vector(getproperty(sample, :x))]
                length(state) == n || continue
                prob = max(0.0, pyconvert(Float64, getproperty(sample, :probability)))
                if !haskey(state_probs, state)
                    push!(ordered_states, state)
                    state_probs[state] = prob
                else
                    state_probs[state] += prob
                end
            end
        end
    end

    opt_state = if pyhasattr(result, "x") && !pyis(getproperty(result, :x), pybuiltins.None)
        candidate = Int[round(Int, v) for v in python_float_vector(getproperty(result, :x))]
        length(candidate) == n ? candidate : zeros(Int, n)
    else
        zeros(Int, n)
    end

    if !haskey(state_probs, opt_state)
        pushfirst!(ordered_states, opt_state)
        state_probs[opt_state] = isempty(state_probs) ? 1.0 : 0.0
    end

    if length(ordered_states) == 1
        return Pair{Vector{Int}, Int}[ordered_states[1] => total_reads]
    end

    counts = Dict{Vector{Int}, Int}()
    for state in ordered_states
        counts[state] = max(1, round(Int, state_probs[state] * total_reads))
    end

    current_total = sum(values(counts))
    diff = total_reads - current_total
    if diff > 0
        counts[opt_state] += diff
    elseif diff < 0
        # Reduce counts starting from states with the largest read counts while keeping each >= 1.
        for state in sort(ordered_states; by = s -> counts[s], rev = true)
            diff == 0 && break
            reducible = counts[state] - 1
            reducible <= 0 && continue
            delta = min(reducible, -diff)
            counts[state] -= delta
            diff += delta
        end
    end

    return Pair{Vector{Int}, Int}[state => counts[state] for state in ordered_states]
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())

    samples = QUBOTools.Sample{T, Int}[]

    retrieve_results = @timed retrieve(sampler, n, L, Q, sense) do state_counts
        for (state, reads) in state_counts
            objective_value = QUBOTools.value(state, L, Q, α, β)
            sample = QUBOTools.Sample{T, Int}(
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
    n::Integer,
    L::AbstractVector{T},
    Q::AbstractMatrix{T},
    sense::MOI.OptimizationSense,
) where {T}
    num_value_qubits_attr = MOI.get(sampler, GAS.NumValueQubits())
    max_iter              = _check_num_iterations(MOI.get(sampler, GAS.NumIterations()))
    coeff_precision       = _check_coefficient_precision(MOI.get(sampler, GAS.CoefficientPrecision()))
    num_reads             = _check_number_of_reads(MOI.get(sampler, GAS.NumberOfReads()))
    final_num_reads       = _check_number_of_reads(MOI.get(sampler, QUBODrivers.FinalNumberOfReads()))
    sampler_seed          = MOI.get(sampler, QUBODrivers.RandomSeed())
    ibm_backend           = MOI.get(sampler, GAS.IBMBackend())
    ibm_fake_backend      = MOI.get(sampler, GAS.IBMFakeBackend())
    channel               = runtime_channel(MOI.get(sampler, GAS.Channel()))
    instance              = runtime_instance(MOI.get(sampler, GAS.Instance()))
    is_local              = MOI.get(sampler, GAS.IsLocal())

    checked_num_value_qubits =
        isnothing(num_value_qubits_attr) ? nothing : _check_num_value_qubits(num_value_qubits_attr)

    seed_simulator  = effective_seed(MOI.get(sampler, GAS.AerSeedSimulator()), sampler_seed, 1)
    seed_transpiler = effective_seed(MOI.get(sampler, GAS.TranspilerSeed()), sampler_seed, 2)
    seed_optimizer  = effective_seed(nothing, sampler_seed, 3)

    aer_config = AerBackendConfig(
        method = MOI.get(sampler, GAS.AerBackendMethod()),
        precision = MOI.get(sampler, GAS.AerPrecision()),
        max_parallel_threads = MOI.get(sampler, GAS.AerMaxParallelThreads()),
        mps_omp_threads = MOI.get(sampler, GAS.AerMPSOmpThreads()),
        mps_truncation_threshold = MOI.get(sampler, GAS.AerMPSTruncationThreshold()),
        mps_max_bond_dimension = MOI.get(sampler, GAS.AerMPSMaxBondDimension()),
        mps_sample_measure_algorithm = MOI.get(sampler, GAS.AerMPSSampleMeasureAlgorithm()),
        seed_simulator = seed_simulator,
        seed_transpiler = seed_transpiler,
    )

    sign = sense == MOI.MIN_SENSE ? one(T) : -one(T)
    L_signed = sign .* L
    Q_signed = sign .* Q
    integerized = integerize_coefficients(L_signed, Q_signed; precision = coeff_precision)
    resolved_num_value_qubits = if isnothing(checked_num_value_qubits)
        estimate_num_value_qubits(integerized.L, integerized.Q)
    else
        checked_num_value_qubits
    end

    backend_selection = configured_execution_backend(
        ibm_backend = ibm_backend,
        local_backend_factory = ibm_fake_backend,
        is_local = is_local,
        channel = channel,
        instance = instance,
        aer_config = aer_config,
    )
    backend = backend_selection.backend
    execution_mode = backend_selection.execution_mode
    backend_label = backend_name(backend)
    backend_release = backend_version(backend)

    if n == 0
        callback(Pair{Vector{Int}, Int}[Int[] => final_num_reads])
        metadata = empty_metadata(
            "GAS",
            backend_label,
            execution_mode;
            backend_version = backend_release,
            backend_config = backend_selection.config,
            backend_config_source = backend_selection.source,
            number_of_reads = final_num_reads,
            final_number_of_reads = final_num_reads,
            optimizer_iterations = 0,
            optimizer_evaluations = 0,
            optimizer_number_of_reads = num_reads,
            seed_sampler = sampler_seed,
            seed_transpiler = aer_config.seed_transpiler,
            seed_optimizer = seed_optimizer,
            pass_manager_source = "default_preset",
            pass_manager_optimization_level = 3,
        )
        metadata["gas"] = Dict{String, Any}(
            "num_value_qubits" => resolved_num_value_qubits,
            "num_value_qubits_source" => isnothing(checked_num_value_qubits) ? "auto" : "user",
            "num_key_qubits" => 0,
            "num_iterations" => max_iter,
            "coefficient_precision" => coeff_precision,
            "coefficients_scaled" => integerized.scaled,
            "coefficient_scale_factor" => integerized.scale_factor,
            "threshold" => nothing,
        )
        return metadata
    end

    gas_res = gas_grover_search(
        n,
        integerized.L,
        integerized.Q;
        num_value_qubits = resolved_num_value_qubits,
        max_iter = max_iter,
        shots = final_num_reads,
        seed_simulator = aer_config.seed_simulator,
        seed_optimizer = seed_optimizer,
    )

    state_counts = gas_res.state_counts
    callback(state_counts)

    evaluations = gas_res.evaluations
    threshold_value = gas_res.threshold

    metadata = empty_metadata(
        "GAS",
        backend_label,
        execution_mode;
        backend_version = backend_release,
        backend_config = backend_selection.config,
        backend_config_source = backend_selection.source,
        number_of_reads = final_num_reads,
        final_number_of_reads = final_num_reads,
        optimizer_iterations = max_iter,
        optimizer_evaluations = evaluations,
        optimizer_number_of_reads = num_reads,
        seed_sampler = sampler_seed,
        seed_transpiler = aer_config.seed_transpiler,
        seed_optimizer = seed_optimizer,
        pass_manager_source = "default_preset",
        pass_manager_optimization_level = 3,
    )
    metadata["gas"] = Dict{String, Any}(
        "num_value_qubits" => resolved_num_value_qubits,
        "num_value_qubits_source" => isnothing(checked_num_value_qubits) ? "auto" : "user",
        "num_key_qubits" => n,
        "num_iterations" => max_iter,
        "coefficient_precision" => coeff_precision,
        "coefficients_scaled" => integerized.scaled,
        "coefficient_scale_factor" => integerized.scale_factor,
        "threshold" => threshold_value,
    )

    return metadata
end

end # module GAS
