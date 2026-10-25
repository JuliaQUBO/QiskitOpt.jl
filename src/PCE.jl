module PCE

using Qiskit
using ..QiskitOpt:
    AerBackendConfig,
    NativeAnsatz,
    NativeLocalBackend,
    backend_name,
    backend_version,
    bind_ansatz_parameters,
    build_pce_basis_circuits,
    configured_execution_backend,
    decode_pce_expectations_from_basis_counts,
    default_local_backend,
    effective_seed,
    efficient_su2_ansatz,
    empty_metadata,
    estimate_pauli_expectations,
    minimize_objective,
    optimizer_result_metadata,
    preset_pass_manager,
    python_float_vector,
    python_int_property,
    qiskit_parameter_names,
    random_unit_values,
    resolve_ansatz,
    runtime_channel,
    runtime_instance,
    validate_initial_parameters
using ..QAOA: _run_circuit_with_hook, extract_layout_permutation

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

function default_ansatz(; num_qubits::Integer, reps::Integer = 2, kwargs...)
    return efficient_su2_ansatz(Int(num_qubits); reps = Int(reps), kwargs...)
end

QUBODrivers.@setup Optimizer begin
    name    = "Pauli Correlation Encoding (Experimental) @ IBM Quantum"
    attributes = begin
        MaximumIterations["max_iter"]::Integer                      = 15
        NumberOfReads["num_reads"]::Integer                         = 100
        NumberOfLayers["num_layers"]::Integer                       = 2
        ParityWeight["parity_weight"]::Integer                      = 2
        NumEncodingQubits["num_encoding_qubits"]::Union{Integer, Nothing} = nothing
        Sharpness["sharpness"]::Float64                             = 2.0
        Polish["polish"]::Bool                                      = false
        PolishMaxPasses["polish_max_passes"]::Integer               = 10
        RandomSeed["seed"]::Union{Integer, Nothing}                 = nothing
        InitialParameters["initial_parameters"]::Union{Vector{Float64}, Nothing} = nothing
        InitialParameterSource["initial_parameter_source"]::Union{String, Nothing} = nothing
        RecordInitialParameters["record_initial_parameters"]::Bool  = true
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
        Ansatz["ansatz"]                                            = default_ansatz
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

function _check_parity_weight(parity_weight::Integer)
    parity_weight >= 1 || throw(ArgumentError("parity_weight must be at least 1"))
    return Int(parity_weight)
end

function _check_num_layers(number_of_layers::Integer)
    number_of_layers >= 1 || throw(ArgumentError("number_of_layers must be at least 1"))
    return Int(number_of_layers)
end

function _check_max_iterations(max_iterations::Integer)
    max_iterations >= 1 || throw(ArgumentError("MaximumIterations must be at least 1"))
    return Int(max_iterations)
end

function _check_number_of_reads(number_of_reads::Integer)
    number_of_reads >= 1 || throw(ArgumentError("NumberOfReads must be at least 1"))
    return Int(number_of_reads)
end

function _check_sharpness(sharpness::Real)
    (isfinite(sharpness) && sharpness > 0) ||
        throw(ArgumentError("Sharpness must be a positive finite real number"))
    return Float64(sharpness)
end

function _check_polish_max_passes(max_passes::Integer)
    max_passes >= 0 || throw(ArgumentError("PolishMaxPasses must be non-negative"))
    return Int(max_passes)
end

raw"""
    PCE.max_encoded_variables(num_qubits; parity_weight=2)

Return the maximum number of binary QUBO variables $3 \binom{m}{k}$ that can be
encoded onto `num_qubits` ($m$) using weight-`parity_weight` ($k$) single-family
$X$, $Y$, and $Z$ Pauli strings.
"""
function max_encoded_variables(num_qubits::Integer; parity_weight::Integer = 2)
    k = _check_parity_weight(parity_weight)
    num_qubits >= k || throw(ArgumentError("num_qubits ($(num_qubits)) must be at least parity_weight ($(k))"))
    return 3 * binomial(Int(num_qubits), k)
end

raw"""
    PCE.minimum_pce_qubits(num_variables; parity_weight=2)

Return the smallest number of encoding qubits $m \ge k$ such that
$3 \binom{m}{k} \ge \text{num\_variables}$.
"""
function minimum_pce_qubits(num_variables::Integer; parity_weight::Integer = 2)
    num_variables >= 0 || throw(ArgumentError("num_variables must be non-negative"))
    k = _check_parity_weight(parity_weight)
    num_variables == 0 && return 0

    m = k
    while max_encoded_variables(m; parity_weight = k) < num_variables
        m += 1
    end
    return m
end

function _weight_k_subsets(n::Int, k::Int)
    subsets = Vector{Vector{Int}}()
    current = collect(1:k)
    while true
        push!(subsets, copy(current))
        pos = k
        while pos >= 1 && current[pos] == n - k + pos
            pos -= 1
        end
        pos == 0 && break
        current[pos] += 1
        for j in (pos + 1):k
            current[j] = current[j - 1] + 1
        end
    end
    return subsets
end

raw"""
    PCE.pauli_correlation_observables(num_variables; num_qubits=nothing, parity_weight=2)

Construct the deterministic mapping from `1:num_variables` binary QUBO variables
to weight-`parity_weight` Pauli strings across the $X$, $Y$, and $Z$ families.

Returns a NamedTuple with fields:
- `num_variables::Int`
- `num_qubits::Int`
- `parity_weight::Int`
- `max_capacity::Int`
- `pauli_strings::Vector{String}`
- `variable_map::Dict{Int,String}`
- `families::Dict{String,Vector{String}}`
"""
function pauli_correlation_observables(
    num_variables::Integer;
    num_qubits::Union{Integer, Nothing} = nothing,
    parity_weight::Integer = 2,
)
    n = Int(num_variables)
    n >= 0 || throw(ArgumentError("num_variables must be non-negative"))
    k = _check_parity_weight(parity_weight)

    if n == 0
        return (
            num_variables = 0,
            num_qubits = isnothing(num_qubits) ? 0 : Int(num_qubits),
            parity_weight = k,
            max_capacity = 0,
            pauli_strings = String[],
            variable_map = Dict{Int, String}(),
            families = Dict{String, Vector{String}}(
                "X" => String[],
                "Y" => String[],
                "Z" => String[],
            ),
        )
    end

    m = isnothing(num_qubits) ? minimum_pce_qubits(n; parity_weight = k) : Int(num_qubits)
    m >= k || throw(ArgumentError("num_qubits ($(m)) must be at least parity_weight ($(k))"))
    capacity = max_encoded_variables(m; parity_weight = k)
    capacity >= n || throw(
        ArgumentError(
            "num_qubits=$(m) with parity_weight=$(k) can encode at most $(capacity) variables, got $(n)",
        ),
    )

    subsets = _weight_k_subsets(m, k)
    all_strings = String[]
    for subset in subsets
        for pauli_char in ('X', 'Y', 'Z')
            chars = fill('I', m)
            for q_idx in subset
                # Place variable qubit index `q_idx` (1..m) at position `m - q_idx + 1` (Qiskit string order)
                chars[m - q_idx + 1] = pauli_char
            end
            push!(all_strings, String(chars))
            length(all_strings) == n && break
        end
        length(all_strings) == n && break
    end

    variable_map = Dict{Int, String}(i => all_strings[i] for i in 1:n)
    families = Dict{String, Vector{String}}(
        "X" => filter(s -> occursin('X', s), all_strings),
        "Y" => filter(s -> occursin('Y', s), all_strings),
        "Z" => filter(s -> occursin('Z', s), all_strings),
    )

    return (
        num_variables = n,
        num_qubits = m,
        parity_weight = k,
        max_capacity = capacity,
        pauli_strings = all_strings,
        variable_map = variable_map,
        families = families,
    )
end

raw"""
    PCE.surrogate_variables(expectations; sharpness=2.0)

Map Pauli expectation values $e_i \in [-1, 1]$ to smooth relaxed variables
$\tilde{x}_i = \frac{1}{2}(1 + \tanh(\text{sharpness} \cdot e_i)) \in (0, 1)$.
"""
function surrogate_variables(
    expectations::AbstractVector{<:Real};
    sharpness::Real = 2.0,
)
    alpha = _check_sharpness(sharpness)
    all(isfinite, expectations) || throw(ArgumentError("expectations must contain finite real values"))
    return Float64[0.5 * (1.0 + tanh(alpha * Float64(e))) for e in expectations]
end

"""
    PCE.surrogate_loss(expectations, L, Q; sharpness=2.0, regularization=0.0)

Evaluate the smooth `tanh` PCE surrogate loss on `(L, Q)` for variational
parameter optimization.

This surrogate is used only inside the classical optimizer loop; returned binary
assignments are always scored against the true QUBO via `QUBOTools.value`.
"""
function surrogate_loss(
    expectations::AbstractVector{<:Real},
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real};
    sharpness::Real = 2.0,
    regularization::Real = 0.0,
)
    n = length(L)
    length(expectations) == n ||
        throw(ArgumentError("expectations length ($(length(expectations))) must match L length ($(n))"))
    size(Q) == (n, n) ||
        throw(ArgumentError("Q must have size ($(n), $(n))"))
    (isfinite(regularization) && regularization >= 0) ||
        throw(ArgumentError("regularization must be a non-negative finite real number"))

    alpha = _check_sharpness(sharpness)
    x_tilde = surrogate_variables(expectations; sharpness = alpha)

    loss = 0.0
    for i in 1:n
        loss += Float64(L[i]) * x_tilde[i]
    end
    for i in 1:n, j in 1:n
        loss += Float64(Q[i, j]) * x_tilde[i] * x_tilde[j]
    end

    if regularization > 0 && n > 0
        reg_penalty = sum((1.0 - tanh(alpha * Float64(e))^2) for e in expectations) / n
        loss += Float64(regularization) * reg_penalty
    end

    return loss
end

raw"""
    PCE.decode_expectations(expectations)

Threshold-decode Pauli expectation values $e_i$ into a binary assignment
$x_i = (e_i \ge 0 ? 1 : 0) \in \{0, 1\}^N$.
"""
function decode_expectations(expectations::AbstractVector{<:Real})
    all(isfinite, expectations) || throw(ArgumentError("expectations must contain finite real values"))
    return Int[Float64(e) >= 0.0 ? 1 : 0 for e in expectations]
end

"""
    PCE.bitflip_polish(state, L, Q, scale=1.0, offset=0.0; sense=MOI.MIN_SENSE, max_passes=10)

Perform opt-in greedy 1-bit-flip local search starting from `state`, evaluating
candidate assignments against the exact QUBO objective `QUBOTools.value(x, L, Q, scale, offset)`.

Returns `(state=..., value=..., raw_value=..., flips=..., passes=...)`.
"""
function bitflip_polish(
    state::AbstractVector{<:Integer},
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real},
    scale::Real = 1.0,
    offset::Real = 0.0;
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
    max_passes::Integer = 10,
)
    n = length(L)
    length(state) == n || throw(ArgumentError("state length ($(length(state))) must match L length ($(n))"))
    size(Q) == (n, n) || throw(ArgumentError("Q must have size ($(n), $(n))"))
    all(v -> v in (0, 1), state) || throw(ArgumentError("state entries must be binary (0 or 1)"))
    checked_passes = _check_polish_max_passes(max_passes)

    current_state = Int[Int(v) for v in state]
    raw_value = Float64(QUBOTools.value(current_state, L, Q, scale, offset))
    current_value = raw_value
    sign = sense == MOI.MIN_SENSE ? 1.0 : -1.0

    flips = 0
    passes = 0
    for _ in 1:checked_passes
        passes += 1
        improved_in_pass = false
        for i in 1:n
            current_state[i] = 1 - current_state[i]
            candidate_value = Float64(QUBOTools.value(current_state, L, Q, scale, offset))
            if sign * candidate_value < sign * current_value - 1.0e-12
                current_value = candidate_value
                flips += 1
                improved_in_pass = true
            else
                current_state[i] = 1 - current_state[i]
            end
        end
        improved_in_pass || break
    end

    return (
        state = current_state,
        value = current_value,
        raw_value = raw_value,
        flips = flips,
        passes = passes,
    )
end

function parameter_names(;
    num_variables::Integer,
    num_encoding_qubits::Union{Integer, Nothing} = nothing,
    parity_weight::Integer = 2,
    number_of_layers::Integer = 2,
    ansatz = default_ansatz,
)
    obs = pauli_correlation_observables(
        num_variables;
        num_qubits = num_encoding_qubits,
        parity_weight = parity_weight,
    )
    p = _check_num_layers(number_of_layers)
    obs.num_qubits == 0 && return String[]
    circuit = ansatz(num_qubits = obs.num_qubits, reps = p)
    return qiskit_parameter_names(circuit)
end

function parameter_count(; kwargs...)
    return length(parameter_names(; kwargs...))
end

function random_initial_parameters(;
    num_variables::Integer,
    num_encoding_qubits::Union{Integer, Nothing} = nothing,
    parity_weight::Integer = 2,
    number_of_layers::Integer = 2,
    seed = nothing,
    rng = nothing,
    ansatz = default_ansatz,
)
    count = parameter_count(
        num_variables = num_variables,
        num_encoding_qubits = num_encoding_qubits,
        parity_weight = parity_weight,
        number_of_layers = number_of_layers,
        ansatz = ansatz,
    )
    return 2π .* random_unit_values(count; seed = seed, rng = rng)
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())

    samples = QUBOTools.Sample{T, Int}[]

    retrieve_results = @timed retrieve(sampler, n, L, Q, α, β, sense) do final_state, final_reads
        objective_value = QUBOTools.value(final_state, L, Q, α, β)
        sample = QUBOTools.Sample{T, Int}(
            final_state,
            sense == MOI.MAX_SENSE ? -objective_value : objective_value,
            final_reads,
        )
        push!(samples, sample)
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
    α::T,
    β::T,
    sense::MOI.OptimizationSense,
) where {T}
    max_iter                  = _check_max_iterations(MOI.get(sampler, PCE.MaximumIterations()))
    num_reads                 = _check_number_of_reads(MOI.get(sampler, PCE.NumberOfReads()))
    final_num_reads           = _check_number_of_reads(MOI.get(sampler, QUBODrivers.FinalNumberOfReads()))
    num_layers                = _check_num_layers(MOI.get(sampler, PCE.NumberOfLayers()))
    parity_weight             = _check_parity_weight(MOI.get(sampler, PCE.ParityWeight()))
    num_encoding_qubits_attr  = MOI.get(sampler, PCE.NumEncodingQubits())
    sharpness                 = _check_sharpness(MOI.get(sampler, PCE.Sharpness()))
    polish_enabled            = MOI.get(sampler, PCE.Polish())
    polish_max_passes         = _check_polish_max_passes(MOI.get(sampler, PCE.PolishMaxPasses()))
    sampler_seed              = MOI.get(sampler, QUBODrivers.RandomSeed())
    ibm_backend               = MOI.get(sampler, PCE.IBMBackend())
    ibm_fake_backend          = MOI.get(sampler, PCE.IBMFakeBackend())
    ansatz_factory            = MOI.get(sampler, PCE.Ansatz())
    channel                   = runtime_channel(MOI.get(sampler, PCE.Channel()))
    instance                  = runtime_instance(MOI.get(sampler, PCE.Instance()))
    initial_parameters        = MOI.get(sampler, PCE.InitialParameters())
    initial_parameter_source  = MOI.get(sampler, PCE.InitialParameterSource())
    record_initial_parameters = MOI.get(sampler, PCE.RecordInitialParameters())
    is_local                  = MOI.get(sampler, PCE.IsLocal())

    obs = pauli_correlation_observables(
        n;
        num_qubits = num_encoding_qubits_attr,
        parity_weight = parity_weight,
    )
    m = obs.num_qubits

    seed_simulator  = effective_seed(MOI.get(sampler, PCE.AerSeedSimulator()), sampler_seed, 1)
    seed_transpiler = effective_seed(MOI.get(sampler, PCE.TranspilerSeed()), sampler_seed, 2)
    aer_config = AerBackendConfig(
        method = MOI.get(sampler, PCE.AerBackendMethod()),
        precision = MOI.get(sampler, PCE.AerPrecision()),
        max_parallel_threads = MOI.get(sampler, PCE.AerMaxParallelThreads()),
        mps_omp_threads = MOI.get(sampler, PCE.AerMPSOmpThreads()),
        mps_truncation_threshold = MOI.get(sampler, PCE.AerMPSTruncationThreshold()),
        mps_max_bond_dimension = MOI.get(sampler, PCE.AerMPSMaxBondDimension()),
        mps_sample_measure_algorithm = MOI.get(sampler, PCE.AerMPSSampleMeasureAlgorithm()),
        seed_simulator = seed_simulator,
        seed_transpiler = seed_transpiler,
    )

    if n == 0
        if !isnothing(initial_parameters)
            validate_initial_parameters(initial_parameters, 0, "PCE")
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
        callback(Int[], final_num_reads)
        metadata = empty_metadata(
            "PCE",
            backend_name(backend),
            backend_selection.execution_mode;
            backend_version = backend_version(backend),
            backend_config = backend_selection.config,
            backend_config_source = backend_selection.source,
            number_of_reads = final_num_reads,
            final_number_of_reads = final_num_reads,
            optimizer_iterations = 0,
            optimizer_evaluations = 0,
            optimizer_number_of_reads = num_reads,
            seed_sampler = sampler_seed,
            seed_transpiler = aer_config.seed_transpiler,
        )
        metadata["pce"] = Dict{String, Any}(
            "experimental" => true,
            "num_variables" => 0,
            "num_encoding_qubits" => 0,
            "parity_weight" => parity_weight,
            "sharpness" => sharpness,
            "observables" => String[],
            "surrogate_loss" => 0.0,
            "raw_quantum_state" => Int[],
            "raw_quantum_value" => Float64(α * β),
            "polish_enabled" => polish_enabled,
            "polish_flips" => 0,
            "polish_passes" => 0,
            "polished_value" => Float64(α * β),
        )
        return metadata
    end

    ansatz = resolve_ansatz(ansatz_factory, m; reps = num_layers)
    param_names = qiskit_parameter_names(ansatz)
    expected_param_count = length(param_names)

    initial_parameter_values = if isnothing(initial_parameters)
        zeros(expected_param_count)
    else
        validate_initial_parameters(initial_parameters, expected_param_count, "PCE")
        Float64.(initial_parameters)
    end
    resolved_param_source = if isnothing(initial_parameter_source)
        isnothing(initial_parameters) ? "default_zero" : "user"
    else
        initial_parameter_source
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

    pass_manager = preset_pass_manager(
        backend;
        optimization_level = 3,
        seed_transpiler = aer_config.seed_transpiler,
    )
    ansatz_isa = pass_manager.run(ansatz)

    sign = sense == MOI.MIN_SENSE ? one(T) : -one(T)
    L_signed = Float64.(sign .* L)
    Q_signed = Float64.(sign .* Q)

    eval_counter = Ref(0)
    function evaluate_expectations(params::AbstractVector{<:Real}, shots_to_use::Integer)
        eval_counter[] += 1
        base_qc = bind_ansatz_parameters(ansatz, params)
        eval_seed = aer_config.seed_simulator === nothing ? nothing : Int(aer_config.seed_simulator) + eval_counter[] - 1
        if backend isa NativeLocalBackend
            return estimate_pauli_expectations(
                base_qc,
                obs.pauli_strings;
                shots = shots_to_use,
                seed = eval_seed,
                precision = something(aer_config.precision, "double"),
            )
        end

        basis_circuits = build_pce_basis_circuits(base_qc, obs.pauli_strings)
        basis_counts = Dict{Char,Any}()
        perm = ansatz_isa isa NativeAnsatz ? ansatz_isa.layout_permutation : nothing
        for basis in sort!(collect(keys(basis_circuits)))
            trans_b = pass_manager.run(basis_circuits[basis])
            perm_b = trans_b isa Qiskit.TranspileResult ? extract_layout_permutation(trans_b; num_variables = m) : perm
            if perm === nothing
                perm = perm_b
            end
            exec_b = trans_b isa Qiskit.TranspileResult ? trans_b.circuit : trans_b
            basis_counts[basis] = _run_circuit_with_hook(
                exec_b,
                backend;
                shots = shots_to_use,
            ).counts
        end
        return decode_pce_expectations_from_basis_counts(
            basis_counts,
            obs.pauli_strings;
            layout_permutation = perm,
        )
    end

    function pce_cost_function(params::AbstractVector{<:Real})
        evs = evaluate_expectations(params, num_reads)
        return surrogate_loss(evs, L_signed, Q_signed; sharpness = sharpness)
    end

    result = minimize_objective(
        pce_cost_function,
        initial_parameter_values;
        max_iter = max_iter,
    )

    optimizer_iterations = python_int_property(result, :nit)
    optimizer_evaluations = python_int_property(result, :nfev)
    optimized_parameter_values = python_float_vector(result.x)
    optimized_parameter_optimizer = optimizer_result_metadata(
        result;
        iterations = optimizer_iterations,
        function_evaluations = optimizer_evaluations,
    )

    final_expectations = evaluate_expectations(result.x, final_num_reads)
    final_surrogate_loss = surrogate_loss(final_expectations, L_signed, Q_signed; sharpness = sharpness)
    raw_quantum_state = decode_expectations(final_expectations)
    raw_quantum_value = Float64(QUBOTools.value(raw_quantum_state, L, Q, α, β))

    final_state = raw_quantum_state
    polished_value = raw_quantum_value
    polish_flips = 0
    polish_passes = 0

    if polish_enabled
        polish_res = bitflip_polish(
            raw_quantum_state,
            L,
            Q,
            α,
            β;
            sense = sense,
            max_passes = polish_max_passes,
        )
        final_state = polish_res.state
        polished_value = polish_res.value
        polish_flips = polish_res.flips
        polish_passes = polish_res.passes
    end

    callback(final_state, final_num_reads)

    metadata = empty_metadata(
        "PCE",
        backend_label,
        execution_mode;
        backend_version = backend_release,
        backend_config = backend_selection.config,
        backend_config_source = backend_selection.source,
        number_of_reads = final_num_reads,
        final_number_of_reads = final_num_reads,
        optimizer_iterations = optimizer_iterations,
        optimizer_evaluations = optimizer_evaluations,
        optimizer_number_of_reads = num_reads,
        seed_sampler = sampler_seed,
        seed_transpiler = aer_config.seed_transpiler,
        seed_optimizer = nothing,
        pass_manager_source = "default_preset",
        pass_manager_optimization_level = 3,
        initial_parameters = record_initial_parameters ? initial_parameter_values : nothing,
        initial_parameter_names = record_initial_parameters ? param_names : nothing,
        initial_parameter_source = record_initial_parameters ? resolved_param_source : nothing,
        optimized_parameters = optimized_parameter_values,
        optimized_parameter_names = param_names,
        optimized_parameter_optimizer = optimized_parameter_optimizer,
    )
    metadata["pce"] = Dict{String, Any}(
        "experimental" => true,
        "num_variables" => n,
        "num_encoding_qubits" => m,
        "num_encoding_qubits_source" => isnothing(num_encoding_qubits_attr) ? "auto" : "user",
        "parity_weight" => parity_weight,
        "sharpness" => sharpness,
        "observables" => obs.pauli_strings,
        "final_expectations" => final_expectations,
        "surrogate_loss" => final_surrogate_loss,
        "raw_quantum_state" => raw_quantum_state,
        "raw_quantum_value" => raw_quantum_value,
        "polish_enabled" => polish_enabled,
        "polish_flips" => polish_flips,
        "polish_passes" => polish_passes,
        "polished_value" => polished_value,
    )

    return metadata
end

end # module PCE
