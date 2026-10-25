module QAOA

using Qiskit
using QiskitIBMRuntime
using ..QiskitOpt:
    AerBackendConfig,
    IsingHamiltonian,
    NativeAnsatz,
    NativeFakeBackend,
    NativeLocalBackend,
    NativeOptimizerResult,
    NativePassManager,
    NativeRuntimeBackend,
    NativeRuntimeJob,
    PauliTerm,
    _TWO_QUBIT_GATE_NAMES,
    _is_py,
    _module_version,
    _py_exception_type_hook,
    _py_float,
    _py_hasattr,
    _py_int,
    _py_int_arg,
    _py_int_vec,
    _py_is_none,
    _py_len,
    _py_list_arg,
    _py_str,
    apply_layout_to_hamiltonian,
    backend_name,
    backend_version,
    bind_ansatz_parameters,
    bitvectors_to_counts,
    build_qaoa_circuit,
    circuit_depth,
    circuit_size,
    configured_execution_backend,
    default_local_backend,
    default_local_target,
    effective_seed,
    empty_metadata,
    estimate_expectation,
    evaluate_ising_bitstring,
    has_measurement_operations,
    iqm_backend,
    iqm_qiskit,
    iqm_server_url,
    iqm_token,
    iqm_tokens_file,
    measure_all,
    measure_all!,
    minimize_objective,
    native_transpile,
    operation_counts_dict,
    parse_sparse_observable,
    preset_pass_manager,
    python_int_property,
    qaoa_ansatz,
    qiskit_parameter_names,
    qiskit,
    qiskit_ibm_runtime,
    quadratic_program,
    qubo_to_ising,
    random_unit_values,
    remove_final_measurements,
    runtime_channel,
    runtime_service,
    runtime_instance,
    runtime_token,
    sample_bits,
    sample_bitvectors_from_circuit,
    sample_from_circuit,
    scipy,
    numpy,
    optimizer_result_metadata,
    python_float_vector,
    to_sparse_observable,
    transpile_layout_final_permutation,
    two_qubit_gate_count,
    validate_initial_parameters

using QUBO
MOI = QUBODrivers.MOI
Sample = QUBODrivers.Sample
SampleSet = QUBODrivers.SampleSet

const FIXED_ANGLE_SOURCE = "Wurtz-Lykov 3-regular tree fixed-angle table; arXiv:2107.00677"
const _HANDOFF_METADATA_SECTIONS = (
    "algorithm",
    "qaoa",
    "parameters",
    "variables",
    "measurement",
    "objective",
    "circuit",
)

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
        PassManagerFactory["pass_manager_factory"] = nothing
        CircuitExecutor["circuit_executor"]        = nothing
        Channel["channel"]::Union{String, Nothing} = nothing
        Instance["instance"]::Union{String, Nothing} = nothing
    end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true

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

function _fixed_angle_record(number_of_layers::Integer, family::Symbol)
    p = _check_number_of_layers(number_of_layers)
    if family != :wurtz_lykov_3regular_tree
        throw(ArgumentError("unsupported fixed-angle family: $(family)"))
    end
    if !haskey(_WURTZ_LYKOV_3REGULAR_TREE_ANGLES, p)
        throw(ArgumentError("fixed-angle family $(family) supports number_of_layers values 1 through 5"))
    end

    return _WURTZ_LYKOV_3REGULAR_TREE_ANGLES[p]
end

function _pass_manager_factory_signature_mismatch(err, factory)
    return err isa MethodError &&
        (
            (
                err.f === Core.kwcall &&
                length(err.args) >= 2 &&
                err.args[2] === factory
            ) ||
            err.f === factory
        )
end

function _try_pass_manager_factory(call::Function, factory)
    try
        return (pass_manager = call(), matched = true)
    catch err
        _pass_manager_factory_signature_mismatch(err, factory) || rethrow()
        return (pass_manager = nothing, matched = false)
    end
end

function _custom_pass_manager(
    factory,
    backend;
    optimization_level::Integer,
    seed_transpiler,
)
    attempts = (
        () -> factory(
            backend;
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        ),
        () -> factory(backend; optimization_level=optimization_level),
        () -> factory(backend; seed_transpiler=seed_transpiler),
        () -> factory(backend),
    )

    for attempt in attempts
        result = _try_pass_manager_factory(attempt, factory)
        result.matched && return result.pass_manager
    end

    throw(
        ArgumentError(
            "QAOA.PassManagerFactory must accept the selected backend and may " *
            "optionally accept optimization_level and seed_transpiler keywords",
        ),
    )
end

function _selected_pass_manager(
    backend;
    factory = nothing,
    optimization_level::Integer = 3,
    seed_transpiler = nothing,
)
    if isnothing(factory)
        return (
            pass_manager = preset_pass_manager(
                backend;
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
            ),
            source = "default_preset",
            optimization_level = optimization_level,
        )
    end

    return (
        pass_manager = _custom_pass_manager(
            factory,
            backend;
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        ),
        source = "custom_factory",
        optimization_level = optimization_level,
    )
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
    return _parameter_names(number_of_layers)
end

function parameter_names(cost_operator::Union{IsingHamiltonian,Qiskit.SparseObservable}; number_of_layers::Integer)
    return _parameter_names(number_of_layers)
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

function _fixed_parameter_sampler(sampler::QUBODrivers.AbstractSampler)
    return sampler
end

function _fixed_parameter_sampler(model::MOI.ModelLike)
    sampler = QAOA.Optimizer{Float64}()
    MOI.copy_to(sampler, model)
    return sampler
end

function _stored_number_of_layers(model)
    return MOI.get(model, QAOA.NumberOfLayers())
end

function _resolve_fixed_parameter_layers(source; reps, number_of_layers)
    if isnothing(reps) && isnothing(number_of_layers)
        return _check_number_of_layers(_stored_number_of_layers(source))
    elseif isnothing(number_of_layers)
        return _check_number_of_layers(reps)
    elseif isnothing(reps)
        return _check_number_of_layers(number_of_layers)
    end

    checked_reps = _check_number_of_layers(reps)
    checked_layers = _check_number_of_layers(number_of_layers)
    checked_reps == checked_layers ||
        throw(ArgumentError("reps and number_of_layers must match when both are provided"))
    return checked_layers
end

function _qiskit_ordered_qaoa_parameters(
    parameters::AbstractVector,
    number_of_layers::Integer,
    parameter_order::Symbol,
)
    p = _check_number_of_layers(number_of_layers)
    values = Float64.(parameters)
    expected_count = 2 * p
    if length(values) != expected_count
        throw(
            ArgumentError(
                "fixed_parameter_circuit parameters length $(length(values)) does not match " *
                "the expected QAOA parameter count $(expected_count)",
            ),
        )
    end

    if parameter_order in (:beta_then_gamma, :qiskit)
        return values
    elseif parameter_order == :gamma_then_beta
        return vcat(values[(p + 1):end], values[1:p])
    end

    throw(ArgumentError("unsupported QAOA parameter_order: $(parameter_order)"))
end

function _objective_sense_name(sense)
    sense == MOI.MIN_SENSE && return "min"
    sense == MOI.MAX_SENSE && return "max"
    return string(sense)
end

function _qiskit_minimization_sign(sense)
    return sense == MOI.MIN_SENSE ? 1 : -1
end

_circuit_operation_counts(circuit::Qiskit.QuantumCircuit) = operation_counts_dict(circuit)
_circuit_operation_counts(res::Qiskit.TranspileResult) = operation_counts_dict(res.circuit)

function _circuit_operation_counts(circuit)
    operations = Dict{String,Int}()
    counts = circuit.count_ops()
    for key in counts.keys()
        operations[_py_str(key)] = _py_int(counts[key])
    end
    return operations
end

function _python_qualified_type(object)
    if object isa Qiskit.QuantumCircuit
        return "Qiskit.QuantumCircuit"
    elseif object isa Qiskit.TranspileResult
        return "Qiskit.TranspileResult"
    elseif object isa Qiskit.TranspileLayout
        return "Qiskit.TranspileLayout"
    elseif object isa Qiskit.Target
        return "Qiskit.Target"
    end
    try
        class_object = getproperty(object, :__class__)
        module_name = _py_str(getproperty(class_object, :__module__))
        class_name = _py_str(getproperty(class_object, :__name__))
        return "$(module_name).$(class_name)"
    catch
        return string(typeof(object))
    end
end

function _python_len_or_nothing(object)
    try
        return Int(_py_len(object))
    catch
        return nothing
    end
end

function _python_string_vector(object)
    values = String[]
    try
        for item in object
            push!(values, _py_str(item))
        end
    catch
        return String[]
    end
    return sort!(unique(values))
end

function _python_object_summary(object; limit::Integer = 600)
    try
        text = _py_str(object)
        length(text) <= limit && return text
        return text[begin:nextind(text, 0, limit)] * "..."
    catch
        return nothing
    end
end

function _circuit_int_call(circuit, name::Symbol)
    if circuit isa Qiskit.QuantumCircuit || circuit isa Qiskit.TranspileResult
        name === :depth && return circuit_depth(circuit)
        name === :size && return circuit_size(circuit)
    end
    try
        return _py_int(getproperty(circuit, name)())
    catch
        return nothing
    end
end

function _instruction_operation_name(instruction)
    if instruction isa Qiskit.CircuitInstruction
        return instruction.name
    end
    try
        return _py_str(getproperty(getproperty(instruction, :operation), :name))
    catch
        return nothing
    end
end

function _instruction_width(instruction, field::Symbol)
    if instruction isa Qiskit.CircuitInstruction
        field === :qubits && return length(instruction.qubits)
        field === :clbits && return length(instruction.clbits)
    end
    try
        return _python_len_or_nothing(getproperty(instruction, field))
    catch
        return nothing
    end
end

function _two_qubit_operation_counts(res::Qiskit.TranspileResult)
    return _two_qubit_operation_counts(res.circuit)
end

function _two_qubit_operation_counts(circuit::Qiskit.QuantumCircuit)
    counts = Dict{String,Int}()
    for instruction in circuit.data
        length(instruction.qubits) == 2 || continue
        name = instruction.name
        name in ("barrier", "delay") && continue
        counts[name] = get(counts, name, 0) + 1
    end
    return counts
end

function _two_qubit_operation_counts(circuit)
    counts = Dict{String,Int}()
    try
        for instruction in circuit.data
            _instruction_width(instruction, :qubits) == 2 || continue
            name = _instruction_operation_name(instruction)
            isnothing(name) && continue
            name in ("barrier", "delay") && continue
            counts[name] = get(counts, name, 0) + 1
        end
    catch
        return Dict{String,Int}()
    end
    return counts
end

_measured_bit_count(res::Qiskit.TranspileResult) = _measured_bit_count(res.circuit)

function _measured_bit_count(circuit::Qiskit.QuantumCircuit)
    measured = 0
    for instruction in circuit.data
        instruction.name == "measure" || continue
        measured += length(instruction.clbits)
    end
    return measured
end

function _measured_bit_count(circuit)
    measured = 0
    try
        for instruction in circuit.data
            _instruction_operation_name(instruction) == "measure" || continue
            width = _instruction_width(instruction, :clbits)
            isnothing(width) || (measured += width)
        end
    catch
        return nothing
    end
    return measured
end

function _circuit_resource_metadata(res::Qiskit.TranspileResult)
    return _circuit_resource_metadata(res.circuit)
end

function _circuit_resource_metadata(circuit::Qiskit.QuantumCircuit)
    two_qubit_counts = _two_qubit_operation_counts(circuit)
    return Dict{String,Any}(
        "num_qubits" => circuit.num_qubits,
        "num_clbits" => circuit.num_clbits,
        "depth" => circuit_depth(circuit),
        "size" => circuit_size(circuit),
        "operations" => operation_counts_dict(circuit),
        "two_qubit_operation_counts" => two_qubit_counts,
        "two_qubit_operation_count" => sum(values(two_qubit_counts); init=0),
        "measured_bit_count" => _measured_bit_count(circuit),
    )
end

function _circuit_resource_metadata(circuit)
    two_qubit_counts = _two_qubit_operation_counts(circuit)
    return Dict{String,Any}(
        "num_qubits" => _py_int(circuit.num_qubits),
        "num_clbits" => _py_int(circuit.num_clbits),
        "depth" => _circuit_int_call(circuit, :depth),
        "size" => _circuit_int_call(circuit, :size),
        "operations" => _circuit_operation_counts(circuit),
        "two_qubit_operation_counts" => two_qubit_counts,
        "two_qubit_operation_count" => sum(values(two_qubit_counts); init=0),
        "measured_bit_count" => _measured_bit_count(circuit),
    )
end

function _target_resource_metadata(target::Qiskit.Target)
    return Dict{String,Any}(
        "type" => "Qiskit.Target",
        "num_qubits" => Int(target.num_qubits),
        "num_instructions" => Int(target.num_instructions),
    )
end

function _target_resource_metadata(target)
    metadata = Dict{String,Any}(
        "type" => _python_qualified_type(target),
        "num_qubits" => python_int_property(target, :num_qubits),
    )

    try
        operation_names = _python_string_vector(getproperty(target, :operation_names))
        isempty(operation_names) || (metadata["operation_names"] = operation_names)
    catch
    end

    try
        coupling_map = target.build_coupling_map()
        edges = coupling_map.get_edges()
        edge_count = _python_len_or_nothing(edges)
        isnothing(edge_count) || (metadata["coupling_edge_count"] = edge_count)
    catch
    end

    return metadata
end

function _backend_resource_metadata(backend, source::AbstractString)
    if backend isa Qiskit.Target
        return Dict{String,Any}(
            "source" => String(source),
            "type" => "Qiskit.Target",
            "name" => "qiskit_target",
            "version" => nothing,
            "num_qubits" => Int(backend.num_qubits),
            "target" => _target_resource_metadata(backend),
        )
    end
    metadata = Dict{String,Any}(
        "source" => String(source),
        "type" => _python_qualified_type(backend),
        "name" => backend_name(backend),
        "version" => backend_version(backend),
        "num_qubits" => python_int_property(backend, :num_qubits),
    )

    try
        metadata["basis_gates"] = _python_string_vector(backend.configuration().basis_gates)
    catch
    end

    try
        metadata["target"] = _target_resource_metadata(getproperty(backend, :target))
    catch
    end

    return metadata
end

function _transpiled_layout_metadata(res::Qiskit.TranspileResult)
    perm = transpile_layout_final_permutation(res.layout)
    return Dict{String,Any}(
        "available" => true,
        "type" => "Qiskit.TranspileLayout",
        "summary" => sprint(show, res.layout),
        "final_layout" => string(perm),
    )
end

function _transpiled_layout_metadata(circuit::Qiskit.QuantumCircuit)
    return Dict{String,Any}("available" => false)
end

function _transpiled_layout_metadata(circuit)
    try
        layout = getproperty(circuit, :layout)
        summary = Dict{String,Any}(
            "available" => true,
            "type" => _python_qualified_type(layout),
            "summary" => _python_object_summary(layout),
        )
        for field in (:initial_layout, :final_layout, :input_qubit_mapping)
            value = _python_object_summary(getproperty(layout, field))
            isnothing(value) || (summary[string(field)] = value)
        end
        return summary
    catch
        return Dict{String,Any}("available" => false)
    end
end

function _fixed_parameter_metadata(
    sampler::QUBODrivers.AbstractSampler,
    circuit,
    number_of_layers::Integer,
    parameter_order::Symbol,
    parameter_names::AbstractVector{<:AbstractString},
    parameter_values::AbstractVector{<:Real},
    measure::Bool,
)
    n, _, _, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())
    n_qubits = circuit isa Qiskit.QuantumCircuit ? circuit.num_qubits : _py_int(circuit.num_qubits)
    n_clbits = circuit isa Qiskit.QuantumCircuit ? circuit.num_clbits : _py_int(circuit.num_clbits)
    depth_val = circuit isa Qiskit.QuantumCircuit ? circuit_depth(circuit) : _py_int(circuit.depth())

    return Dict{String,Any}(
        "algorithm" => Dict{String,Any}(
            "name" => "QAOA",
            "mode" => "fixed_parameter_circuit",
        ),
        "qaoa" => Dict{String,Any}(
            "number_of_layers" => number_of_layers,
            "cost_layer" => "qiskit.circuit.library.QAOAAnsatz cost_operator",
            "mixer" => "qiskit.circuit.library.QAOAAnsatz default mixer",
        ),
        "parameters" => Dict{String,Any}(
            "input_order" => String(parameter_order),
            "qiskit_order" => "beta_then_gamma",
            "parameter_names" => String.(parameter_names),
            "values_order" => "qiskit_order",
            "values_aligned_to" => "parameter_names",
            "values" => Float64.(parameter_values),
        ),
        "variables" => Dict{String,Any}(
            "count" => n,
            "order" => string.(1:n),
            "qubit_order" => "variable 1 maps to Qiskit qubit 0",
        ),
        "measurement" => Dict{String,Any}(
            "enabled" => measure,
            "classical_bit_order" => measure ? "variable 1 maps to classical bit 0" : nothing,
            "qiskit_count_key_order" => "Qiskit count keys print classical bits from highest to lowest index",
            "variable_bit_order" => "QAOA.count_key_bits(key) returns bits in variable order",
        ),
        "objective" => Dict{String,Any}(
            "sense" => _objective_sense_name(sense),
            "scale" => α,
            "offset" => β,
            "qiskit_minimization_sign" => _qiskit_minimization_sign(sense),
            "value_convention" => "QUBOTools.value(bits, linear, quadratic, scale, offset)",
        ),
        "circuit" => Dict{String,Any}(
            "num_qubits" => n_qubits,
            "num_clbits" => n_clbits,
            "depth" => depth_val,
            "operations" => _circuit_operation_counts(circuit),
        ),
    )
end

function _check_runtime_backend(backend::AbstractString)
    backend_name = strip(String(backend))
    isempty(backend_name) && throw(ArgumentError("backend must be a non-empty IBM backend name"))
    return backend_name
end

function _check_runtime_shots(shots::Integer)
    shots >= 1 || throw(ArgumentError("shots must be at least 1"))
    return Int(shots)
end

function _check_optimization_level(optimization_level::Integer)
    0 <= optimization_level <= 3 ||
        throw(ArgumentError("optimization_level must be between 0 and 3"))
    return Int(optimization_level)
end

function _check_transpiler_seed(seed::Union{Integer,Nothing})
    isnothing(seed) && return nothing
    return Int(seed)
end

function _check_measured_circuit(circuit::Qiskit.QuantumCircuit)
    if circuit.num_clbits >= 1 && has_measurement_operations(circuit)
        return nothing
    end
    throw(
        ArgumentError(
            "ibm_runtime_handoff expects a measured circuit; call " *
            "QAOA.fixed_parameter_circuit(...; measure=true)",
        ),
    )
end

function _check_measured_circuit(res::Qiskit.TranspileResult)
    return _check_measured_circuit(res.circuit)
end

function _check_measured_circuit(circuit)
    try
        _py_int(circuit.num_clbits) >= 1 && return nothing
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("circuit must expose Qiskit num_clbits"))
    end

    throw(
        ArgumentError(
            "ibm_runtime_handoff expects a measured circuit; call " *
            "QAOA.fixed_parameter_circuit(...; measure=true)",
        ),
    )
end

function _safe_fixed_parameter_metadata(fixed_metadata)
    isnothing(fixed_metadata) && return Dict{String,Any}()

    metadata = Dict{String,Any}()
    for key in _HANDOFF_METADATA_SECTIONS
        haskey(fixed_metadata, key) || continue
        metadata[key] = deepcopy(fixed_metadata[key])
    end
    return metadata
end

function _julia_package_version()
    try
        version = Base.pkgversion(parentmodule(@__MODULE__))
        return isnothing(version) ? nothing : string(version)
    catch
        return nothing
    end
end

function _python_package_version(lazy_module)
    try
        loaded = lazy_module()
        return _py_str(loaded.__version__)
    catch
        return nothing
    end
end

function _runtime_handoff_package_versions()
    return Dict{String,Any}(
        "QiskitOpt" => _julia_package_version(),
        "qiskit" => something(_module_version(Qiskit), "0.6.0"),
        "qiskit_ibm_runtime" => something(_module_version(QiskitIBMRuntime), "0.3.0"),
    )
end

function _runtime_handoff_metadata(;
    fixed_metadata,
    backend::AbstractString,
    shots::Integer,
    transpiler_seed::Union{Integer,Nothing},
    optimization_level::Integer,
    dry_run::Bool,
    channel::Union{Nothing,AbstractString},
    instance::Union{Nothing,AbstractString},
)
    resolved_channel = runtime_channel(channel)
    resolved_instance = runtime_instance(instance)

    return Dict{String,Any}(
        "runtime_handoff" => Dict{String,Any}(
            "mode" => dry_run ? "dry_run" : "live",
            "status" => dry_run ? "dry_run" : "preparing_submission",
            "backend" => String(backend),
            "shots" => Int(shots),
            "optimization_level" => Int(optimization_level),
            "transpiler_seed" => transpiler_seed,
            "channel" => resolved_channel,
            "instance_configured" => !isnothing(resolved_instance),
            "credentials_recorded" => false,
            "credential_fields_omitted" => ["token", "instance", "account_file", "crn"],
            "instance_hint" =>
                "QISKIT_IBM_INSTANCE or QAOA.Instance() may be required even when a token is present",
        ),
        "fixed_parameter_circuit" => _safe_fixed_parameter_metadata(fixed_metadata),
        "count_scoring" => Dict{String,Any}(
            "count_key_order" =>
                "Qiskit count keys print classical bits from highest to lowest index",
            "bit_conversion" => "QAOA.count_key_bits",
            "value_convention" =>
                "QUBOTools.value(bits, linear, quadratic, scale, offset)",
        ),
        "packages" => _runtime_handoff_package_versions(),
    )
end

function _transpile_for_runtime(circuit, backend; optimization_level::Integer, transpiler_seed)
    if circuit isa Qiskit.QuantumCircuit
        pm = preset_pass_manager(
            backend;
            optimization_level=optimization_level,
            seed_transpiler=transpiler_seed,
        )
        res = pm.run(circuit)
        return res isa Qiskit.TranspileResult ? res.circuit : res
    end

    if isnothing(transpiler_seed)
        return qiskit().transpile(
            circuit;
            backend=backend,
            optimization_level=optimization_level,
        )
    end

    return qiskit().transpile(
        circuit;
        backend=backend,
        seed_transpiler=transpiler_seed,
        optimization_level=optimization_level,
    )
end

function _python_call_string(object, name::Symbol)
    try
        value = getproperty(object, name)
        try
            value = value()
        catch
        end
        value isa AbstractString && return String(value)
        value === nothing && return nothing
        return _py_str(value)
    catch
        return nothing
    end
end

function _redact_runtime_message(message::AbstractString, secrets)
    redacted = String(message)
    for secret in secrets
        isnothing(secret) && continue
        secret_text = String(secret)
        isempty(secret_text) && continue
        redacted = replace(redacted, secret_text => "[redacted]")
    end

    redacted = replace(redacted, r"(?i)(token\s*[=:]\s*)[^\s,;]+" => s"\1[redacted]")
    redacted = replace(redacted, r"(?i)(tokens[_ -]?file\s*[=:]\s*)[^\s,;]+" => s"\1[redacted]")
    redacted = replace(redacted, r"(?i)(instance\s*[=:]\s*)[^\s,;]+" => s"\1[redacted]")
    redacted = replace(redacted, r"(?i)(account[_ -]?file\s*[=:]\s*)[^\s,;]+" => s"\1[redacted]")
    redacted = replace(redacted, r"(?i)(bearer\s+)[^\s,;]+" => s"\1[redacted]")
    redacted = replace(redacted, r"(?i)\bcrn:[^\s,;]+" => "[redacted]")
    return redacted
end

function _resource_audit_package_versions()
    return Dict{String,Any}(
        "QiskitOpt" => _julia_package_version(),
        "qiskit" => something(_module_version(Qiskit), "0.6.0"),
    )
end

function _resource_audit_exception_type(err)
    py_type = _py_exception_type_hook[](err)
    !isnothing(py_type) && return py_type
    return string(nameof(typeof(err)))
end

function _resource_audit_failure_metadata(err)
    return Dict{String,Any}(
        "exception_type" => _resource_audit_exception_type(err),
        "message" => _redact_runtime_message(
            sprint(showerror, err),
            (runtime_token(), runtime_instance()),
        ),
    )
end

function _copy_resource_audit_sections!(metadata, fixed_metadata)
    safe_metadata = _safe_fixed_parameter_metadata(fixed_metadata)
    for section in ("qaoa", "parameters", "variables", "measurement", "objective")
        haskey(safe_metadata, section) || continue
        metadata[section] = safe_metadata[section]
    end
    isempty(safe_metadata) || (metadata["fixed_parameter_circuit"] = safe_metadata)
    return metadata
end

function _resource_audit_backend(backend, transpile::Bool)
    transpile || return (backend=nothing, source="not_requested")
    isnothing(backend) && return (backend=default_local_backend(), source="default_local_backend")
    return (backend=backend, source="user_backend")
end

function _resource_audit_base_metadata(
    circuit;
    fixed_metadata,
    backend,
    backend_source::AbstractString,
    optimization_level::Integer,
    transpiler_seed,
)
    metadata = Dict{String,Any}(
        "algorithm" => Dict{String,Any}(
            "name" => "QAOA",
            "mode" => "resource_audit",
        ),
        "untranspiled_circuit" => _circuit_resource_metadata(circuit),
        "transpilation" => Dict{String,Any}(
            "optimization_level" => optimization_level,
            "transpiler_seed" => transpiler_seed,
        ),
        "packages" => _resource_audit_package_versions(),
    )

    if isnothing(backend)
        metadata["backend"] = Dict{String,Any}("source" => String(backend_source))
    else
        metadata["backend"] = _backend_resource_metadata(backend, backend_source)
    end

    _copy_resource_audit_sections!(metadata, fixed_metadata)
    return metadata
end

function _mark_resource_audit_skipped!(metadata, reason::AbstractString)
    metadata["status"] = "skipped_transpilation"
    transpilation = metadata["transpilation"]
    transpilation["status"] = "skipped"
    transpilation["reason"] = String(reason)
    return metadata
end

function _record_resource_audit_pass_manager!(metadata, pass_manager_selection)
    transpilation = metadata["transpilation"]
    transpilation["pass_manager"] = Dict{String,Any}(
        "source" => pass_manager_selection.source,
        "optimization_level" => pass_manager_selection.optimization_level,
    )
    return metadata
end

function _mark_resource_audit_success!(metadata, transpiled_circuit)
    metadata["status"] = "success"
    transpilation = metadata["transpilation"]
    transpilation["status"] = "success"
    metadata["transpiled_circuit"] = _circuit_resource_metadata(transpiled_circuit)
    metadata["transpiled_circuit"]["layout"] = _transpiled_layout_metadata(transpiled_circuit)
    return metadata
end

function _mark_resource_audit_failed!(metadata, err)
    metadata["status"] = "failed_transpilation"
    transpilation = metadata["transpilation"]
    transpilation["status"] = "failed"
    transpilation["failure"] = _resource_audit_failure_metadata(err)
    return metadata
end

"""
    QAOA.resource_audit(source; parameters, reps=nothing, number_of_layers=nothing, parameter_order=:beta_then_gamma, measure=true, backend=nothing, transpile=true, optimization_level=3, transpiler_seed=73001, pass_manager_factory=nothing, return_transpiled_circuit=false)

Build a fixed-parameter QAOA circuit and return dry-run resource metadata
without submitting a job or touching IBM account state.

When `transpile=true` and `backend` is not provided, the audit uses the
credential-free local Aer backend. Pass a Qiskit backend object, such as a fake
backend or an `AerSimulator.from_backend(...)` instance, to audit against a
specific target. Set `transpile=false` to record only the untranspiled circuit
metadata.
"""
function resource_audit(
    n::Integer,
    L::AbstractDict,
    Q::AbstractDict;
    p::Union{Integer,Nothing} = nothing,
    reps::Union{Integer,Nothing} = nothing,
    number_of_layers::Union{Integer,Nothing} = nothing,
    gamma::Union{AbstractVector{<:Real},Nothing} = nothing,
    beta::Union{AbstractVector{<:Real},Nothing} = nothing,
    parameters::Union{AbstractVector{<:Real},Nothing} = nothing,
    parameter_order::Symbol = :beta_then_gamma,
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
    measure::Bool = false,
    target::Union{Nothing,Qiskit.Target} = nothing,
    backend = nothing,
    pass_manager = nothing,
    pass_manager_factory = nothing,
    optimization_level::Integer = 3,
    transpiler_seed::Union{Integer,Nothing} = 73001,
)
    layers = if !isnothing(p)
        _check_number_of_layers(p)
    elseif !isnothing(reps)
        _check_number_of_layers(reps)
    elseif !isnothing(number_of_layers)
        _check_number_of_layers(number_of_layers)
    else
        !isnothing(gamma) ? _check_number_of_layers(length(gamma)) : 1
    end
    qc = fixed_parameter_circuit(
        n,
        L,
        Q;
        p = layers,
        gamma = gamma,
        beta = beta,
        parameters = parameters,
        parameter_order = parameter_order,
        sense = sense,
        measure = measure,
    )
    transpiled = transpile_for_execution(
        qc;
        target = target,
        backend = backend,
        pass_manager = pass_manager,
        pass_manager_factory = pass_manager_factory,
        optimization_level = optimization_level,
        transpiler_seed = transpiler_seed,
    )
    t_qc = transpiled isa Qiskit.TranspileResult ? transpiled.circuit : transpiled
    ops = operation_counts_dict(transpiled)
    perm = extract_layout_permutation(transpiled; num_qubits = Int(n))
    return (
        num_qubits = t_qc.num_qubits,
        reps = layers,
        depth = circuit_depth(transpiled),
        size = circuit_size(transpiled),
        op_counts = ops,
        two_qubit_gates = two_qubit_gate_count(ops),
        parameter_count = 2 * layers,
        layout_permutation = perm,
        transpiled_circuit = t_qc,
        transpile_result = transpiled,
    )
end

function resource_audit(
    source::MOI.ModelLike;
    parameters,
    reps = nothing,
    number_of_layers = nothing,
    parameter_order::Symbol = :beta_then_gamma,
    measure::Bool = true,
    kwargs...
)
    circuit, fixed_metadata = fixed_parameter_circuit(
        source;
        parameters=parameters,
        reps=reps,
        number_of_layers=number_of_layers,
        parameter_order=parameter_order,
        measure=measure,
    )
    return resource_audit(circuit; fixed_metadata=fixed_metadata, kwargs...)
end

"""
    QAOA.resource_audit(circuit; fixed_metadata=nothing, backend=nothing, transpile=true, optimization_level=3, transpiler_seed=73001, pass_manager_factory=nothing, return_transpiled_circuit=false)

Return dry-run resource metadata for an existing Qiskit circuit.

The result is a named tuple with `metadata` and `transpiled_circuit`. Failure to
transpile is reported as sanitized structured metadata with
`metadata["transpilation"]["status"] == "failed"`; no job is submitted.
"""
function resource_audit(
    circuit;
    fixed_metadata = nothing,
    backend = nothing,
    transpile::Bool = true,
    optimization_level::Integer = 3,
    transpiler_seed::Union{Integer,Nothing} = 73001,
    pass_manager_factory = nothing,
    return_transpiled_circuit::Bool = false,
)
    checked_optimization_level = _check_optimization_level(optimization_level)
    checked_seed = _check_transpiler_seed(transpiler_seed)
    backend_selection = _resource_audit_backend(backend, transpile)

    metadata = _resource_audit_base_metadata(
        circuit;
        fixed_metadata=fixed_metadata,
        backend=backend_selection.backend,
        backend_source=backend_selection.source,
        optimization_level=checked_optimization_level,
        transpiler_seed=checked_seed,
    )

    if !transpile
        _mark_resource_audit_skipped!(metadata, "transpile=false")
        return (metadata=metadata, transpiled_circuit=nothing)
    end

    transpiled_circuit = nothing
    try
        pass_manager_selection = if circuit isa Qiskit.QuantumCircuit && isnothing(pass_manager_factory) &&
            (backend_selection.backend === nothing || backend_selection.backend isa Qiskit.Target || backend_selection.source == "default_local_backend")
            t_arg = backend_selection.backend isa Qiskit.Target ? backend_selection.backend : nothing
            (
                pass_manager = NativePassManager(t_arg, checked_optimization_level, checked_seed),
                source = "default_preset",
                optimization_level = checked_optimization_level,
            )
        else
            _selected_pass_manager(
                backend_selection.backend;
                factory=pass_manager_factory,
                optimization_level=checked_optimization_level,
                seed_transpiler=checked_seed,
            )
        end
        _record_resource_audit_pass_manager!(metadata, pass_manager_selection)
        transpiled_circuit = pass_manager_selection.pass_manager.run(circuit)
        _mark_resource_audit_success!(
            metadata,
            transpiled_circuit,
        )
    catch err
        _mark_resource_audit_failed!(metadata, err)
    end

    return (
        metadata=metadata,
        transpiled_circuit=return_transpiled_circuit ? transpiled_circuit : nothing,
    )
end

"""
    QAOA.RuntimeHandoffError

Failure type thrown by `QAOA.ibm_runtime_handoff(...; dry_run=false)` when the
live Runtime setup, transpilation, or submission path fails before a successful
job is returned.

Catch this exception when a workflow needs to persist the sanitized failure
metadata. The `metadata` field contains the same handoff metadata shape returned
by dry runs, with `metadata["runtime_handoff"]["status"] ==
"failed_before_submission"` and a sanitized failure message. The `cause` field
contains the original exception object.
"""
struct RuntimeHandoffError <: Exception
    metadata::Dict{String,Any}
    cause::Any
end

function Base.showerror(io::IO, err::RuntimeHandoffError)
    print(io, "IBM Runtime handoff failed before a successful job submission")
    failure = get(get(err.metadata, "runtime_handoff", Dict{String,Any}()), "failure", nothing)
    if failure isa AbstractDict && haskey(failure, "message")
        print(io, ": ", failure["message"])
    else
        print(io, ": ")
        showerror(io, err.cause)
    end
end

"""
    QAOA.ibm_runtime_handoff(circuit; fixed_metadata=nothing, backend, shots, dry_run=true, transpiler_seed=nothing, optimization_level=3, channel=nothing, instance=nothing)

Prepare a fixed-parameter QAOA circuit for IBM Runtime `SamplerV2`.

The default `dry_run=true` path does not contact IBM Runtime. It returns
sanitized metadata describing the intended backend, shots, transpiler seed,
fixed-circuit parameter metadata, count-key scoring convention, package
versions, and whether an instance selector was configured. Token values,
instance/CRN values, and account file paths are never recorded.

Set `dry_run=false` to resolve the backend through `QiskitRuntimeService`,
transpile the measured circuit, and submit it to `qiskit_ibm_runtime.SamplerV2`.
Live submission reads credentials from normal Qiskit Runtime configuration and
returns the submitted job object plus the same sanitized metadata with job
status fields populated.

If live setup, transpilation, or submission fails before a successful job is
returned, this throws `QAOA.RuntimeHandoffError`. Catch it and read
`err.metadata` to persist sanitized failure metadata before rethrowing or
stopping the workflow. Failure-message redaction is best effort: it removes
QiskitOpt-resolved token and instance values plus obvious `token=...`,
`instance=...`, `account_file=...`, and `crn:...` fragments from the captured
message, but callers should still keep Runtime credentials outside project
artifacts and avoid logging raw upstream exceptions.
"""
function ibm_runtime_handoff(
    circuit;
    fixed_metadata = nothing,
    backend::AbstractString,
    shots::Integer,
    dry_run::Bool = true,
    transpiler_seed::Union{Integer,Nothing} = nothing,
    optimization_level::Integer = 3,
    channel::Union{Nothing,AbstractString} = nothing,
    instance::Union{Nothing,AbstractString} = nothing,
)
    _check_measured_circuit(circuit)
    backend_label = _check_runtime_backend(backend)
    shot_count = _check_runtime_shots(shots)
    checked_seed = _check_transpiler_seed(transpiler_seed)
    checked_optimization_level = _check_optimization_level(optimization_level)

    metadata = _runtime_handoff_metadata(
        fixed_metadata=fixed_metadata,
        backend=backend_label,
        shots=shot_count,
        transpiler_seed=checked_seed,
        optimization_level=checked_optimization_level,
        dry_run=dry_run,
        channel=channel,
        instance=instance,
    )

    if dry_run
        return (
            metadata=metadata,
            transpiled_circuit=nothing,
            job=nothing,
        )
    end

    service = runtime_backend = transpiled_circuit = job = nothing
    try
        service = runtime_service(channel=channel, instance=instance)
        runtime_backend = service.backend(backend_label)
        transpiled_circuit = _transpile_for_runtime(
            circuit,
            runtime_backend;
            optimization_level=checked_optimization_level,
            transpiler_seed=checked_seed,
        )
        exec_qc = transpiled_circuit isa Qiskit.TranspileResult ? transpiled_circuit.circuit : transpiled_circuit
        job = if runtime_backend isa NativeRuntimeBackend
            runtime_backend.submit_sampler(exec_qc, shot_count)
        elseif runtime_backend isa QiskitIBMRuntime.Backend && service isa QiskitIBMRuntime.Service
            NativeRuntimeJob(service, QiskitIBMRuntime.run_sampler_job(service, runtime_backend, exec_qc, shot_count))
        elseif !_is_py(runtime_backend) && hasproperty(runtime_backend, :submit_sampler)
            runtime_backend.submit_sampler(exec_qc, shot_count)
        elseif !_is_py(runtime_backend) && hasproperty(runtime_backend, :run)
            runtime_backend.run(exec_qc; shots=shot_count)
        else
            sampler = qiskit_ibm_runtime().SamplerV2(mode=runtime_backend)
            sampler.run(_py_list_arg([exec_qc]), shots=_py_int_arg(shot_count))
        end
    catch err
        handoff = metadata["runtime_handoff"]
        handoff["status"] = "failed_before_submission"
        handoff["failure"] = Dict{String,Any}(
            "message" => _redact_runtime_message(
                sprint(showerror, err),
                (runtime_token(), runtime_instance(instance)),
            ),
            "instance_hint" =>
                "QISKIT_IBM_INSTANCE or QAOA.Instance() may be required when Runtime cannot auto-resolve an account instance",
        )
        throw(RuntimeHandoffError(metadata, err))
    end

    handoff = metadata["runtime_handoff"]
    handoff["status"] = "submitted"
    handoff["job"] = Dict{String,Any}(
        "id" => _python_call_string(job, :job_id),
        "status" => _python_call_string(job, :status),
    )

    return (
        metadata=metadata,
        transpiled_circuit=transpiled_circuit,
        job=job,
    )
end

function _check_layout_permutation(
    layout_permutation::AbstractVector{<:Integer},
    num_bits::Union{Integer,Nothing} = nothing,
)
    n = length(layout_permutation)
    n >= 0 || throw(ArgumentError("layout_permutation length must be non-negative"))
    perm = Int[Int(idx) for idx in layout_permutation]
    all(idx -> idx >= 1, perm) ||
        throw(ArgumentError("layout_permutation entries must be 1-based positive indices"))
    length(unique(perm)) == n ||
        throw(ArgumentError("layout_permutation must not contain duplicate classical-bit indices"))
    if !isnothing(num_bits)
        all(idx -> idx <= num_bits, perm) ||
            throw(
                ArgumentError(
                    "layout_permutation indices $(perm) exceed available classical bits ($(num_bits))",
                ),
            )
    end
    return perm
end

"""
    QAOA.extract_layout_permutation(circuit; num_variables=nothing)

Inspect a Qiskit circuit (optionally transpiled) and return the 1-based classical
bit permutation `perm` such that `perm[i]` is the 1-based measured classical-bit
index (in reversed Qiskit count-key order) corresponding to logical QUBO
variable `i`.

Combines `circuit.layout.final_index_layout()` (virtual qubit -> physical qubit)
with the circuit's `measure` instructions (physical qubit -> classical bit).
Falls back to `collect(1:n)` when no layout or measurement remapping is present.
"""
function extract_layout_permutation(
    layout::Qiskit.TranspileLayout;
    num_variables::Union{Integer,Nothing} = nothing,
    num_qubits::Union{Integer,Nothing} = nothing,
)
    raw_perm = transpile_layout_final_permutation(layout)
    req_n = !isnothing(num_variables) ? Int(num_variables) : (!isnothing(num_qubits) ? Int(num_qubits) : length(raw_perm))
    req_n <= 0 && return Int[]
    if length(raw_perm) >= req_n
        sub = raw_perm[1:req_n]
        if length(unique(sub)) == req_n
            return sub
        end
    end
    return collect(1:req_n)
end

function extract_layout_permutation(
    res::Qiskit.TranspileResult;
    num_variables::Union{Integer,Nothing} = nothing,
    num_qubits::Union{Integer,Nothing} = nothing,
)
    phys_perm = transpile_layout_final_permutation(res.layout)
    qc = res.circuit
    req_n = if !isnothing(num_variables)
        Int(num_variables)
    elseif !isnothing(num_qubits)
        Int(num_qubits)
    elseif !isempty(phys_perm)
        length(phys_perm)
    else
        qc.num_clbits > 0 ? qc.num_clbits : qc.num_qubits
    end
    req_n <= 0 && return Int[]

    phys_to_clbit = Dict{Int,Int}()
    offset = qc.offset
    for inst in qc.data
        inst.name == "measure" || continue
        for (q_raw, c_raw) in zip(inst.qubits, inst.clbits)
            phys_to_clbit[q_raw - offset + 1] = c_raw - offset + 1
        end
    end

    if isempty(phys_to_clbit)
        if length(phys_perm) >= req_n && length(unique(phys_perm[1:req_n])) == req_n
            return phys_perm[1:req_n]
        end
        return collect(1:req_n)
    end

    perm = Vector{Int}(undef, req_n)
    for i in 1:req_n
        phys_q = i <= length(phys_perm) ? phys_perm[i] : i
        perm[i] = get(phys_to_clbit, phys_q, phys_q)
    end
    if length(unique(perm)) != req_n
        return collect(1:req_n)
    end
    return perm
end

function extract_layout_permutation(
    qc::Qiskit.QuantumCircuit;
    num_variables::Union{Integer,Nothing} = nothing,
    num_qubits::Union{Integer,Nothing} = nothing,
)
    req_n = if !isnothing(num_variables)
        Int(num_variables)
    elseif !isnothing(num_qubits)
        Int(num_qubits)
    else
        qc.num_clbits > 0 ? qc.num_clbits : qc.num_qubits
    end
    req_n <= 0 && return Int[]

    phys_to_clbit = Dict{Int,Int}()
    offset = qc.offset
    for inst in qc.data
        inst.name == "measure" || continue
        for (q_raw, c_raw) in zip(inst.qubits, inst.clbits)
            phys_to_clbit[q_raw - offset + 1] = c_raw - offset + 1
        end
    end
    if isempty(phys_to_clbit)
        return collect(1:req_n)
    end
    perm = Int[get(phys_to_clbit, i, i) for i in 1:req_n]
    return length(unique(perm)) == req_n ? perm : collect(1:req_n)
end

function extract_layout_permutation(
    circuit;
    num_variables::Union{Integer,Nothing} = nothing,
    num_qubits::Union{Integer,Nothing} = nothing,
)
    req_var = !isnothing(num_variables) ? num_variables : num_qubits
    n = if !isnothing(req_var)
        Int(req_var)
    else
        try
            _py_int(circuit.num_clbits)
        catch
            0
        end
    end
    n <= 0 && return Int[]

    phys_to_clbit = Dict{Int,Int}()
    try
        for instruction in circuit.data
            _instruction_operation_name(instruction) == "measure" || continue
            qubits = getproperty(instruction, :qubits)
            clbits = getproperty(instruction, :clbits)
            m = min(Int(_py_len(qubits)), Int(_py_len(clbits)))
            for idx in 0:(m - 1)
                q_idx = _py_int(circuit.find_bit(qubits[idx]).index)
                c_idx = _py_int(circuit.find_bit(clbits[idx]).index)
                phys_to_clbit[q_idx] = c_idx
            end
        end
    catch
    end

    virt_to_phys = nothing
    try
        if !_is_py(circuit) && hasproperty(circuit, :layout)
            layout = getproperty(circuit, :layout)
            if layout isa Qiskit.TranspileLayout
                p1 = transpile_layout_final_permutation(layout)
                virt_to_phys = Int[q - 1 for q in p1]
            elseif hasproperty(layout, :final_index_layout)
                fl = getproperty(layout, :final_index_layout)
                raw = fl isa Function ? fl() : fl
                virt_to_phys = Int[Int(x) for x in raw]
            end
        elseif _py_hasattr(circuit, "layout")
            layout = getproperty(circuit, :layout)
            if !_py_is_none(layout) && _py_hasattr(layout, "final_index_layout")
                virt_to_phys = _py_int_vec(layout.final_index_layout())
            end
        end
    catch
    end

    perm = Vector{Int}(undef, n)
    for i in 1:n
        phys_q = if !isnothing(virt_to_phys) && i <= length(virt_to_phys)
            virt_to_phys[i]
        else
            i - 1
        end
        c_idx = get(phys_to_clbit, phys_q, isempty(phys_to_clbit) ? phys_q : (i - 1))
        perm[i] = c_idx + 1
    end

    if length(unique(perm)) != n
        return collect(1:n)
    end
    return perm
end

"""
    QAOA.transpile_for_execution(circuit; target=nothing, backend=nothing, pass_manager=nothing, pass_manager_factory=nothing, optimization_level=3, transpiler_seed=nothing)

Transpile `circuit` against a native `Qiskit.Target`, backend, or custom pass manager,
returning a `Qiskit.TranspileResult` (or the pass manager output).
"""
function transpile_for_execution(
    circuit::Qiskit.QuantumCircuit;
    target::Union{Nothing,Qiskit.Target} = nothing,
    backend = nothing,
    pass_manager = nothing,
    pass_manager_factory = nothing,
    optimization_level::Integer = 3,
    transpiler_seed::Union{Nothing,Integer} = nothing,
)
    checked_level = _check_optimization_level(optimization_level)
    checked_seed = _check_transpiler_seed(transpiler_seed)
    if !isnothing(pass_manager)
        if pass_manager isa Qiskit.Target
            return native_transpile(
                circuit;
                target = pass_manager,
                optimization_level = checked_level,
                seed = checked_seed,
            )
        elseif hasproperty(pass_manager, :run)
            return pass_manager.run(circuit)
        elseif pass_manager isa Function
            out = pass_manager(circuit)
            if out isa Qiskit.Target
                return native_transpile(
                    circuit;
                    target = out,
                    optimization_level = checked_level,
                    seed = checked_seed,
                )
            end
            return out
        end
    end
    if !isnothing(pass_manager_factory)
        pm_sel = _selected_pass_manager(
            something(backend, target, default_local_target(circuit.num_qubits));
            factory = pass_manager_factory,
            optimization_level = checked_level,
            seed_transpiler = checked_seed,
        )
        return pm_sel.pass_manager.run(circuit)
    end
    resolved_target = if !isnothing(target)
        target
    elseif backend isa Qiskit.Target
        backend
    elseif !isnothing(backend) && !_is_py(backend) && hasproperty(backend, :target) && getproperty(backend, :target) isa Qiskit.Target
        getproperty(backend, :target)
    else
        default_local_target(circuit.num_qubits)
    end
    return native_transpile(
        circuit;
        target = resolved_target,
        optimization_level = checked_level,
        seed = checked_seed,
    )
end

"""
    QAOA.count_key_bits(key; layout_permutation=nothing)

Convert a Qiskit or IQM count key into QUBO variable order `[x1, x2, ...]`.

Qiskit count dictionaries print classical bits from the highest classical-bit
index down to zero. This helper reverses the rendered key and, when
`layout_permutation` is provided, maps logical variable `i` from the
`layout_permutation[i]`-th measured classical bit (1-based after reversal).
"""
function count_key_bits(key; layout_permutation = nothing)
    raw_bits = sample_bits(key)
    isnothing(layout_permutation) && return raw_bits
    perm = _check_layout_permutation(layout_permutation, length(raw_bits))
    return Int[raw_bits[idx] for idx in perm]
end

"""
    QAOA.count_key_bitstring(key; layout_permutation=nothing)

Convert a Qiskit or IQM count key into a bitstring in QUBO variable order.
"""
function count_key_bitstring(key; layout_permutation = nothing)
    return join(count_key_bits(key; layout_permutation=layout_permutation))
end

function _normalize_counts_dict(counts)
    normalized = Pair{String,Int}[]
    if counts isa AbstractDict
        for (k, v) in counts
            reads = Int(v)
            reads < 0 && throw(ArgumentError("count multiplicity for key $(repr(k)) must be non-negative"))
            reads == 0 && continue
            push!(normalized, String(k) => reads)
        end
        return normalized
    elseif counts isa AbstractVector{<:Pair}
        for (k, v) in counts
            reads = Int(v)
            reads < 0 && throw(ArgumentError("count multiplicity for key $(repr(k)) must be non-negative"))
            reads == 0 && continue
            push!(normalized, String(k) => reads)
        end
        return normalized
    end

    try
        for key in counts.keys()
            key_str = _py_str(key)
            reads = _py_int(counts[key])
            reads < 0 && throw(ArgumentError("count multiplicity for key $(repr(key_str)) must be non-negative"))
            reads == 0 && continue
            push!(normalized, key_str => reads)
        end
        return normalized
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("counts must be a Julia AbstractDict or Python count dictionary"))
    end
end

function _decode_counts_entries(
    counts,
    num_variables::Integer;
    layout_permutation = nothing,
)
    perm = if isnothing(layout_permutation)
        nothing
    else
        checked = _check_layout_permutation(layout_permutation)
        length(checked) == num_variables ||
            throw(
                ArgumentError(
                    "layout_permutation length $(length(checked)) must match QUBO variable count $(num_variables)",
                ),
            )
        checked
    end

    if counts isa AbstractVector{BitVector}
        isempty(counts) && throw(ArgumentError("counts must contain at least one positive measurement count"))
        ordered_states = Vector{Int}[]
        aggregated_reads = Dict{Vector{Int},Int}()
        for shot in counts
            if length(shot) < num_variables || (!isnothing(perm) && any(idx -> idx > length(shot), perm))
                throw(
                    ArgumentError(
                        "decoded shot bitvector length $(length(shot)) does not match QUBO variable count $(num_variables)",
                    ),
                )
            end
            # QiskitIBMRuntime.Samples / BitVector shots are ALREADY in LSB-first order:
            # shot[j] is classical bit j-1 (1-based index j). Never reverse!
            state = Int[shot[isnothing(perm) ? i : perm[i]] ? 1 : 0 for i in 1:num_variables]
            if !haskey(aggregated_reads, state)
                push!(ordered_states, state)
                aggregated_reads[state] = 1
            else
                aggregated_reads[state] += 1
            end
        end
        return Pair{Vector{Int},Int}[state => aggregated_reads[state] for state in ordered_states]
    end

    entries = _normalize_counts_dict(counts)
    isempty(entries) && throw(ArgumentError("counts must contain at least one positive measurement count"))

    ordered_states = Vector{Int}[]
    aggregated_reads = Dict{Vector{Int},Int}()

    for (key, reads) in entries
        state = count_key_bits(key; layout_permutation=perm)
        if length(state) != num_variables
            throw(
                ArgumentError(
                    "decoded bitstring length $(length(state)) for count key $(repr(key)) " *
                    "does not match QUBO variable count $(num_variables)",
                ),
            )
        end
        if !haskey(aggregated_reads, state)
            push!(ordered_states, state)
            aggregated_reads[state] = reads
        else
            aggregated_reads[state] += reads
        end
    end

    return Pair{Vector{Int},Int}[state => aggregated_reads[state] for state in ordered_states]
end

"""
    QAOA.decode_counts_sampleset(source, counts; layout_permutation=nothing, transpiled_circuit=nothing, backend_name="custom_backend", backend_version=nothing, execution_mode="local", algorithm_name="QAOA", effective_time=0.0, total_time=effective_time, extra_metadata=nothing)

Decode a dictionary of circuit measurement counts into a `QUBOTools.SampleSet`
evaluated against `source` (`MOI.ModelLike` or `QUBODrivers.AbstractSampler`)
without requiring `qiskit_ibm_runtime`.

Preserves QUBO variable ordering, transpiler layout / classical-bit permutation,
objective scale (`α`) and offset (`β`), minimization/maximization sense, and
read multiplicities.
"""
function decode_counts_sampleset(
    source::MOI.ModelLike,
    counts;
    layout_permutation = nothing,
    transpiled_circuit = nothing,
    backend_name::AbstractString = "custom_backend",
    backend_version::Union{AbstractString,Nothing} = nothing,
    execution_mode::AbstractString = "local",
    algorithm_name::AbstractString = "QAOA",
    effective_time::Real = 0.0,
    total_time::Real = effective_time,
    extra_metadata::Union{AbstractDict,Nothing} = nothing,
)
    sampler = _fixed_parameter_sampler(source)
    return _decode_counts_sampleset_from_sampler(
        sampler,
        counts;
        layout_permutation=layout_permutation,
        transpiled_circuit=transpiled_circuit,
        backend_name=backend_name,
        backend_version=backend_version,
        execution_mode=execution_mode,
        algorithm_name=algorithm_name,
        effective_time=effective_time,
        total_time=total_time,
        extra_metadata=extra_metadata,
    )
end

function _decode_counts_sampleset_from_sampler(
    sampler::QUBODrivers.AbstractSampler{T},
    counts;
    layout_permutation = nothing,
    transpiled_circuit = nothing,
    backend_name::AbstractString = "custom_backend",
    backend_version::Union{AbstractString,Nothing} = nothing,
    execution_mode::AbstractString = "local",
    algorithm_name::AbstractString = "QAOA",
    effective_time::Real = 0.0,
    total_time::Real = effective_time,
    extra_metadata::Union{AbstractDict,Nothing} = nothing,
) where {T}
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())

    resolved_permutation = if !isnothing(layout_permutation)
        _check_layout_permutation(layout_permutation)
    elseif !isnothing(transpiled_circuit) && n > 0
        extract_layout_permutation(transpiled_circuit; num_variables=n)
    else
        nothing
    end

    decoded_entries = _decode_counts_entries(counts, n; layout_permutation=resolved_permutation)
    samples = QUBOTools.Sample{T,Int}[]
    total_reads = 0

    for (state, reads) in decoded_entries
        raw_value = QUBOTools.value(state, L, Q, α, β)
        value = sense == MOI.MAX_SENSE ? -raw_value : raw_value
        push!(samples, QUBOTools.Sample{T,Int}(state, value, reads))
        total_reads += reads
    end

    metadata = empty_metadata(
        algorithm_name,
        backend_name,
        execution_mode;
        backend_version=backend_version,
        backend_config_source="count_decoding_hook",
        number_of_reads=total_reads,
        final_number_of_reads=total_reads,
    )
    metadata["time"]["effective"] = Float64(effective_time)
    metadata["time"]["total"] = Float64(total_time)
    metadata["count_decoding"] = Dict{String,Any}(
        "variable_count" => n,
        "layout_permutation" => isnothing(resolved_permutation) ? collect(1:n) : resolved_permutation,
        "layout_permutation_source" => if !isnothing(layout_permutation)
            "explicit"
        elseif !isnothing(transpiled_circuit)
            "transpiled_circuit"
        else
            "identity"
        end,
        "objective_sense" => _objective_sense_name(sense),
        "scale" => α,
        "offset" => β,
        "total_reads" => total_reads,
        "unique_states" => length(samples),
    )

    if !isnothing(extra_metadata)
        for (k, v) in extra_metadata
            metadata[String(k)] = deepcopy(v)
        end
    end

    return SampleSet{T}(samples, metadata)
end

function _extract_counts_from_execution_output(output)
    output isa AbstractDict && return output
    output isa AbstractVector{<:Pair} && return output
    output isa AbstractVector{BitVector} && return output

    # Handle (job, counts) or NamedTuple with :counts / :result / :job
    if output isa NamedTuple
        if haskey(output, :samples) && !isnothing(output.samples)
            val = output.samples
            s = val isa Function ? val() : val
            s isa AbstractVector{BitVector} && return s
        end
        if haskey(output, :counts) && !isnothing(output.counts)
            val = output.counts
            return val isa Function ? val() : val
        end
        if haskey(output, :get_counts) && !isnothing(output.get_counts)
            val = output.get_counts
            return val isa Function ? val() : val
        end
        if haskey(output, :result) && !isnothing(output.result)
            val = output.result
            res = val isa Function ? val() : val
            return _extract_counts_from_execution_output(res)
        end
        if haskey(output, :job) && !isnothing(output.job)
            val = output.job
            jb = val isa Function ? val() : val
            return _extract_counts_from_execution_output(jb)
        end
    end

    # Try .get_counts() directly (Qiskit Result or Julia mock result)
    try
        get_counts_fn = getproperty(output, :get_counts)
        return get_counts_fn()
    catch
    end

    # Try .result() -> .get_counts() or SamplerV2 PrimitiveResult
    try
        result_fn = getproperty(output, :result)
        res = result_fn()
        try
            return getproperty(res, :get_counts)()
        catch
        end
        try
            return res[0].data.meas.get_counts()
        catch
        end
    catch
    end

    # Try Python dict with .keys()
    try
        _ = output.keys()
        return output
    catch
    end

    throw(ArgumentError("unable to extract measurement counts from execution output of type $(typeof(output))"))
end

"""
    QAOA.backend_run_executor(circuit, backend; shots=1024)

Execute a measured QAOA circuit on a Qiskit or IQM `BackendV2` object using
`backend.run(circuit, shots=shots)` without requiring `qiskit_ibm_runtime`.
Returns a NamedTuple `(job=job, counts=counts)`.
"""
function backend_run_executor(circuit, backend; shots::Integer = 1024)
    shot_count = _check_runtime_shots(shots)
    exec_circuit = circuit isa Qiskit.TranspileResult ? circuit.circuit : circuit
    run_fn = getproperty(backend, :run)
    job = if _is_py(backend)
        run_fn(exec_circuit; shots=_py_int_arg(shot_count))
    else
        try
            run_fn(exec_circuit; shots=shot_count)
        catch err
            err isa MethodError || rethrow()
            run_fn(exec_circuit, shot_count)
        end
    end
    counts = _extract_counts_from_execution_output(job)
    return (job=job, counts=counts)
end

function _run_circuit_with_hook(circuit, backend; shots::Integer, executor = nothing)
    shot_count = _check_runtime_shots(shots)
    exec_circuit = circuit isa Qiskit.TranspileResult ? circuit.circuit : circuit
    if !isnothing(executor)
        raw_out = try
            executor(exec_circuit, backend; shots=shot_count)
        catch err
            err isa MethodError || rethrow()
            executor(exec_circuit, backend, shot_count)
        end
        job = raw_out isa NamedTuple && haskey(raw_out, :job) ? raw_out.job : nothing
        counts = _extract_counts_from_execution_output(raw_out)
        return (job=job, counts=counts, execution_path="custom_executor")
    end

    has_run = if _is_py(backend)
        _py_hasattr(backend, "run")
    else
        hasproperty(backend, :run)
    end
    if has_run
        out = backend_run_executor(exec_circuit, backend; shots=shot_count)
        return (job=out.job, counts=out.counts, execution_path="backend_run")
    end

    qiskit_sampler = qiskit_ibm_runtime().SamplerV2(mode=backend)
    job = qiskit_sampler.run(_py_list_arg([exec_circuit]), shots=_py_int_arg(shot_count))
    counts = job.result()[0].data.meas.get_counts()
    return (job=job, counts=counts, execution_path="ibm_runtime_sampler_v2")
end

function _transpile_with_fallback(
    circuit,
    backend;
    pass_manager_factory = nothing,
    optimization_level::Union{Integer,Nothing} = 3,
    seed_transpiler = nothing,
)
    isnothing(optimization_level) && return (
        circuit = circuit,
        source = "skipped",
        optimization_level = nothing,
    )
    checked_level = _check_optimization_level(optimization_level)
    checked_seed = _check_transpiler_seed(seed_transpiler)

    if !isnothing(pass_manager_factory)
        pm_sel = _selected_pass_manager(
            backend;
            factory=pass_manager_factory,
            optimization_level=checked_level,
            seed_transpiler=checked_seed,
        )
        return (
            circuit = pm_sel.pass_manager.run(circuit),
            source = pm_sel.source,
            optimization_level = pm_sel.optimization_level,
        )
    end

    # If `backend` is a lightweight Julia mock backend without Qiskit Target, skip Qiskit preset transpilation
    if !_is_py(backend) && !hasproperty(backend, :target)
        return (
            circuit = circuit,
            source = "untranspiled_mock_backend",
            optimization_level = checked_level,
        )
    end

    pm_sel = _selected_pass_manager(
        backend;
        factory=nothing,
        optimization_level=checked_level,
        seed_transpiler=checked_seed,
    )
    return (
        circuit = pm_sel.pass_manager.run(circuit),
        source = pm_sel.source,
        optimization_level = pm_sel.optimization_level,
    )
end

"""
    QAOA.execute_fixed_parameter_circuit(source; parameters, reps=nothing, number_of_layers=nothing, parameter_order=:beta_then_gamma, backend=nothing, shots=1024, transpiler_seed=nothing, optimization_level=3, pass_manager_factory=nothing, layout_permutation=nothing, executor=nothing, execution_mode="local")

Build a fixed-parameter QAOA circuit from `source`, transpile it for `backend`
(when applicable), execute it through `executor` or `backend.run(circuit; shots=shots)`,
and decode the resulting measurement counts into a `QUBOTools.SampleSet` via
`QAOA.decode_counts_sampleset`.

This execution seam does not require `qiskit_ibm_runtime` and supports IQM
backends (`iqm.qiskit_iqm`), fake backends, local Aer, and custom/mock backend
adapters.
"""
function execute_fixed_parameter_circuit(
    source::MOI.ModelLike;
    parameters,
    reps = nothing,
    number_of_layers = nothing,
    parameter_order::Symbol = :beta_then_gamma,
    backend = nothing,
    shots::Integer = 1024,
    transpiler_seed::Union{Integer,Nothing} = nothing,
    optimization_level::Union{Integer,Nothing} = 3,
    pass_manager_factory = nothing,
    layout_permutation = nothing,
    executor = nothing,
    execution_mode::AbstractString = "local",
)
    shot_count = _check_runtime_shots(shots)
    circuit, fixed_metadata = fixed_parameter_circuit(
        source;
        parameters=parameters,
        reps=reps,
        number_of_layers=number_of_layers,
        parameter_order=parameter_order,
        measure=true,
    )

    target_backend = if isnothing(backend)
        default_local_backend()
    elseif backend isa Function || backend isa AbstractString || backend isa Symbol
        iqm_backend(backend)
    else
        backend
    end

    timed_exec = @timed begin
        transpiled = _transpile_with_fallback(
            circuit,
            target_backend;
            pass_manager_factory=pass_manager_factory,
            optimization_level=optimization_level,
            seed_transpiler=transpiler_seed,
        )
        run_res = _run_circuit_with_hook(
            transpiled.circuit,
            target_backend;
            shots=shot_count,
            executor=executor,
        )
        (transpiled=transpiled, run_res=run_res)
    end

    transpiled = timed_exec.value.transpiled
    run_res = timed_exec.value.run_res
    b_name = backend_name(target_backend)
    b_ver = backend_version(target_backend)

    extra_meta = Dict{String,Any}(
        "fixed_parameter_circuit" => _safe_fixed_parameter_metadata(fixed_metadata),
        "execution_hook" => Dict{String,Any}(
            "execution_path" => run_res.execution_path,
            "transpilation_source" => transpiled.source,
            "optimization_level" => transpiled.optimization_level,
            "transpiler_seed" => transpiler_seed,
        ),
    )

    sampleset = decode_counts_sampleset(
        source,
        run_res.counts;
        layout_permutation=layout_permutation,
        transpiled_circuit=transpiled.circuit,
        backend_name=b_name,
        backend_version=b_ver,
        execution_mode=execution_mode,
        effective_time=timed_exec.time,
        total_time=timed_exec.time,
        extra_metadata=extra_meta,
    )

    return (
        sampleset=sampleset,
        counts=run_res.counts,
        transpiled_circuit=transpiled.circuit,
        job=run_res.job,
        metadata=QUBOTools.metadata(sampleset),
    )
end

function _iqm_handoff_package_versions(; allow_import::Bool = false)
    iqm_ver = if allow_import || iqm_qiskit.cache[] !== nothing
        _python_package_version(iqm_qiskit)
    else
        nothing
    end
    return Dict{String,Any}(
        "QiskitOpt" => _julia_package_version(),
        "qiskit" => something(_module_version(Qiskit), "0.6.0"),
        "iqm.qiskit_iqm" => iqm_ver,
    )
end

function _iqm_handoff_backend_label(backend, server_url, quantum_computer)
    if backend isa AbstractString || backend isa Symbol
        label = strip(String(backend))
        !isempty(label) && return label
    elseif !isnothing(backend) && !(backend isa Function)
        return backend_name(backend)
    end
    qc = isnothing(quantum_computer) ? nothing : strip(String(quantum_computer))
    !isnothing(qc) && !isempty(qc) && return "iqm_$(qc)"
    !isnothing(iqm_server_url(server_url)) && return "iqm_server"
    return "iqm_backend"
end

function _iqm_handoff_metadata(;
    fixed_metadata,
    backend_label::AbstractString,
    server_url::Union{Nothing,AbstractString},
    quantum_computer::Union{Nothing,AbstractString},
    shots::Integer,
    transpiler_seed::Union{Integer,Nothing},
    optimization_level::Union{Integer,Nothing},
    dry_run::Bool,
    layout_permutation = nothing,
)
    resolved_url = iqm_server_url(server_url)
    resolved_token = iqm_token()
    resolved_tokens_file = iqm_tokens_file()

    return Dict{String,Any}(
        "iqm_handoff" => Dict{String,Any}(
            "provider" => "iqm",
            "python_package" => "iqm-client[qiskit]",
            "python_module" => "iqm.qiskit_iqm",
            "mode" => dry_run ? "dry_run" : "live",
            "status" => dry_run ? "dry_run" : "preparing_submission",
            "backend" => String(backend_label),
            "quantum_computer" => isnothing(quantum_computer) ? nothing : String(quantum_computer),
            "server_url_configured" => !isnothing(resolved_url),
            "token_configured" => !isnothing(resolved_token),
            "tokens_file_configured" => !isnothing(resolved_tokens_file),
            "shots" => Int(shots),
            "optimization_level" => optimization_level,
            "transpiler_seed" => transpiler_seed,
            "credentials_recorded" => false,
            "credential_fields_omitted" => ["token", "tokens_file", "server_url", "auth_header"],
        ),
        "fixed_parameter_circuit" => _safe_fixed_parameter_metadata(fixed_metadata),
        "count_scoring" => Dict{String,Any}(
            "count_key_order" =>
                "Qiskit/IQM count keys print classical bits from highest to lowest index",
            "bit_conversion" => "QAOA.count_key_bits",
            "sampleset_decoder" => "QAOA.decode_counts_sampleset",
            "layout_permutation" => isnothing(layout_permutation) ? nothing : collect(Int, layout_permutation),
            "value_convention" =>
                "QUBOTools.value(bits, linear, quadratic, scale, offset)",
        ),
        "packages" => _iqm_handoff_package_versions(; allow_import=!dry_run),
    )
end

"""
    QAOA.iqm_runtime_handoff(circuit; fixed_metadata=nothing, backend=nothing, server_url=nothing, quantum_computer=nothing, shots=1024, dry_run=true, transpiler_seed=nothing, optimization_level=3, pass_manager_factory=nothing, layout_permutation=nothing, decode_source=nothing, executor=nothing)

Prepare or execute a fixed-parameter QAOA circuit on an IQM Qiskit backend
(`iqm.qiskit_iqm` from `iqm-client[qiskit]`) without coupling to
`qiskit_ibm_runtime`.

When `dry_run=true` (default), no IQM server connection or `iqm-client` import
is required; sanitized handoff metadata is returned with `IQM_TOKEN`,
`IQM_TOKENS_FILE`, and `IQM_SERVER_URL` omitted.

When `dry_run=false`, resolves the IQM backend (from a passed backend/fake-backend
object, factory, fake backend name such as `"IQMFakeAdonis"`, or
`IQMProvider(server_url; quantum_computer=...)`), transpiles the circuit, executes
it with `shots`, and optionally decodes a `QUBOTools.SampleSet` when
`decode_source` (`MOI.ModelLike` or `QUBODrivers.AbstractSampler`) is provided.
Failures before submission throw `QAOA.RuntimeHandoffError` with redacted
metadata.
"""
function iqm_runtime_handoff(
    circuit;
    fixed_metadata = nothing,
    backend = nothing,
    server_url::Union{Nothing,AbstractString} = nothing,
    quantum_computer::Union{Nothing,AbstractString} = nothing,
    shots::Integer = 1024,
    dry_run::Bool = true,
    transpiler_seed::Union{Integer,Nothing} = nothing,
    optimization_level::Union{Integer,Nothing} = 3,
    pass_manager_factory = nothing,
    layout_permutation = nothing,
    decode_source = nothing,
    executor = nothing,
)
    _check_measured_circuit(circuit)
    if backend isa AbstractString && isempty(strip(backend))
        throw(ArgumentError("backend must be non-empty when provided as a String"))
    end
    shot_count = _check_runtime_shots(shots)
    checked_seed = _check_transpiler_seed(transpiler_seed)
    checked_optimization_level =
        isnothing(optimization_level) ? nothing : _check_optimization_level(optimization_level)
    checked_perm =
        isnothing(layout_permutation) ? nothing : _check_layout_permutation(layout_permutation)

    backend_label = _iqm_handoff_backend_label(backend, server_url, quantum_computer)
    metadata = _iqm_handoff_metadata(
        fixed_metadata=fixed_metadata,
        backend_label=backend_label,
        server_url=server_url,
        quantum_computer=quantum_computer,
        shots=shot_count,
        transpiler_seed=checked_seed,
        optimization_level=checked_optimization_level,
        dry_run=dry_run,
        layout_permutation=checked_perm,
    )

    if dry_run
        return (
            metadata=metadata,
            transpiled_circuit=nothing,
            job=nothing,
            counts=nothing,
            sampleset=nothing,
        )
    end

    target_backend = transpiled_circuit = job = counts = sampleset = nothing
    try
        target_backend = iqm_backend(
            backend;
            server_url=server_url,
            quantum_computer=quantum_computer,
        )
        metadata["iqm_handoff"]["backend"] = backend_name(target_backend)
        transpiled = _transpile_with_fallback(
            circuit,
            target_backend;
            pass_manager_factory=pass_manager_factory,
            optimization_level=checked_optimization_level,
            seed_transpiler=checked_seed,
        )
        transpiled_circuit = transpiled.circuit
        run_res = _run_circuit_with_hook(
            transpiled_circuit,
            target_backend;
            shots=shot_count,
            executor=executor,
        )
        job = run_res.job
        counts = run_res.counts
        if !isnothing(decode_source)
            sampleset = decode_counts_sampleset(
                decode_source,
                counts;
                layout_permutation=checked_perm,
                transpiled_circuit=transpiled_circuit,
                backend_name=backend_name(target_backend),
                backend_version=backend_version(target_backend),
                execution_mode="local",
                extra_metadata=Dict{String,Any}("iqm_handoff" => deepcopy(metadata["iqm_handoff"])),
            )
        end
    catch err
        handoff = metadata["iqm_handoff"]
        handoff["status"] = "failed_before_submission"
        handoff["failure"] = Dict{String,Any}(
            "exception_type" => _resource_audit_exception_type(err),
            "message" => _redact_runtime_message(
                sprint(showerror, err),
                (
                    runtime_token(),
                    runtime_instance(),
                    iqm_token(),
                    iqm_tokens_file(),
                    iqm_server_url(server_url),
                ),
            ),
            "package_hint" =>
                "Install 'iqm-client[qiskit]' into the active Python environment and set IQM_TOKEN or IQM_TOKENS_FILE for remote IQM Resonance backends",
        )
        # Mirror runtime_handoff failure key so RuntimeHandoffError showerror formats cleanly
        metadata["runtime_handoff"] = handoff
        throw(RuntimeHandoffError(metadata, err))
    end

    handoff = metadata["iqm_handoff"]
    handoff["status"] = "submitted"
    if !isnothing(job)
        handoff["job"] = Dict{String,Any}(
            "id" => _python_call_string(job, :job_id),
            "status" => _python_call_string(job, :status),
        )
    end

    return (
        metadata=metadata,
        transpiled_circuit=transpiled_circuit,
        job=job,
        counts=counts,
        sampleset=sampleset,
    )
end

"""
    QAOA.fixed_parameter_circuit(source; parameters, reps=nothing, number_of_layers=nothing, parameter_order=:beta_then_gamma, measure=true)

Build a Qiskit `QAOAAnsatz` circuit from an MOI model-like source or
QUBODrivers sampler and bind an explicit QAOA parameter vector without running
`QAOA.Optimizer`.

`parameters` defaults to Qiskit's QAOA list-binding order: all beta angles,
then all gamma angles. Pass `parameter_order=:gamma_then_beta` to provide the
opposite block order. The returned metadata records variable order, count-key
bit order, parameter order, objective scale/offset, objective sign convention,
and backend-independent circuit properties. Metadata parameter `values` are
always stored in Qiskit binding order and align positionally with
`parameter_names`; `input_order` records the caller's input format.
"""
function fixed_parameter_circuit(
    n::Integer,
    L::AbstractDict,
    Q::AbstractDict;
    p::Union{Integer,Nothing} = nothing,
    reps::Union{Integer,Nothing} = nothing,
    number_of_layers::Union{Integer,Nothing} = nothing,
    gamma::Union{AbstractVector{<:Real},Nothing} = nothing,
    beta::Union{AbstractVector{<:Real},Nothing} = nothing,
    parameters::Union{AbstractVector{<:Real},Nothing} = nothing,
    parameter_order::Symbol = :beta_then_gamma,
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
    measure::Bool = false,
    initial_state = nothing,
    mixer = nothing,
)
    layers = if !isnothing(p)
        _check_number_of_layers(p)
    elseif !isnothing(reps)
        _check_number_of_layers(reps)
    elseif !isnothing(number_of_layers)
        _check_number_of_layers(number_of_layers)
    else
        1
    end
    resolved_params = if !isnothing(parameters)
        _qiskit_ordered_qaoa_parameters(parameters, layers, parameter_order)
    elseif !isnothing(beta) && !isnothing(gamma)
        (length(beta) == layers && length(gamma) == layers) ||
            throw(ArgumentError("beta ($(length(beta))) and gamma ($(length(gamma))) lengths must match p ($layers)."))
        vcat(Float64.(beta), Float64.(gamma))
    else
        zeros(Float64, 2 * layers)
    end
    ising_h = qubo_to_ising(n, L, Q, sense)
    qc = build_qaoa_circuit(
        ising_h,
        layers,
        resolved_params;
        initial_state = initial_state,
        mixer = mixer,
    )
    measure && measure_all!(qc)
    return qc
end

function fixed_parameter_circuit(
    source::MOI.ModelLike;
    parameters,
    reps = nothing,
    number_of_layers = nothing,
    parameter_order::Symbol = :beta_then_gamma,
    measure::Bool = true,
)
    p = _resolve_fixed_parameter_layers(
        source;
        reps=reps,
        number_of_layers=number_of_layers,
    )
    sampler = _fixed_parameter_sampler(source)
    ising_hamiltonian = qubo_to_ising(sampler)
    ansatz = qaoa_ansatz(
        ising_hamiltonian;
        reps=p,
    )
    parameter_names = qiskit_parameter_names(ansatz)
    parameter_values = _qiskit_ordered_qaoa_parameters(parameters, p, parameter_order)
    validate_initial_parameters(parameter_values, length(parameter_names), "QAOA")

    circuit = ansatz(parameter_values)
    measure && measure_all!(circuit)

    metadata = _fixed_parameter_metadata(
        sampler,
        circuit,
        p,
        parameter_order,
        parameter_names,
        parameter_values,
        measure,
    )
    return circuit, metadata
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
    QAOA.linear_ramp_initial_parameters(; number_of_layers, delta_beta=0.35, delta_gamma=0.75, gamma_sign=-1)

Return Linear Ramp QAOA angles in Qiskit's list-binding order.
"""
function linear_ramp_initial_parameters(;
    number_of_layers::Integer,
    delta_beta::Real = 0.35,
    delta_gamma::Real = 0.75,
    gamma_sign::Real = -1,
)
    p = _check_number_of_layers(number_of_layers)
    beta = [delta_beta * (1 - i / p) for i in 0:(p - 1)]
    gamma = [gamma_sign * delta_gamma * ((i + 1) / p) for i in 0:(p - 1)]
    return Float64[v for v in vcat(beta, gamma)]
end

"""
    QAOA.tqa_initial_parameters(; number_of_layers, delta_t=0.75, gamma_sign=-1)

Return Trotterized Quantum Annealing QAOA angles in Qiskit's list-binding order.
"""
function tqa_initial_parameters(;
    number_of_layers::Integer,
    delta_t::Real = 0.75,
    gamma_sign::Real = -1,
)
    p = _check_number_of_layers(number_of_layers)
    beta = [delta_t * (1 - i / p) for i in 1:p]
    gamma = [gamma_sign * delta_t * (i / p) for i in 1:p]
    return Float64[v for v in vcat(beta, gamma)]
end

function _interpolate_to_next_depth(values::AbstractVector{<:Real})
    p = length(values)
    interpolated = Vector{Float64}(undef, p + 1)
    for i in 1:(p + 1)
        left = i == 1 ? 0.0 : Float64(values[i - 1])
        right = i == p + 1 ? 0.0 : Float64(values[i])
        interpolated[i] = ((i - 1) / p) * left + ((p - i + 1) / p) * right
    end
    return interpolated
end

"""
    QAOA.interpolated_initial_parameters(previous_parameters; gamma_sign=1)

Convert a depth-`p` Qiskit-ordered QAOA parameter vector into a depth-`p + 1`
vector by linearly interpolating the beta and gamma blocks separately.
"""
function interpolated_initial_parameters(
    previous_parameters::AbstractVector;
    gamma_sign::Real = 1,
)
    count = length(previous_parameters)
    count >= 2 || throw(ArgumentError("previous_parameters must contain beta and gamma blocks for at least one layer"))
    iseven(count) || throw(ArgumentError("previous_parameters length must be even"))
    all(parameter -> parameter isa Real, previous_parameters) ||
        throw(ArgumentError("previous_parameters must contain real values"))

    p = div(count, 2)
    numeric_parameters = Float64.(previous_parameters)
    beta = _interpolate_to_next_depth(numeric_parameters[1:p])
    gamma = _interpolate_to_next_depth(numeric_parameters[(p + 1):end])
    return Float64[v for v in vcat(beta, gamma_sign .* gamma)]
end

"""
    QAOA.fixed_angle_guarantee(; number_of_layers, family=:wurtz_lykov_3regular_tree)

Return the approximation-ratio guarantee reported with the built-in fixed-angle
table.
"""
function fixed_angle_guarantee(;
    number_of_layers::Integer,
    family::Symbol = :wurtz_lykov_3regular_tree,
)
    return Float64(_fixed_angle_record(number_of_layers, family).guarantee)
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
    angles = _fixed_angle_record(number_of_layers, family)
    return Float64[v for v in vcat(angles.beta, gamma_sign .* angles.gamma)]
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())

    # Results vector
    samples = QUBOTools.Sample{T,Int}[]

    retrieve_results = @timed retrieve(sampler) do _, sample_results
        for (state, reads) in _decode_counts_entries(sample_results, n)
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
    max_iter        = MOI.get(sampler, QAOA.MaximumIterations())
    num_reads       = MOI.get(sampler, QAOA.NumberOfReads())
    final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())
    num_layers      = MOI.get(sampler, QAOA.NumberOfLayers())
    sampler_seed    = MOI.get(sampler, QUBODrivers.RandomSeed())
    ibm_backend     = MOI.get(sampler, QAOA.IBMBackend())
    ibm_fake_backend = MOI.get(sampler, QAOA.IBMFakeBackend())
    channel         = runtime_channel(MOI.get(sampler, QAOA.Channel()))
    instance        = runtime_instance(MOI.get(sampler, QAOA.Instance()))
    initial_parameters   = MOI.get(sampler, QAOA.InitialParameters())
    initial_parameter_source = MOI.get(sampler, QAOA.InitialParameterSource())
    record_initial_parameters = MOI.get(sampler, QAOA.RecordInitialParameters())
    is_local         = MOI.get(sampler, QAOA.IsLocal())
    pass_manager_factory = MOI.get(sampler, QAOA.PassManagerFactory())
    circuit_executor = MOI.get(sampler, QAOA.CircuitExecutor())
    seed_simulator = effective_seed(MOI.get(sampler, QAOA.AerSeedSimulator()), sampler_seed, 1)
    seed_transpiler = effective_seed(MOI.get(sampler, QAOA.TranspilerSeed()), sampler_seed, 2)
    aer_config = AerBackendConfig(
        method=MOI.get(sampler, QAOA.AerBackendMethod()),
        precision=MOI.get(sampler, QAOA.AerPrecision()),
        max_parallel_threads=MOI.get(sampler, QAOA.AerMaxParallelThreads()),
        mps_omp_threads=MOI.get(sampler, QAOA.AerMPSOmpThreads()),
        mps_truncation_threshold=MOI.get(sampler, QAOA.AerMPSTruncationThreshold()),
        mps_max_bond_dimension=MOI.get(sampler, QAOA.AerMPSMaxBondDimension()),
        mps_sample_measure_algorithm=MOI.get(sampler, QAOA.AerMPSSampleMeasureAlgorithm()),
        seed_simulator=seed_simulator,
        seed_transpiler=seed_transpiler,
    )

    n, L, Q, _, _ = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())
    ising_hamiltonian = qubo_to_ising(n, L, Q; sense = sense)
    ansatz = qaoa_ansatz(ising_hamiltonian; reps=num_layers)
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

    pass_manager_selection = _selected_pass_manager(
        backend;
        factory=pass_manager_factory,
        optimization_level=3,
        seed_transpiler=aer_config.seed_transpiler,
    )
    pass_manager = pass_manager_selection.pass_manager

    ansatz_isa = pass_manager.run(ansatz)
    isa_hamiltonian = if ansatz_isa isa NativeAnsatz && ansatz_isa.layout !== nothing
        apply_layout_to_hamiltonian(ising_hamiltonian, ansatz_isa.layout)
    elseif !(ansatz_isa isa NativeAnsatz) && !(ansatz_isa isa Qiskit.QuantumCircuit) && hasproperty(ansatz_isa, :layout)
        layout_obj = getproperty(ansatz_isa, :layout)
        layout_obj isa Qiskit.TranspileLayout ? apply_layout_to_hamiltonian(ising_hamiltonian, layout_obj) : ising_hamiltonian
    else
        ising_hamiltonian
    end

    eval_counter = Ref(0)
    function cost_function(params::AbstractVector{<:Real})
        eval_counter[] += 1
        bound_qc = bind_ansatz_parameters(ansatz_isa, params)
        eval_seed = aer_config.seed_simulator === nothing ? nothing : Int(aer_config.seed_simulator) + eval_counter[] - 1
        if !(backend isa NativeLocalBackend || backend isa NativeFakeBackend) && ising_hamiltonian.num_qubits > 0
            meas_qc = copy(bound_qc)
            measure_all!(meas_qc)
            raw_counts = _run_circuit_with_hook(
                meas_qc,
                backend;
                shots = num_reads,
                executor = circuit_executor,
            ).counts
            val = 0.0
            total_r = 0
            for (state, reads) in _decode_counts_entries(raw_counts, isa_hamiltonian.num_qubits)
                val += reads * evaluate_ising_bitstring(isa_hamiltonian, state)
                total_r += reads
            end
            return total_r > 0 ? val / total_r : 0.0
        end
        return estimate_expectation(
            bound_qc,
            isa_hamiltonian;
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

    samples = if !isnothing(circuit_executor) || !(backend isa NativeLocalBackend)
        _run_circuit_with_hook(
            optimized_qc,
            backend;
            shots=final_num_reads,
            executor=circuit_executor,
        ).counts
    else
        sample_from_circuit(
            optimized_qc;
            shots = final_num_reads,
            seed = aer_config.seed_simulator,
            precision = something(aer_config.precision, "double"),
            max_bond_dimension = aer_config.mps_max_bond_dimension,
            truncation_threshold = aer_config.mps_truncation_threshold,
        )
    end

    callback(result, samples)

    return empty_metadata(
        "QAOA",
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
        pass_manager_source=pass_manager_selection.source,
        pass_manager_optimization_level=pass_manager_selection.optimization_level,
        initial_parameters=record_initial_parameters ? initial_parameter_values : nothing,
        initial_parameter_names=record_initial_parameters ? parameter_names : nothing,
        initial_parameter_source=record_initial_parameters ? initial_parameter_source : nothing,
        optimized_parameters=optimized_parameter_values,
        optimized_parameter_names=parameter_names,
        optimized_parameter_optimizer=optimized_parameter_optimizer,
    )
end

end # module QAOA
