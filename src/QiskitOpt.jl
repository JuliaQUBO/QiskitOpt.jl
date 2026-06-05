module QiskitOpt

using PythonCall
using QUBO
MOI = QUBODrivers.MOI

const DEFAULT_RUNTIME_CHANNEL = "ibm_quantum_platform"

struct PythonPackageError <: Exception
    module_name::String
    package_name::String
    python_executable::Union{Nothing,String}
    cause::String
end

function Base.showerror(io::IO, err::PythonPackageError)
    python = isnothing(err.python_executable) ? "unknown" : err.python_executable
    print(
        io,
        "Python package '$(err.package_name)' (import name '$(err.module_name)') could not be imported. ",
        "Python executable: $(python). ",
        "Install '$(err.package_name)' into that Python environment or configure PythonCall to use one that provides it.",
    )
    if !isempty(err.cause)
        print(io, "\nOriginal error: ", err.cause)
    end
end

Base.@kwdef struct RuntimeDiagnostic
    name::String
    ok::Bool
    version::Union{Nothing,String} = nothing
    message::String = ""
end

Base.@kwdef struct RuntimeDiagnostics
    ok::Bool
    julia::RuntimeDiagnostic
    pythoncall::RuntimeDiagnostic
    packages::Dict{String,RuntimeDiagnostic}
    local_backend::Union{Nothing,RuntimeDiagnostic} = nothing
    ibm_runtime::Union{Nothing,RuntimeDiagnostic} = nothing
end

struct LazyPythonModule
    module_name::String
    package_name::String
    cache::Base.RefValue{Any}
end

function LazyPythonModule(module_name::AbstractString, package_name::AbstractString)
    return LazyPythonModule(String(module_name), String(package_name), Ref{Any}(nothing))
end

function LazyPythonModule(module_name::AbstractString)
    return LazyPythonModule(module_name, replace(String(module_name), "_" => "-"))
end

function _reset_python_module!(lazy_module::LazyPythonModule)
    lazy_module.cache[] = nothing
    return lazy_module
end

function python_executable()
    try
        return PythonCall.pyconvert(String, PythonCall.pyimport("sys").executable)
    catch
        return nothing
    end
end

function _import_python_module(module_name::AbstractString, package_name::AbstractString)
    try
        return PythonCall.pyimport(module_name)
    catch err
        throw(
            PythonPackageError(
                String(module_name),
                String(package_name),
                python_executable(),
                sprint(showerror, err),
            ),
        )
    end
end

function _load_python_module!(lazy_module::LazyPythonModule)
    cached = lazy_module.cache[]
    cached !== nothing && return cached

    loaded = _import_python_module(lazy_module.module_name, lazy_module.package_name)
    lazy_module.cache[] = loaded
    return loaded
end

(lazy_module::LazyPythonModule)() = _load_python_module!(lazy_module)

function Base.getproperty(lazy_module::LazyPythonModule, name::Symbol)
    if name === :module_name || name === :package_name || name === :cache
        return getfield(lazy_module, name)
    end

    return getproperty(_load_python_module!(lazy_module), name)
end

function Base.show(io::IO, lazy_module::LazyPythonModule)
    status = lazy_module.cache[] === nothing ? "not loaded" : "loaded"
    print(io, "LazyPythonModule(", lazy_module.module_name, ", ", status, ")")
end

# :: Python Qiskit Modules ::
const qiskit              = LazyPythonModule("qiskit", "qiskit")
const qiskit_optimization = LazyPythonModule("qiskit_optimization", "qiskit-optimization")
const qiskit_ibm_runtime  = LazyPythonModule("qiskit_ibm_runtime", "qiskit-ibm-runtime")
const qiskit_aer          = LazyPythonModule("qiskit_aer", "qiskit-aer")
const scipy               = LazyPythonModule("scipy", "scipy")
const numpy               = LazyPythonModule("numpy", "numpy")
const _runtime_service_builder = Ref{Function}()
const DEFAULT_AER_BACKEND_METHOD = "matrix_product_state"
const _LOCAL_SOLVER_PRIMITIVES_NOTE =
    "QAOA/VQE local solves still use qiskit_ibm_runtime EstimatorV2/SamplerV2; pass ibm=true to verify those primitives."

struct AerBackendConfig
    method::Union{String,Nothing}
    precision::Union{String,Nothing}
    max_parallel_threads::Union{Integer,Nothing}
    mps_omp_threads::Union{Integer,Nothing}
    mps_truncation_threshold::Union{Real,Nothing}
    mps_max_bond_dimension::Union{Integer,Nothing}
    mps_sample_measure_algorithm::Union{String,Nothing}
    seed_simulator::Union{Integer,Nothing}
    seed_transpiler::Union{Integer,Nothing}
end

function AerBackendConfig(;
    method::Union{AbstractString,Nothing} = DEFAULT_AER_BACKEND_METHOD,
    precision::Union{AbstractString,Nothing} = nothing,
    max_parallel_threads::Union{Integer,Nothing} = nothing,
    mps_omp_threads::Union{Integer,Nothing} = nothing,
    mps_truncation_threshold::Union{Real,Nothing} = nothing,
    mps_max_bond_dimension::Union{Integer,Nothing} = nothing,
    mps_sample_measure_algorithm::Union{AbstractString,Nothing} = nothing,
    seed_simulator::Union{Integer,Nothing} = nothing,
    seed_transpiler::Union{Integer,Nothing} = nothing,
)
    return AerBackendConfig(
        nonempty_or_nothing(method),
        nonempty_or_nothing(precision),
        max_parallel_threads,
        mps_omp_threads,
        mps_truncation_threshold,
        mps_max_bond_dimension,
        nonempty_or_nothing(mps_sample_measure_algorithm),
        seed_simulator,
        seed_transpiler,
    )
end

struct AerBackendFactory
    config::AerBackendConfig
end

function (factory::AerBackendFactory)()
    return local_aer_backend(factory.config)
end

function __init__()
    foreach(
        _reset_python_module!,
        (qiskit, qiskit_optimization, qiskit_ibm_runtime, qiskit_aer, scipy, numpy),
    )
    _runtime_service_builder[] = default_runtime_service
end

function _diagnostic_error_message(err)
    return sprint(showerror, err)
end

function _module_version(jl_module::Module)
    try
        version = Base.pkgversion(jl_module)
        return isnothing(version) ? nothing : string(version)
    catch
        return nothing
    end
end

function _python_module_version(py_module)
    try
        return PythonCall.pyconvert(String, getproperty(py_module, :__version__))
    catch
        return nothing
    end
end

function _diagnose_julia_package()
    version = _module_version(@__MODULE__)
    return RuntimeDiagnostic(
        name="QiskitOpt",
        ok=true,
        version=version,
        message="Julia version: $(VERSION)",
    )
end

function _diagnose_pythoncall()
    executable = python_executable()
    ok = !isnothing(executable)
    message = ok ? "Python executable: $(executable)" : "Python executable could not be resolved"
    return RuntimeDiagnostic(
        name="PythonCall",
        ok=ok,
        version=_module_version(PythonCall),
        message=message,
    )
end

function _diagnose_python_module(lazy_module::LazyPythonModule)
    try
        loaded = _load_python_module!(lazy_module)
        version = _python_module_version(loaded)
        executable = python_executable()
        message = isnothing(executable) ? "" : "Python executable: $(executable)"
        return RuntimeDiagnostic(
            name=lazy_module.module_name,
            ok=true,
            version=version,
            message=message,
        )
    catch err
        return RuntimeDiagnostic(
            name=lazy_module.module_name,
            ok=false,
            message=_diagnostic_error_message(err),
        )
    end
end

function _diagnose_python_module(module_name::AbstractString, package_name::AbstractString)
    return _diagnose_python_module(LazyPythonModule(module_name, package_name))
end

function _diagnose_local_backend()
    try
        backend = default_local_backend()
        return RuntimeDiagnostic(
            name="default_local_backend",
            ok=true,
            message="Usable backend: $(backend_name(backend)). $(_LOCAL_SOLVER_PRIMITIVES_NOTE)",
        )
    catch err
        return RuntimeDiagnostic(
            name="default_local_backend",
            ok=false,
            message=_diagnostic_error_message(err),
        )
    end
end

function _print_runtime_diagnostic(io::IO, label::AbstractString, diagnostic::RuntimeDiagnostic)
    status = diagnostic.ok ? "OK" : "FAIL"
    version = isnothing(diagnostic.version) ? "" : " ($(diagnostic.version))"
    println(io, "  $(label): $(status)$(version)")
    if !isempty(diagnostic.message)
        println(io, "    ", replace(diagnostic.message, "\n" => "\n    "))
    end
end

function _print_runtime_diagnostics(io::IO, result::RuntimeDiagnostics)
    println(io, "QiskitOpt runtime diagnostics: ", result.ok ? "OK" : "FAILED")
    _print_runtime_diagnostic(io, "Julia package", result.julia)
    _print_runtime_diagnostic(io, "PythonCall", result.pythoncall)
    for name in sort(collect(keys(result.packages)))
        _print_runtime_diagnostic(io, name, result.packages[name])
    end
    if !isnothing(result.local_backend)
        _print_runtime_diagnostic(io, "Local backend", result.local_backend)
    end
    if !isnothing(result.ibm_runtime)
        _print_runtime_diagnostic(io, "IBM Runtime", result.ibm_runtime)
    end
    return nothing
end

function check_runtime(; local_backend::Bool=true, ibm::Bool=false, verbose::Bool=true, kwargs...)
    unknown_keywords = setdiff(collect(keys(kwargs)), [:local])
    if !isempty(unknown_keywords)
        throw(ArgumentError("unsupported check_runtime keyword(s): $(join(unknown_keywords, ", "))"))
    end
    if haskey(kwargs, :local)
        kwargs[:local] isa Bool || throw(ArgumentError("check_runtime local keyword must be a Bool"))
        local_backend = kwargs[:local]
    end

    julia = _diagnose_julia_package()
    pythoncall = _diagnose_pythoncall()
    packages = Dict{String,RuntimeDiagnostic}()

    for lazy_module in (qiskit, qiskit_aer, qiskit_optimization, scipy, numpy)
        diagnostic = _diagnose_python_module(lazy_module)
        packages[diagnostic.name] = diagnostic
    end

    ibm_runtime = if ibm
        diagnostic = _diagnose_python_module(qiskit_ibm_runtime)
        diagnostic
    else
        nothing
    end

    local_backend_diagnostic = local_backend ? _diagnose_local_backend() : nothing

    diagnostics = RuntimeDiagnostic[julia, pythoncall]
    append!(diagnostics, values(packages))
    if !isnothing(ibm_runtime)
        push!(diagnostics, ibm_runtime)
    end
    if !isnothing(local_backend_diagnostic)
        push!(diagnostics, local_backend_diagnostic)
    end

    result = RuntimeDiagnostics(
        ok=all(diagnostic -> diagnostic.ok, diagnostics),
        julia=julia,
        pythoncall=pythoncall,
        packages=packages,
        local_backend=local_backend_diagnostic,
        ibm_runtime=ibm_runtime,
    )

    verbose && _print_runtime_diagnostics(stdout, result)
    return result
end

function nonempty_or_nothing(value)
    if isnothing(value)
        return nothing
    end

    text = strip(String(value))
    return isempty(text) ? nothing : text
end

function runtime_channel(channel::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(channel)
    if !isnothing(provided)
        return provided == "ibm_quantum" ? DEFAULT_RUNTIME_CHANNEL : provided
    end

    env_channel = nonempty_or_nothing(get(ENV, "QISKIT_IBM_CHANNEL", nothing))
    if isnothing(env_channel) || env_channel == "ibm_quantum"
        return DEFAULT_RUNTIME_CHANNEL
    end

    return env_channel
end

function runtime_token()
    for name in ("QISKIT_IBM_TOKEN", "IBMQ_API_TOKEN")
        token = nonempty_or_nothing(get(ENV, name, nothing))
        !isnothing(token) && return token
    end

    return nothing
end

function runtime_instance(instance::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(instance)
    !isnothing(provided) && return provided

    for name in ("QISKIT_IBM_INSTANCE", "IBMQ_INSTANCE")
        resolved = nonempty_or_nothing(get(ENV, name, nothing))
        !isnothing(resolved) && return resolved
    end

    return nothing
end

function default_runtime_service(;
    channel::Union{Nothing,AbstractString} = nothing,
    token::Union{Nothing,AbstractString} = nothing,
    instance::Union{Nothing,AbstractString} = nothing,
)
    resolved_channel = runtime_channel(channel)
    resolved_token = nonempty_or_nothing(token)
    resolved_instance = runtime_instance(instance)
    runtime = qiskit_ibm_runtime()

    if isnothing(resolved_token) && isnothing(resolved_instance)
        return runtime.QiskitRuntimeService(channel=resolved_channel)
    elseif isnothing(resolved_token)
        return runtime.QiskitRuntimeService(
            channel=resolved_channel,
            instance=resolved_instance,
        )
    elseif isnothing(resolved_instance)
        return runtime.QiskitRuntimeService(
            channel=resolved_channel,
            token=resolved_token,
        )
    end

    return runtime.QiskitRuntimeService(
        channel=resolved_channel,
        token=resolved_token,
        instance=resolved_instance,
    )
end

function runtime_service(;
    channel::Union{Nothing,AbstractString} = nothing,
    instance::Union{Nothing,AbstractString} = nothing,
)
    return _runtime_service_builder[](
        channel=runtime_channel(channel),
        token=runtime_token(),
        instance=runtime_instance(instance),
    )
end

function _push_option!(options::Vector{Pair{Symbol,Any}}, name::Symbol, value)
    isnothing(value) || push!(options, name => value)
    return options
end

function aer_backend_options(config::AerBackendConfig)
    options = Pair{Symbol,Any}[]
    _push_option!(options, :method, config.method)
    _push_option!(options, :precision, config.precision)
    _push_option!(options, :max_parallel_threads, config.max_parallel_threads)
    _push_option!(options, :mps_omp_threads, config.mps_omp_threads)
    _push_option!(options, :mps_truncation_threshold, config.mps_truncation_threshold)
    _push_option!(options, :mps_max_bond_dimension, config.mps_max_bond_dimension)
    _push_option!(options, :mps_sample_measure_algorithm, config.mps_sample_measure_algorithm)
    _push_option!(options, :seed_simulator, config.seed_simulator)

    names = Tuple(first.(options))
    values = Tuple(last.(options))
    return NamedTuple{names}(values)
end

function local_aer_backend(config::AerBackendConfig = AerBackendConfig())
    return qiskit_aer().AerSimulator(; aer_backend_options(config)...)
end

function default_local_backend(; kwargs...)
    return local_aer_backend(AerBackendConfig(; kwargs...))
end

function local_aer_backend_factory(config::AerBackendConfig)
    return AerBackendFactory(config)
end

function local_aer_backend_factory(; kwargs...)
    return local_aer_backend_factory(AerBackendConfig(; kwargs...))
end

function local_aer_backend_from_backend(remote_backend, config::AerBackendConfig)
    return qiskit_aer().AerSimulator.from_backend(
        remote_backend;
        aer_backend_options(config)...,
    )
end

function configured_local_backend(local_backend_factory, config::AerBackendConfig)
    if local_backend_factory === default_local_backend
        return (
            backend = local_aer_backend(config),
            config = config,
            source = "aer_attributes",
        )
    elseif local_backend_factory isa AerBackendFactory
        return (
            backend = local_backend_factory(),
            config = local_backend_factory.config,
            source = "aer_backend_factory",
        )
    end

    return (
        backend = local_backend_factory(),
        config = nothing,
        source = "user_backend_factory",
    )
end

function configured_execution_backend(;
    ibm_backend::Union{AbstractString,Nothing},
    local_backend_factory,
    is_local::Bool,
    channel::AbstractString,
    instance::Union{AbstractString,Nothing},
    aer_config::AerBackendConfig,
)
    if isnothing(ibm_backend)
        local_backend = configured_local_backend(local_backend_factory, aer_config)
        return (
            backend = local_backend.backend,
            execution_mode = "local",
            config = local_backend.config,
            source = local_backend.source,
        )
    end

    remote_backend = runtime_service(channel=channel, instance=instance).backend(ibm_backend)
    if is_local
        return (
            backend = local_aer_backend_from_backend(remote_backend, aer_config),
            execution_mode = "local",
            config = aer_config,
            source = "aer_from_backend",
        )
    end

    return (
        backend = remote_backend,
        execution_mode = "cloud",
        config = nothing,
        source = "cloud_backend",
    )
end

function preset_pass_manager(backend; optimization_level::Integer = 3, seed_transpiler = nothing)
    if isnothing(seed_transpiler)
        return qiskit().transpiler.preset_passmanagers.generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
    end

    return qiskit().transpiler.preset_passmanagers.generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
end

function backend_name(backend)
    try
        return PythonCall.pyconvert(String, backend.name)
    catch
        return PythonCall.pyconvert(String, backend.backend_name)
    end
end

function sample_bits(key)
    bitstring = replace(PythonCall.pyconvert(String, key), " " => "")
    return reverse(Int[(digit == '1') for digit in bitstring])
end

function aer_backend_metadata(config::AerBackendConfig, source::AbstractString)
    return Dict{String,Any}(
        "source" => String(source),
        "method" => config.method,
        "precision" => config.precision,
        "max_parallel_threads" => config.max_parallel_threads,
        "mps_omp_threads" => config.mps_omp_threads,
        "mps_truncation_threshold" => config.mps_truncation_threshold,
        "mps_max_bond_dimension" => config.mps_max_bond_dimension,
        "mps_sample_measure_algorithm" => config.mps_sample_measure_algorithm,
    )
end

function seed_metadata(;
    seed_simulator = nothing,
    seed_transpiler = nothing,
)
    return Dict{String,Any}(
        "simulator" => seed_simulator,
        "transpiler" => seed_transpiler,
    )
end

function _seed_state(seed::Integer)
    modulus = BigInt(typemax(UInt64)) + 1
    return UInt64(mod(BigInt(seed), modulus))
end

function _splitmix64_next(state::UInt64)
    state += 0x9e3779b97f4a7c15
    value = state
    value = xor(value, value >> 30) * 0xbf58476d1ce4e5b9
    value = xor(value, value >> 27) * 0x94d049bb133111eb
    return state, xor(value, value >> 31)
end

function _unit_float(value::UInt64)
    return Float64(value >> 11) * 0x1.0p-53
end

function _seeded_unit_values(seed::Integer, count::Integer)
    state = _seed_state(seed)
    values = Vector{Float64}(undef, count)
    for index in eachindex(values)
        state, value = _splitmix64_next(state)
        values[index] = _unit_float(value)
    end
    return values
end

function random_unit_values(count::Integer; seed = nothing, rng = nothing)
    count >= 0 || throw(ArgumentError("count must be nonnegative"))
    if !isnothing(seed) && !isnothing(rng)
        throw(ArgumentError("provide either seed or rng, not both"))
    end

    if !isnothing(rng)
        return rand(rng, Float64, count)
    elseif !isnothing(seed)
        seed isa Integer || throw(ArgumentError("seed must be an integer"))
        return _seeded_unit_values(seed, count)
    end

    return rand(Float64, count)
end

function qiskit_parameter_names(circuit)
    return String[
        PythonCall.pyconvert(String, PythonCall.pystr(parameter))
        for parameter in circuit.parameters
    ]
end

function validate_initial_parameters(
    initial_parameters::AbstractVector,
    expected_count::Integer,
    algorithm::AbstractString,
)
    actual_count = length(initial_parameters)
    if actual_count != expected_count
        throw(
            ArgumentError(
                "$(algorithm) InitialParameters length $(actual_count) does not match " *
                "the expected Qiskit ansatz parameter count $(expected_count)",
            ),
        )
    end

    return nothing
end

function initial_parameter_metadata(
    source::AbstractString,
    values::AbstractVector,
    names::AbstractVector{<:AbstractString},
)
    return Dict{String,Any}(
        "source" => String(source),
        "parameter_names" => String.(names),
        "values" => Float64.(values),
    )
end

function empty_metadata(
    algorithm::AbstractString,
    backend::AbstractString,
    execution_mode::AbstractString;
    backend_config::Union{AerBackendConfig,Nothing} = nothing,
    backend_config_source::Union{AbstractString,Nothing} = nothing,
    seed_transpiler = nothing,
    initial_parameters::Union{AbstractVector,Nothing} = nothing,
    initial_parameter_names::Union{AbstractVector{<:AbstractString},Nothing} = nothing,
    initial_parameter_source::Union{AbstractString,Nothing} = nothing,
)
    metadata = Dict{String,Any}(
        "origin" => "$(algorithm) @ $(backend)",
        "backend" => backend,
        "execution_mode" => execution_mode,
        "time" => Dict{String,Any}(),
        "evals" => Float64[],
    )

    if !isnothing(backend_config) || !isnothing(backend_config_source)
        source = isnothing(backend_config_source) ? "unknown" : backend_config_source
        metadata["backend_configuration"] = if isnothing(backend_config)
            Dict{String,Any}("source" => String(source))
        else
            aer_backend_metadata(backend_config, source)
        end
    end

    if !isnothing(backend_config) || !isnothing(seed_transpiler)
        seed_simulator = isnothing(backend_config) ? nothing : backend_config.seed_simulator
        metadata["seeds"] = seed_metadata(
            seed_simulator=seed_simulator,
            seed_transpiler=seed_transpiler,
        )
    end

    if !isnothing(initial_parameters)
        source = isnothing(initial_parameter_source) ? "unknown" : initial_parameter_source
        names = isnothing(initial_parameter_names) ? String[] : initial_parameter_names
        metadata["initial_parameters"] = initial_parameter_metadata(source, initial_parameters, names)
    end

    return metadata
end

function quadratic_program(sampler::QUBODrivers.AbstractSampler{T}) where {T}
    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dense)

    # Build Qiskit Model
    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()
    variable_names = string.(1:n)

    sense = MOI.get(sampler, MOI.ObjectiveSense())

    for i in 1:n
        linear[variable_names[i]] = L[i] * (sense == MOI.MIN_SENSE ? 1 : -1)
    end
    for i in 1:n, j in 1:n
        quadratic[variable_names[i], variable_names[j]] = Q[i,j] * (sense == MOI.MIN_SENSE ? 1 : -1)
    end

    qp = qiskit_optimization().QuadraticProgram()

    for v in variable_names
        qp.binary_var(v)
    end

    qp.minimize(linear = linear, quadratic = quadratic)
    
    return qp.to_ising()
end

export  VQE, QAOA, check_runtime

include("QAOA.jl")
include("VQE.jl")

end # module QiskitOpt
