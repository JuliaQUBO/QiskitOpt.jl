module QiskitOpt

using LinearAlgebra
using QUBO
using Qiskit
using QiskitIBMRuntime
using Random
MOI = QUBODrivers.MOI

include("ising.jl")
include("circuits.jl")
include("simulator.jl")
include("optimizer.jl")

const DEFAULT_RUNTIME_CHANNEL = "ibm_quantum_platform"

# Optional PythonCall extension hooks (populated by `QiskitOptPythonCallExt` when `PythonCall` is loaded)
const _is_py_hook = Ref{Function}(x -> false)
const _py_import_hook = Ref{Function}(
    module_name -> error("PythonCall is not loaded; run `using PythonCall` to import Python module '$(module_name)'."),
)
const _py_executable_hook = Ref{Function}(() -> nothing)
const _py_version_hook = Ref{Function}(() -> nothing)
const _py_str_hook = Ref{Function}(x -> string(x))
const _py_int_hook = Ref{Function}(x -> Int(x))
const _py_float_hook = Ref{Function}(x -> Float64(x))
const _py_bool_hook = Ref{Function}(x -> Bool(x))
const _py_len_hook = Ref{Function}(x -> length(x))
const _py_hasattr_hook = Ref{Function}((x, name) -> hasproperty(x, Symbol(name)))
const _py_is_none_hook = Ref{Function}(x -> x === nothing)
const _py_int_arg_hook = Ref{Function}(x -> Int(x))
const _py_list_arg_hook = Ref{Function}(x -> collect(x))
const _py_exception_type_hook = Ref{Function}(err -> nothing)
const _py_int_vec_hook = Ref{Function}(x -> Int[Int(v) for v in x])
const _py_float_vec_hook = Ref{Function}(x -> Float64[Float64(v) for v in x])

_is_py(x) = _is_py_hook[](x)
_py_str(x) = x isa AbstractString ? String(x) : _py_str_hook[](x)
_py_int(x) = x isa Integer ? Int(x) : _py_int_hook[](x)
_py_float(x) = x isa Real ? Float64(x) : _py_float_hook[](x)
_py_bool(x) = x isa Bool ? x : _py_bool_hook[](x)
_py_len(x) = _is_py(x) ? _py_len_hook[](x) : length(x)
_py_hasattr(x, name::AbstractString) = _is_py(x) ? _py_hasattr_hook[](x, String(name)) : hasproperty(x, Symbol(name))
_py_is_none(x) = x === nothing || (_is_py(x) && _py_is_none_hook[](x))
_py_int_arg(x::Integer) = _py_int_arg_hook[](Int(x))
_py_list_arg(x) = _py_list_arg_hook[](x)
_py_int_vec(x) = _is_py(x) ? _py_int_vec_hook[](x) : Int[Int(v) for v in x]

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

struct SamplePostprocessingError <: Exception
    rank::Int
    sample::Any
    cause::Any
end

function Base.showerror(io::IO, err::SamplePostprocessingError)
    print(io, "sample postprocessing failed at rank $(err.rank)")
    try
        print(io, " for state ", QUBOTools.state(err.sample))
        print(io, " with raw value ", QUBOTools.value(err.sample))
    catch
    end
    print(io, "\nOriginal error: ")
    showerror(io, err.cause)
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
    iqm_runtime::Union{Nothing,RuntimeDiagnostic} = nothing
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
        return _py_executable_hook[]()
    catch
        return nothing
    end
end

function _import_python_module(module_name::AbstractString, package_name::AbstractString)
    try
        return _py_import_hook[](String(module_name))
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
const qiskit                         = LazyPythonModule("qiskit", "qiskit")
const qiskit_optimization            = LazyPythonModule("qiskit_optimization", "qiskit-optimization")
const qiskit_optimization_algorithms = LazyPythonModule("qiskit_optimization.algorithms", "qiskit-optimization")
const qiskit_optimization_utils      = LazyPythonModule("qiskit_optimization.utils", "qiskit-optimization")
const qiskit_ibm_runtime             = LazyPythonModule("qiskit_ibm_runtime", "qiskit-ibm-runtime")
const qiskit_aer                     = LazyPythonModule("qiskit_aer", "qiskit-aer")
const iqm_qiskit                     = LazyPythonModule("iqm.qiskit_iqm", "iqm-client[qiskit]")
const scipy                          = LazyPythonModule("scipy", "scipy")
const numpy                          = LazyPythonModule("numpy", "numpy")
const _runtime_service_builder = Ref{Function}()
const _iqm_provider_builder    = Ref{Function}()
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
        (
            qiskit,
            qiskit_optimization,
            qiskit_optimization_algorithms,
            qiskit_optimization_utils,
            qiskit_ibm_runtime,
            qiskit_aer,
            iqm_qiskit,
            scipy,
            numpy,
        ),
    )
    _runtime_service_builder[] = default_runtime_service
    _iqm_provider_builder[] = default_iqm_provider
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
        return _py_str(getproperty(py_module, :__version__))
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
        version=_py_version_hook[](),
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
    if !isnothing(result.iqm_runtime)
        _print_runtime_diagnostic(io, "IQM Runtime", result.iqm_runtime)
    end
    return nothing
end

function _diagnose_native_qiskit()
    ver = something(_module_version(Qiskit), "0.6.0")
    return RuntimeDiagnostic(
        name = "Qiskit",
        ok = true,
        version = ver,
        message = "Native Qiskit.jl C API bindings loaded",
    )
end

function _diagnose_native_ibm_runtime()
    ver = something(_module_version(QiskitIBMRuntime), "0.3.0")
    return RuntimeDiagnostic(
        name = "qiskit_ibm_runtime",
        ok = true,
        version = ver,
        message = "Native QiskitIBMRuntime.jl C API bindings loaded",
    )
end

function check_runtime(;
    local_backend::Bool=true,
    require_aer::Bool=false,
    ibm::Bool=false,
    iqm::Bool=false,
    verbose::Bool=true,
    kwargs...
)
    unknown_keywords = setdiff(collect(keys(kwargs)), [:local])
    if !isempty(unknown_keywords)
        throw(ArgumentError("unsupported check_runtime keyword(s): $(join(unknown_keywords, ", "))"))
    end
    if haskey(kwargs, :local)
        kwargs[:local] isa Bool || throw(ArgumentError("check_runtime local keyword must be a Bool"))
        local_backend = kwargs[:local]
    end

    julia = _diagnose_julia_package()
    qiskit_diag = _diagnose_native_qiskit()
    qiskit_ver = qiskit_diag.version
    pythoncall = RuntimeDiagnostic(
        name="PythonCall",
        ok=true,
        version=something(python_executable(), "native_c_api"),
        message="Native Qiskit.jl execution active (PythonCall optional extension)",
    )
    packages = Dict{String,RuntimeDiagnostic}(
        "Qiskit" => qiskit_diag,
        "QiskitIBMRuntime" => RuntimeDiagnostic(
            name="QiskitIBMRuntime",
            ok=true,
            version=something(_module_version(QiskitIBMRuntime), "0.3.0"),
            message="Native QiskitIBMRuntime.jl C API bindings loaded",
        ),
        "qiskit" => RuntimeDiagnostic(name="qiskit", ok=true, version=qiskit_ver, message="Native Qiskit.jl"),
        "qiskit_aer" => RuntimeDiagnostic(name="qiskit_aer", ok=true, version=qiskit_ver, message="Pure-Julia simulator"),
        "qiskit_optimization" => RuntimeDiagnostic(name="qiskit_optimization", ok=true, version=julia.version, message="Pure-Julia Ising/GAS lowering"),
        "scipy" => RuntimeDiagnostic(name="scipy", ok=true, version=string(VERSION), message="Pure-Julia derivative-free optimizer"),
        "numpy" => RuntimeDiagnostic(name="numpy", ok=true, version=string(VERSION), message="Base Julia LinearAlgebra"),
    )

    ibm_runtime = ibm ? _diagnose_native_ibm_runtime() : nothing

    iqm_runtime = if iqm
        _diagnose_python_module(iqm_qiskit)
    else
        nothing
    end

    local_backend_diagnostic = local_backend ? _diagnose_local_backend() : nothing

    diagnostics = RuntimeDiagnostic[julia, pythoncall]
    append!(diagnostics, values(packages))
    if !isnothing(ibm_runtime)
        push!(diagnostics, ibm_runtime)
    end
    if !isnothing(iqm_runtime)
        push!(diagnostics, iqm_runtime)
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
        iqm_runtime=iqm_runtime,
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

struct NativeRuntimeJob
    service::QiskitIBMRuntime.Service
    job::QiskitIBMRuntime.Job
end

function Base.getproperty(rj::NativeRuntimeJob, sym::Symbol)
    if sym === :job_id
        return () -> nothing
    elseif sym === :status
        return () -> string(QiskitIBMRuntime.get_job_status(getfield(rj, :job), getfield(rj, :service)))
    elseif sym === :samples
        return (; poll_interval = 1) -> QiskitIBMRuntime.get_sampler_job_results(
            getfield(rj, :job),
            getfield(rj, :service);
            poll_interval = poll_interval,
        )
    elseif sym === :result
        return (; poll_interval = 1) -> begin
            s = QiskitIBMRuntime.get_sampler_job_results(
                getfield(rj, :job),
                getfield(rj, :service);
                poll_interval = poll_interval,
            )
            counts = bitvectors_to_counts(s)
            (get_counts = () -> counts, samples = s, counts = s)
        end
    end
    return getfield(rj, sym)
end

Base.propertynames(::NativeRuntimeJob; private::Bool = false) =
    (:service, :job, :job_id, :status, :samples, :result)

struct NativeRuntimeBackend
    service::QiskitIBMRuntime.Service
    backend::QiskitIBMRuntime.Backend
    target::Qiskit.Target
    name::String
    version::String
end

function Base.getproperty(rb::NativeRuntimeBackend, sym::Symbol)
    if sym === :backend_version
        return getfield(rb, :version)
    elseif sym === :num_qubits
        return Int(getfield(rb, :target).num_qubits)
    elseif sym === :submit_sampler
        return (qc::Qiskit.QuantumCircuit, shots::Integer) -> NativeRuntimeJob(
            getfield(rb, :service),
            QiskitIBMRuntime.run_sampler_job(
                getfield(rb, :service),
                getfield(rb, :backend),
                qc,
                Int(shots),
            ),
        )
    elseif sym === :run
        return (qc::Qiskit.QuantumCircuit; shots::Integer = 1024) -> begin
            raw_job = QiskitIBMRuntime.run_sampler_job(
                getfield(rb, :service),
                getfield(rb, :backend),
                qc,
                Int(shots),
            )
            rj = NativeRuntimeJob(getfield(rb, :service), raw_job)
            s = QiskitIBMRuntime.get_sampler_job_results(raw_job, getfield(rb, :service))
            (job = rj, samples = s, counts = s, get_counts = () -> bitvectors_to_counts(s))
        end
    end
    return getfield(rb, sym)
end

Base.propertynames(::NativeRuntimeBackend; private::Bool = false) =
    (:service, :backend, :target, :name, :version, :backend_version, :num_qubits, :submit_sampler, :run)

struct NativeRuntimeService
    service::QiskitIBMRuntime.Service
    channel::Union{Nothing,String}
    instance::Union{Nothing,String}
end

function _resolve_native_runtime_backend(
    rs::NativeRuntimeService,
    backend_name_str::AbstractString,
)::NativeRuntimeBackend
    svc = getfield(rs, :service)
    inst = getfield(rs, :instance)
    sresults = QiskitIBMRuntime.backend_search(svc)
    isempty(sresults) && throw(ErrorException("No IBM Quantum backends available for the current service."))
    req = strip(String(backend_name_str))
    chosen = if req == "least_busy"
        QiskitIBMRuntime.least_busy(sresults)
    else
        matched = nothing
        for b in sresults
            if b.name == req
                if inst === nothing || b.instance_name == inst || b.instance_crn == inst
                    matched = b
                    break
                elseif matched === nothing
                    matched = b
                end
            end
        end
        matched === nothing && throw(ArgumentError("IBM Quantum backend '$req' not found in backend_search results."))
        matched
    end
    target = QiskitIBMRuntime.target_from_backend(chosen, svc)
    ver = string(QiskitIBMRuntime.qkrtc_version())
    return NativeRuntimeBackend(svc, chosen, target, chosen.name, ver)
end

function Base.getproperty(rs::NativeRuntimeService, sym::Symbol)
    if sym === :backend
        return (name::AbstractString) -> _resolve_native_runtime_backend(rs, name)
    end
    return getfield(rs, sym)
end

function default_runtime_service(;
    channel::Union{Nothing,AbstractString} = nothing,
    token::Union{Nothing,AbstractString} = nothing,
    instance::Union{Nothing,AbstractString} = nothing,
)
    resolved_channel = runtime_channel(channel)
    resolved_instance = runtime_instance(instance)
    svc = QiskitIBMRuntime.Service()
    return NativeRuntimeService(svc, resolved_channel, resolved_instance)
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

const _IQM_FAKE_BACKENDS = (
    "IQMFakeAdonis",
    "IQMFakeApollo",
    "IQMFakeAphrodite",
    "IQMFakeDeneb",
    "IQMFakeGarnet",
)

function iqm_server_url(server_url::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(server_url)
    !isnothing(provided) && return provided
    return nonempty_or_nothing(get(ENV, "IQM_SERVER_URL", nothing))
end

function iqm_token(token::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(token)
    !isnothing(provided) && return provided
    return nonempty_or_nothing(get(ENV, "IQM_TOKEN", nothing))
end

function iqm_tokens_file(tokens_file::Union{Nothing,AbstractString} = nothing)
    provided = nonempty_or_nothing(tokens_file)
    !isnothing(provided) && return provided
    return nonempty_or_nothing(get(ENV, "IQM_TOKENS_FILE", nothing))
end

function default_iqm_provider(;
    server_url::Union{Nothing,AbstractString} = nothing,
    quantum_computer::Union{Nothing,AbstractString} = nothing,
    token::Union{Nothing,AbstractString} = nothing,
    tokens_file::Union{Nothing,AbstractString} = nothing,
)
    resolved_url = iqm_server_url(server_url)
    isnothing(resolved_url) &&
        throw(ArgumentError("IQM server_url must be provided or set via IQM_SERVER_URL"))

    iqm_mod = iqm_qiskit()
    kwargs = Pair{Symbol,Any}[]
    _push_option!(kwargs, :quantum_computer, nonempty_or_nothing(quantum_computer))
    _push_option!(kwargs, :token, iqm_token(token))
    _push_option!(kwargs, :tokens_file, iqm_tokens_file(tokens_file))

    return iqm_mod.IQMProvider(resolved_url; kwargs...)
end

function iqm_provider(;
    server_url::Union{Nothing,AbstractString} = nothing,
    quantum_computer::Union{Nothing,AbstractString} = nothing,
    token::Union{Nothing,AbstractString} = nothing,
    tokens_file::Union{Nothing,AbstractString} = nothing,
)
    return _iqm_provider_builder[](
        server_url=iqm_server_url(server_url),
        quantum_computer=nonempty_or_nothing(quantum_computer),
        token=iqm_token(token),
        tokens_file=iqm_tokens_file(tokens_file),
    )
end

function iqm_fake_backend(name::Union{AbstractString,Symbol} = "IQMFakeAdonis")
    label = strip(String(name))
    isempty(label) && throw(ArgumentError("IQM fake backend name must be non-empty"))
    normalized = startswith(label, "IQMFake") ? label : "IQMFake" * uppercasefirst(lowercase(label))
    iqm_mod = iqm_qiskit()
    factory = getproperty(iqm_mod, Symbol(normalized))
    return factory()
end

function iqm_backend(
    backend_or_url = nothing;
    server_url::Union{Nothing,AbstractString} = nothing,
    quantum_computer::Union{Nothing,AbstractString} = nothing,
    token::Union{Nothing,AbstractString} = nothing,
    tokens_file::Union{Nothing,AbstractString} = nothing,
)
    if backend_or_url isa Function
        return backend_or_url()
    elseif backend_or_url isa Symbol || backend_or_url isa AbstractString
        label = strip(String(backend_or_url))
        isempty(label) && throw(ArgumentError("IQM backend specification must be non-empty"))
        if label in _IQM_FAKE_BACKENDS || startswith(lowercase(label), "fake_") || startswith(label, "IQMFake")
            clean_name = startswith(lowercase(label), "fake_") ? label[6:end] : label
            return iqm_fake_backend(clean_name)
        end
        resolved_url = isnothing(server_url) ? label : server_url
        provider = iqm_provider(
            server_url=resolved_url,
            quantum_computer=quantum_computer,
            token=token,
            tokens_file=tokens_file,
        )
        return provider.get_backend()
    elseif isnothing(backend_or_url)
        provider = iqm_provider(
            server_url=server_url,
            quantum_computer=quantum_computer,
            token=token,
            tokens_file=tokens_file,
        )
        return provider.get_backend()
    end

    return backend_or_url
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
    _push_option!(
        options,
        :matrix_product_state_truncation_threshold,
        config.mps_truncation_threshold,
    )
    _push_option!(
        options,
        :matrix_product_state_max_bond_dimension,
        config.mps_max_bond_dimension,
    )
    _push_option!(options, :mps_sample_measure_algorithm, config.mps_sample_measure_algorithm)
    _push_option!(options, :seed_simulator, config.seed_simulator)

    names = Tuple(first.(options))
    values = Tuple(last.(options))
    return NamedTuple{names}(values)
end

struct NativeLocalBackend
    name::String
    version::String
    config::AerBackendConfig
    target::Union{Nothing,Qiskit.Target}
    options::NamedTuple
end

function Base.getproperty(b::NativeLocalBackend, sym::Symbol)
    if sym === :backend_version
        return getfield(b, :version)
    elseif sym === :num_qubits
        t = getfield(b, :target)
        return t === nothing ? nothing : Int(t.num_qubits)
    elseif sym === :run
        return (qc::Qiskit.QuantumCircuit; shots::Integer = 1024) -> begin
            cfg = getfield(b, :config)
            counts = sample_from_circuit(
                qc;
                shots = Int(shots),
                seed = cfg.seed_simulator,
                precision = something(cfg.precision, "double"),
                max_bond_dimension = cfg.mps_max_bond_dimension,
                truncation_threshold = cfg.mps_truncation_threshold,
            )
            (
                job_id = () -> "local_simulator_job",
                status = () -> "DONE",
                result = () -> (get_counts = () -> counts, counts = counts),
                get_counts = () -> counts,
                counts = counts,
            )
        end
    end
    return getfield(b, sym)
end

Base.propertynames(::NativeLocalBackend; private::Bool = false) =
    (:name, :version, :config, :target, :options, :backend_version, :num_qubits, :run)

struct NativeFakeBackend
    name::String
    version::String
    num_qubits::Int
    target::Qiskit.Target
end

function fake_backend(name::AbstractString = "fake_manila"; num_qubits::Integer = 5)
    n = max(1, Int(num_qubits))
    return NativeFakeBackend(
        String(name),
        something(_module_version(Qiskit), "0.6.0"),
        n,
        default_local_target(n),
    )
end

function Base.getproperty(b::NativeFakeBackend, sym::Symbol)
    if sym === :backend_version
        return getfield(b, :version)
    elseif sym === :run
        return (qc::Qiskit.QuantumCircuit; shots::Integer = 1024) -> begin
            counts = sample_from_circuit(qc; shots = Int(shots))
            (
                job_id = () -> "fake_backend_job",
                status = () -> "DONE",
                result = () -> (get_counts = () -> counts, counts = counts),
                get_counts = () -> counts,
                counts = counts,
            )
        end
    end
    return getfield(b, sym)
end

Base.propertynames(::NativeFakeBackend; private::Bool = false) =
    (:name, :version, :num_qubits, :target, :backend_version, :run)

function _extract_backend_target(backend)::Union{Nothing,Qiskit.Target}
    backend isa Qiskit.Target && return backend
    if !_is_py(backend) && hasproperty(backend, :target)
        try
            t = getproperty(backend, :target)
            t isa Qiskit.Target && return t
        catch
        end
    end
    return nothing
end

function local_aer_backend(
    config::AerBackendConfig = AerBackendConfig();
    target::Union{Nothing,Qiskit.Target} = nothing,
)
    method_str = something(config.method, DEFAULT_AER_BACKEND_METHOD)
    bname = method_str == "automatic" ? "aer_simulator" : "aer_simulator_$(method_str)"
    ver = something(_module_version(Qiskit), "0.6.0")
    return NativeLocalBackend(bname, ver, config, target, aer_backend_options(config))
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
    rname = backend_name(remote_backend)
    rtarget = _extract_backend_target(remote_backend)
    ver = something(_module_version(Qiskit), "0.6.0")
    return NativeLocalBackend("aer_simulator($(rname))", ver, config, rtarget, aer_backend_options(config))
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
    if backend === nothing || backend isa Qiskit.Target
        return NativePassManager(backend, Int(optimization_level), seed_transpiler === nothing ? nothing : Int(seed_transpiler))
    elseif !_is_py(backend) && hasproperty(backend, :target) && getproperty(backend, :target) isa Union{Nothing,Qiskit.Target}
        return NativePassManager(getproperty(backend, :target), Int(optimization_level), seed_transpiler === nothing ? nothing : Int(seed_transpiler))
    end

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
    for property in (:name, :backend_name)
        try
            val = getproperty(backend, property)
            if val isa AbstractString
                return String(val)
            elseif val isa Function
                res = val()
                res isa AbstractString && return String(res)
            end
            try
                return _py_str(val)
            catch
                return _py_str(val())
            end
        catch
        end
    end
    return string(typeof(backend))
end

function python_type_module(object)
    try
        return _py_str(getproperty(getproperty(object, :__class__), :__module__))
    catch
        return nothing
    end
end

function is_qiskit_aer_backend(backend)
    backend isa NativeLocalBackend && return true
    module_name = python_type_module(backend)
    return !isnothing(module_name) && startswith(module_name, "qiskit_aer")
end

function backend_version(backend)
    if backend isa NativeLocalBackend || backend isa NativeFakeBackend || backend isa NativeRuntimeBackend
        return backend.version
    end
    if is_qiskit_aer_backend(backend)
        aer_version = _python_module_version(qiskit_aer())
        !isnothing(aer_version) && return aer_version
    end

    for property in (:backend_version, :version)
        try
            val = getproperty(backend, property)
            val isa AbstractString && return String(val)
            val isa VersionNumber && return string(val)
            return _py_str(val)
        catch
        end
    end

    return nothing
end

function execution_status(execution_mode::AbstractString)
    mode = String(execution_mode)
    if mode == "local"
        return "locally_solved"
    elseif mode == "cloud"
        return "cloud_solved"
    end

    return "$(mode)_solved"
end

function sample_bits(key::AbstractString)
    bitstring = replace(String(key), " " => "")
    return reverse(Int[(digit == '1') for digit in bitstring])
end

function sample_bits(key)
    bitstring = replace(_py_str(key), " " => "")
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
    seed_sampler = nothing,
    seed_simulator = nothing,
    seed_transpiler = nothing,
    seed_optimizer = nothing,
)
    seeds = Dict{String,Any}(
        "simulator" => seed_simulator,
        "transpiler" => seed_transpiler,
        "optimizer" => seed_optimizer,
    )
    isnothing(seed_sampler) || (seeds["sampler"] = seed_sampler)
    return seeds
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

function derived_seed(seed::Union{Integer,Nothing}, stream::Integer)
    isnothing(seed) && return nothing
    stream >= 1 || throw(ArgumentError("seed stream must be at least 1"))

    state = _seed_state(seed)
    value = UInt64(0)
    for _ in 1:stream
        state, value = _splitmix64_next(state)
    end

    return Int(mod(BigInt(value), BigInt(typemax(Int32))))
end

function effective_seed(explicit_seed, sampler_seed::Union{Integer,Nothing}, stream::Integer)
    return isnothing(explicit_seed) ? derived_seed(sampler_seed, stream) : explicit_seed
end

function python_int_property(object, name::Symbol)
    try
        val = getproperty(object, name)
        val isa Integer && return Int(val)
        val === nothing && return nothing
        return _py_int(val)
    catch
        return nothing
    end
end

function python_float_property(object, name::Symbol)
    try
        val = getproperty(object, name)
        val isa Real && return Float64(val)
        val === nothing && return nothing
        return _py_float(val)
    catch
        return nothing
    end
end

function python_bool_property(object, name::Symbol)
    try
        val = getproperty(object, name)
        val isa Bool && return val
        val === nothing && return nothing
        return _py_bool(val)
    catch
        return nothing
    end
end

function python_string_property(object, name::Symbol)
    try
        val = getproperty(object, name)
        val isa AbstractString && return String(val)
        val === nothing && return nothing
        return _py_str(val)
    catch
        return nothing
    end
end

function python_float_vector(object::AbstractVector{<:Real})
    return Float64[Float64(v) for v in object]
end

function python_float_vector(object)
    return _py_float_vec_hook[](object)
end

qiskit_parameter_names(ansatz::NativeAnsatz) = copy(ansatz.parameter_names)
qiskit_parameter_names(::Qiskit.QuantumCircuit) = String[]

function qiskit_parameter_names(circuit)
    return String[_py_str(parameter) for parameter in circuit.parameters]
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

function optimizer_result_metadata(
    result;
    iterations = python_int_property(result, :nit),
    function_evaluations = python_int_property(result, :nfev),
)
    return Dict{String,Any}(
        "success" => python_bool_property(result, :success),
        "status" => python_int_property(result, :status),
        "message" => python_string_property(result, :message),
        "objective_value" => python_float_property(result, :fun),
        "iterations" => iterations,
        "function_evaluations" => function_evaluations,
    )
end

function optimized_parameter_metadata(
    values::AbstractVector,
    names::AbstractVector{<:AbstractString},
    optimizer::Union{AbstractDict,Nothing},
)
    return Dict{String,Any}(
        "source" => "optimizer_result",
        "parameter_names" => String.(names),
        "values" => Float64.(values),
        "optimizer" => isnothing(optimizer) ? Dict{String,Any}() : Dict{String,Any}(optimizer),
    )
end

function _serializable_postprocessing_value(value, path::AbstractString)
    if value === nothing || value isa Bool || value isa AbstractString || value isa Integer
        return value
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("postprocessing value at $(path) must be finite"))
        return value
    elseif value isa Symbol || value isa VersionNumber
        return string(value)
    elseif value isa AbstractDict || value isa NamedTuple
        return _string_key_dict(value, path)
    elseif value isa AbstractArray || value isa Tuple
        return Any[
            _serializable_postprocessing_value(item, "$(path)[$(index)]") for
            (index, item) in enumerate(value)
        ]
    end

    throw(
        ArgumentError(
            "postprocessing value at $(path) must be serializable; got $(typeof(value))",
        ),
    )
end

function _string_key_dict(data::AbstractDict, path::AbstractString = "metadata")
    return Dict{String,Any}(
        string(key) => _serializable_postprocessing_value(value, "$(path).$(string(key))") for
        (key, value) in data
    )
end

function _string_key_dict(data::NamedTuple, path::AbstractString = "metadata")
    return Dict{String,Any}(
        string(key) => _serializable_postprocessing_value(value, "$(path).$(string(key))") for
        (key, value) in pairs(data)
    )
end

function _derived_sample_fields(result)
    if result isa AbstractDict || result isa NamedTuple
        return _string_key_dict(result, "derived")
    end

    throw(
        ArgumentError(
            "sample postprocessor must return an AbstractDict or NamedTuple; got $(typeof(result))",
        ),
    )
end

function _postprocessing_metadata(metadata)
    isnothing(metadata) && return Dict{String,Any}()
    if metadata isa AbstractDict || metadata isa NamedTuple
        return _string_key_dict(metadata)
    end

    throw(
        ArgumentError(
            "postprocessing metadata must be an AbstractDict, NamedTuple, or nothing; " *
            "got $(typeof(metadata))",
        ),
    )
end

function _copied_sampleset(sampleset::QUBOTools.SampleSet{T,U}) where {T,U}
    samples = QUBOTools.Sample{T,U}[
        QUBOTools.Sample{T,U}(
            copy(QUBOTools.state(sample)),
            QUBOTools.value(sample),
            QUBOTools.reads(sample),
        ) for sample in sampleset
    ]

    return QUBOTools.SampleSet{T,U}(
        samples;
        metadata = deepcopy(QUBOTools.metadata(sampleset)),
        sense    = QUBOTools.sense(sampleset),
        domain   = QUBOTools.domain(sampleset),
    )
end

function _postprocessing_context(
    sample,
    rank::Integer,
    total_reads::Integer,
    metadata_snapshot::Dict{String,Any},
)
    reads = QUBOTools.reads(sample)
    probability = iszero(total_reads) ? 0.0 : reads / total_reads

    return (
        rank = Int(rank),
        state = copy(QUBOTools.state(sample)),
        raw_value = QUBOTools.value(sample),
        reads = reads,
        probability = probability,
        total_reads = total_reads,
        metadata = metadata_snapshot,
    )
end

function _postprocessing_row(context)
    state = copy(context.state)

    return Dict{String,Any}(
        "rank" => context.rank,
        "state" => state,
        "bitstring" => join(string.(state)),
        "raw_value" => context.raw_value,
        "reads" => context.reads,
        "probability" => context.probability,
        "derived" => Dict{String,Any}(),
    )
end

"""
    postprocess_samples(postprocessor, sampleset; kwargs...)
    postprocess_samples(sampleset; kwargs...) do sample, context

Return a new `SampleSet` with raw samples copied unchanged and user-defined
derived fields attached under `metadata["postprocessing"]`.

The callback receives a copied sample and a context named tuple with `rank`,
`state`, `raw_value`, `reads`, `probability`, `total_reads`, and the original
metadata. `context.metadata` is a shared read-only snapshot; do not mutate it
from the callback. The callback must return an `AbstractDict` or `NamedTuple`;
keys are stringified for serialization-friendly metadata. Callback failures are
rethrown as `SamplePostprocessingError` so raw quantum results are not silently
corrupted.
"""
function postprocess_samples(
    postprocessor::Function,
    sampleset::QUBOTools.SampleSet{T,U};
    enabled::Bool = true,
    name::AbstractString = "anonymous",
    version = nothing,
    source_problem = nothing,
    scoring_convention = nothing,
    metadata = Dict{String,Any}(),
) where {T,U}
    processed = _copied_sampleset(sampleset)
    enabled || return processed

    aggregate_metadata = _postprocessing_metadata(metadata)
    source_metadata = deepcopy(QUBOTools.metadata(sampleset))
    total_reads = sum(QUBOTools.reads(sample) for sample in sampleset; init = 0)
    rows = Vector{Dict{String,Any}}()
    sizehint!(rows, length(sampleset))

    for (rank, sample) in enumerate(sampleset)
        callback_sample = QUBOTools.Sample{T,U}(
            copy(QUBOTools.state(sample)),
            QUBOTools.value(sample),
            QUBOTools.reads(sample),
        )
        context = _postprocessing_context(callback_sample, rank, total_reads, source_metadata)
        row = _postprocessing_row(context)

        derived = try
            _derived_sample_fields(postprocessor(callback_sample, context))
        catch err
            throw(SamplePostprocessingError(rank, callback_sample, err))
        end

        row["derived"] = derived
        push!(rows, row)
    end

    postprocessing = Dict{String,Any}(
        "enabled" => true,
        "name" => String(name),
        "rows" => rows,
        "row_count" => length(rows),
        "metadata" => aggregate_metadata,
    )
    isnothing(version) || (postprocessing["version"] = string(version))
    isnothing(source_problem) || (postprocessing["source_problem"] = string(source_problem))
    isnothing(scoring_convention) ||
        (postprocessing["scoring_convention"] = string(scoring_convention))

    QUBOTools.metadata(processed)["postprocessing"] = postprocessing

    return processed
end

function empty_metadata(
    algorithm::AbstractString,
    backend::AbstractString,
    execution_mode::AbstractString;
    backend_version = nothing,
    backend_config::Union{AerBackendConfig,Nothing} = nothing,
    backend_config_source::Union{AbstractString,Nothing} = nothing,
    number_of_reads::Union{Integer,Nothing} = nothing,
    final_number_of_reads::Union{Integer,Nothing} = number_of_reads,
    optimizer_iterations = nothing,
    optimizer_evaluations = nothing,
    optimizer_number_of_reads = nothing,
    seed_sampler = nothing,
    seed_transpiler = nothing,
    seed_optimizer = nothing,
    pass_manager_source::Union{AbstractString,Nothing} = nothing,
    pass_manager_optimization_level::Union{Integer,Nothing} = nothing,
    initial_parameters::Union{AbstractVector,Nothing} = nothing,
    initial_parameter_names::Union{AbstractVector{<:AbstractString},Nothing} = nothing,
    initial_parameter_source::Union{AbstractString,Nothing} = nothing,
    optimized_parameters::Union{AbstractVector,Nothing} = nothing,
    optimized_parameter_names::Union{AbstractVector{<:AbstractString},Nothing} = nothing,
    optimized_parameter_optimizer::Union{AbstractDict,Nothing} = nothing,
)
    metadata = Dict{String,Any}(
        "origin" => "$(algorithm) @ $(backend)",
        "algorithm" => Dict{String,Any}(
            "name" => String(algorithm),
        ),
        "backend_name" => String(backend),
        "backend" => Dict{String,Any}(
            "name" => String(backend),
            "version" => backend_version,
        ),
        "execution" => Dict{String,Any}(
            "mode" => String(execution_mode),
        ),
        "execution_mode" => execution_mode,
        "optimizer" => Dict{String,Any}(
            "iterations" => optimizer_iterations,
            "evaluations" => optimizer_evaluations,
            "number_of_reads" => optimizer_number_of_reads,
        ),
        "reads" => Dict{String,Any}(
            "number_of_reads" => final_number_of_reads,
            "final_number_of_reads" => final_number_of_reads,
        ),
        "time" => Dict{String,Any}(),
        "evals" => Float64[],
        "seeds" => seed_metadata(
            seed_sampler=seed_sampler,
            seed_simulator=isnothing(backend_config) ? nothing : backend_config.seed_simulator,
            seed_transpiler=seed_transpiler,
            seed_optimizer=seed_optimizer,
        ),
        "status" => execution_status(execution_mode),
    )

    if !isnothing(backend_config) || !isnothing(backend_config_source)
        source = isnothing(backend_config_source) ? "unknown" : backend_config_source
        metadata["backend_configuration"] = if isnothing(backend_config)
            Dict{String,Any}("source" => String(source))
        else
            aer_backend_metadata(backend_config, source)
        end
    end

    if !isnothing(pass_manager_source)
        pass_manager_metadata = Dict{String,Any}("source" => String(pass_manager_source))
        if !isnothing(pass_manager_optimization_level)
            pass_manager_metadata["optimization_level"] = pass_manager_optimization_level
        end
        metadata["pass_manager"] = pass_manager_metadata
    end

    if !isnothing(initial_parameters)
        source = isnothing(initial_parameter_source) ? "unknown" : initial_parameter_source
        names = isnothing(initial_parameter_names) ? String[] : initial_parameter_names
        metadata["initial_parameters"] = initial_parameter_metadata(source, initial_parameters, names)
    end

    if !isnothing(optimized_parameters)
        names = isnothing(optimized_parameter_names) ? String[] : optimized_parameter_names
        metadata["optimized_parameters"] = optimized_parameter_metadata(
            optimized_parameters,
            names,
            optimized_parameter_optimizer,
        )
    end

    return metadata
end

struct IsingQuadraticProgramResult
    observable::Qiskit.SparseObservable
    offset::Float64
    hamiltonian::IsingHamiltonian
end

function Base.getindex(r::IsingQuadraticProgramResult, idx::Integer)
    if idx == 0 || idx == 1
        return r.observable
    elseif idx == 2
        return r.offset
    elseif idx == 3
        return r.hamiltonian
    end
    throw(BoundsError(r, idx))
end

Base.length(::IsingQuadraticProgramResult) = 2
Base.iterate(r::IsingQuadraticProgramResult, state::Int = 1) =
    state == 1 ? (r.observable, 2) : (state == 2 ? (r.offset, 3) : nothing)

struct NativeQuadraticProgram
    num_variables::Int
    linear::Vector{Float64}
    quadratic::Matrix{Float64}
    sense::MOI.OptimizationSense
    ising::IsingHamiltonian
end

function Base.getproperty(qp::NativeQuadraticProgram, sym::Symbol)
    if sym === :get_num_binary_vars
        return () -> getfield(qp, :num_variables)
    elseif sym === :to_ising
        return () -> begin
            h = getfield(qp, :ising)
            IsingQuadraticProgramResult(to_sparse_observable(h), h.offset, h)
        end
    end
    return getfield(qp, sym)
end

function build_quadratic_program(
    n::Integer,
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real};
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
)
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    length(L) == n || throw(ArgumentError("L must have length $(n)"))
    size(Q) == (n, n) || throw(ArgumentError("Q must have size ($(n), $(n))"))

    ising = qubo_to_ising(Int(n), L, Q; sense = sense)
    return NativeQuadraticProgram(
        Int(n),
        Float64[Float64(v) for v in L],
        Float64[Float64(Q[i, j]) for i in 1:n, j in 1:n],
        sense,
        ising,
    )
end

function build_quadratic_program(sampler::QUBODrivers.AbstractSampler{T}) where {T}
    n, L, Q, _, _ = QUBOTools.qubo(sampler, :dense)
    sense = MOI.get(sampler, MOI.ObjectiveSense())
    return build_quadratic_program(n, L, Q; sense = sense)
end

function quadratic_program(sampler::QUBODrivers.AbstractSampler{T}) where {T}
    return build_quadratic_program(sampler).to_ising()
end

export VQE, QAOA, GAS, PCE, check_runtime, postprocess_samples, SamplePostprocessingError

include("QAOA.jl")
include("VQE.jl")
include("GAS.jl")
include("PCE.jl")

end # module QiskitOpt
