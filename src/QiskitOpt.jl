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
            message="Usable backend: $(backend_name(backend))",
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
        packages[diagnostic.name] = diagnostic
        diagnostic
    else
        nothing
    end

    local_backend_diagnostic = local_backend ? _diagnose_local_backend() : nothing

    diagnostics = RuntimeDiagnostic[julia, pythoncall]
    append!(diagnostics, values(packages))
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

function default_local_backend()
    return qiskit_aer().AerSimulator(method="matrix_product_state")
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

function empty_metadata(algorithm::AbstractString, backend::AbstractString, execution_mode::AbstractString)
    return Dict{String,Any}(
        "origin" => "$(algorithm) @ $(backend)",
        "backend" => backend,
        "execution_mode" => execution_mode,
        "time" => Dict{String,Any}(),
        "evals" => Float64[],
    )
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
