using Test
using QiskitOpt
using QiskitOpt: QAOA, QUBODrivers, VQE

MOI = QUBODrivers.MOI

const TEST_Q = [
    -1 2 2
    2 -1 2
    2 2 -1
]

const FIXED_VARIABLE_Q = [
    -1 2 0 1
    2 -1 2 0
    0 2 -1 2
    1 0 2 -1
]

# Mirrors the 36-variable shape from issue #8 without vendoring external CSV data.
const LARGE_LOCAL_Q = let Q = zeros(36, 36)
    for i in 1:35
        Q[i, i + 1] = isodd(i) ? -3.0 : 2.0
    end
    for i in 1:6:30
        Q[i, i + 5] = -1.5
    end
    Q
end

const LARGE_LOCAL_L = [
    39.7483,
    -42.7194,
    -42.7084,
    -85.3741,
    85.4966,
    -42.7483,
    175.9522,
    85.4966,
    48.9413,
    52.4653,
    -210.3895,
    91.6596,
    85.4966,
    86.2356,
    86.3366,
    86.7676,
    6.99,
    172.3262,
    0.0,
    -42.7483,
    -42.7483,
    -42.7483,
    -42.7483,
    -42.7483,
    -42.7483,
    -42.7483,
    -42.7483,
    -42.7483,
    -128.2449,
    -128.2449,
    128.2449,
    128.2449,
    128.2449,
    42.7483,
    42.7483,
    42.7483,
]

function python_module_loaded(name)
    sys = QiskitOpt.PythonCall.pyimport("sys")
    return QiskitOpt.PythonCall.pyconvert(Bool, sys.modules.__contains__(name))
end

struct MockRuntimeService
    backend::Function
end

function build_model(optimizer_factory; Q=TEST_Q, L=nothing, scale=1.0, offset=0.0, sense=:Min)
    model = MOI.instantiate(optimizer_factory; with_cache_type=Float64)
    num_variables = size(Q, 1)
    linear_coefficients = isnothing(L) ? zeros(num_variables) : L

    x = MOI.add_variables(model, num_variables)
    for variable in x
        MOI.add_constraint(model, variable, MOI.ZeroOne())
    end

    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:num_variables, j in 1:num_variables
        iszero(Q[i, j]) && continue
        coefficient = (i == j ? 2 : 1) * scale * Q[i, j]
        push!(quadratic_terms, MOI.ScalarQuadraticTerm(coefficient, x[i], x[j]))
    end

    linear_terms = MOI.ScalarAffineTerm{Float64}[]
    for i in 1:num_variables
        iszero(linear_coefficients[i]) && continue
        push!(linear_terms, MOI.ScalarAffineTerm(scale * linear_coefficients[i], x[i]))
    end

    objective = MOI.ScalarQuadraticFunction(quadratic_terms, linear_terms, scale * offset)
    objective_sense = sense == :Min ? MOI.MIN_SENSE : MOI.MAX_SENSE
    MOI.set(model, MOI.ObjectiveSense(), objective_sense)
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)

    return model, x
end

function assert_solution_consistency(model, x; Q=TEST_Q, L=nothing, scale=1.0, offset=0.0)
    result_count = MOI.get(model, MOI.ResultCount())
    @test result_count >= 1

    for result in 1:result_count
        assignment = round.(Int, MOI.get.(model, MOI.VariablePrimal(result), x))
        linear_coefficients = isnothing(L) ? zeros(length(assignment)) : L
        expected_objective =
            scale * (
                assignment' * Q * assignment +
                sum(linear_coefficients[i] * assignment[i] for i in eachindex(assignment)) +
                offset
            )
        observed_objective = MOI.get(model, MOI.ObjectiveValue(result))

        @test isapprox(observed_objective, expected_objective; atol=1.0e-6)
    end
end

function configure_solver!(model, optimizer_module; max_iterations=5, number_of_reads=64)
    MOI.set(model, optimizer_module.MaximumIterations(), max_iterations)
    MOI.set(model, optimizer_module.NumberOfReads(), number_of_reads)
end

function solve_locally_without_runtime_service(
    optimizer_factory,
    optimizer_module;
    Q=TEST_Q,
    L=nothing,
    scale=1.0,
    offset=0.0,
    sense=:Min,
    fixed_variables=Pair{Int,Int}[],
    max_iterations=5,
    number_of_reads=64,
)
    withenv(
        "QISKIT_IBM_TOKEN" => nothing,
        "QISKIT_IBM_INSTANCE" => nothing,
        "QISKIT_IBM_CHANNEL" => nothing,
        "IBMQ_API_TOKEN" => nothing,
        "IBMQ_INSTANCE" => nothing,
    ) do
        previous_builder = QiskitOpt._runtime_service_builder[]
        QiskitOpt._runtime_service_builder[] = (; kwargs...) -> error(
            "runtime service should not be created for local execution",
        )

        try
            model, x = build_model(
                optimizer_factory;
                Q=Q,
                L=L,
                scale=scale,
                offset=offset,
                sense=sense,
            )
            for (index, value) in fixed_variables
                MOI.add_constraint(model, x[index], MOI.EqualTo(Float64(value)))
            end
            configure_solver!(
                model,
                optimizer_module;
                max_iterations=max_iterations,
                number_of_reads=number_of_reads,
            )

            MOI.optimize!(model)
            assert_solution_consistency(model, x; Q=Q, L=L, scale=scale, offset=offset)
        finally
            QiskitOpt._runtime_service_builder[] = previous_builder
        end
    end
end

function run_qubodrivers_suite(optimizer_module)
    QUBODrivers.test(optimizer_module.Optimizer) do model
        MOI.set(model, optimizer_module.MaximumIterations(), 5)
        MOI.set(model, optimizer_module.NumberOfReads(), 64)
    end
end

@testset "Runtime diagnostics and lazy imports" begin
    @test !python_module_loaded("qiskit_ibm_runtime")

    local_diagnostics = QiskitOpt.check_runtime(; local_backend=true, ibm=false, verbose=false)
    @test local_diagnostics.ok
    @test local_diagnostics.packages["qiskit"].ok
    @test local_diagnostics.packages["qiskit_aer"].ok
    @test local_diagnostics.packages["qiskit_optimization"].ok
    @test !isnothing(local_diagnostics.packages["qiskit"].version)
    @test isnothing(local_diagnostics.ibm_runtime)
    @test !haskey(local_diagnostics.packages, "qiskit_ibm_runtime")
    @test !isnothing(local_diagnostics.local_backend)
    @test local_diagnostics.local_backend.ok
    @test occursin("qiskit_ibm_runtime", local_diagnostics.local_backend.message)
    @test !python_module_loaded("qiskit_ibm_runtime")

    missing_error = try
        QiskitOpt._import_python_module(
            "qiskitopt_missing_python_package_for_test",
            "qiskitopt-missing-python-package-for-test",
        )
    catch err
        err
    end
    @test missing_error isa QiskitOpt.PythonPackageError
    missing_message = sprint(showerror, missing_error)
    @test occursin("qiskitopt-missing-python-package-for-test", missing_message)
    @test occursin("Python executable:", missing_message)

    missing_diagnostic = QiskitOpt._diagnose_python_module(
        "qiskitopt_missing_python_package_for_test",
        "qiskitopt-missing-python-package-for-test",
    )
    @test !missing_diagnostic.ok
    @test occursin("qiskitopt-missing-python-package-for-test", missing_diagnostic.message)
    @test occursin("Python executable:", missing_diagnostic.message)

    pair_alias_diagnostics = QiskitOpt.check_runtime(; :local => false, ibm=false, verbose=false)
    @test isnothing(pair_alias_diagnostics.local_backend)

    ibm_diagnostics = QiskitOpt.check_runtime(; local_backend=false, ibm=true, verbose=false)
    @test !isnothing(ibm_diagnostics.ibm_runtime)
    @test !haskey(ibm_diagnostics.packages, "qiskit_ibm_runtime")
    @test ibm_diagnostics.ibm_runtime.ok
    @test ibm_diagnostics.ok
end

@testset "Local-first execution" begin
    solve_locally_without_runtime_service(QAOA.Optimizer, QAOA)
    solve_locally_without_runtime_service(VQE.Optimizer, VQE)
end

@testset "Max-sense objective reporting stays consistent" begin
    solve_locally_without_runtime_service(QAOA.Optimizer, QAOA; sense=:Max)
    solve_locally_without_runtime_service(VQE.Optimizer, VQE; sense=:Max)
end

@testset "Runtime configuration helpers preserve compatibility aliases" begin
    withenv(
        "QISKIT_IBM_TOKEN" => nothing,
        "QISKIT_IBM_INSTANCE" => nothing,
        "QISKIT_IBM_CHANNEL" => "ibm_quantum",
        "IBMQ_API_TOKEN" => "legacy-token",
        "IBMQ_INSTANCE" => "legacy-instance",
    ) do
        @test QiskitOpt.runtime_token() == "legacy-token"
        @test QiskitOpt.runtime_instance() == "legacy-instance"
        @test QiskitOpt.runtime_channel() == QiskitOpt.DEFAULT_RUNTIME_CHANNEL
        @test QiskitOpt.runtime_channel("ibm_quantum") == QiskitOpt.DEFAULT_RUNTIME_CHANNEL
    end
end

@testset "Runtime service stays opt-in" begin
    previous_builder = QiskitOpt._runtime_service_builder[]
    QiskitOpt._runtime_service_builder[] = (; kwargs...) -> error("runtime service invoked")

    try
        model, _ = build_model(QAOA.Optimizer)
        MOI.set(model, QAOA.IBMBackend(), "ibm_fez")
        MOI.set(model, QAOA.MaximumIterations(), 1)

        @test_throws ErrorException MOI.optimize!(model)
    finally
        QiskitOpt._runtime_service_builder[] = previous_builder
    end
end

@testset "Named backend local emulation uses env-configured runtime service" begin
    withenv(
        "QISKIT_IBM_TOKEN" => "token-from-env",
        "QISKIT_IBM_INSTANCE" => "instance-from-env",
        "QISKIT_IBM_CHANNEL" => "channel-from-env",
    ) do
        runtime_calls = NamedTuple[]
        previous_builder = QiskitOpt._runtime_service_builder[]
        QiskitOpt._runtime_service_builder[] = (; channel, token, instance) -> begin
            push!(runtime_calls, (channel=channel, token=token, instance=instance))
            return MockRuntimeService(_ -> QiskitOpt.qiskit_ibm_runtime.fake_provider.FakeManilaV2())
        end

        try
            model, x = build_model(VQE.Optimizer)
            MOI.set(model, VQE.IBMBackend(), "ibm_fez")
            MOI.set(model, VQE.IsLocal(), true)
            MOI.set(model, VQE.MaximumIterations(), 2)
            MOI.set(model, VQE.NumberOfReads(), 32)

            MOI.optimize!(model)
            assert_solution_consistency(model, x)

            @test length(runtime_calls) == 1
            @test runtime_calls[1] == (
                channel="channel-from-env",
                token="token-from-env",
                instance="instance-from-env",
            )
        finally
            QiskitOpt._runtime_service_builder[] = previous_builder
        end
    end
end

@testset "VQE handles fixed-variable reductions" begin
    solve_locally_without_runtime_service(
        VQE.Optimizer,
        VQE;
        Q=FIXED_VARIABLE_Q,
        fixed_variables=[4 => 1],
    )
end

@testset "Default local backend handles large QUBOs" begin
    solve_locally_without_runtime_service(
        QAOA.Optimizer,
        QAOA;
        Q=LARGE_LOCAL_Q,
        L=LARGE_LOCAL_L,
        offset=601.4762,
        max_iterations=1,
        number_of_reads=8,
    )
    solve_locally_without_runtime_service(
        VQE.Optimizer,
        VQE;
        Q=LARGE_LOCAL_Q,
        L=LARGE_LOCAL_L,
        offset=601.4762,
        max_iterations=1,
        number_of_reads=8,
    )
end

@testset "QUBODrivers compatibility suite" begin
    run_qubodrivers_suite(QAOA)
    run_qubodrivers_suite(VQE)
end
