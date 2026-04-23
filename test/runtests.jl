using JuMP
using Test
using QiskitOpt
using QiskitOpt: QAOA, QUBODrivers, VQE

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

struct MockRuntimeService
    backend::Function
end

function build_model(optimizer_factory; Q=TEST_Q, sense=:Min)
    model = Model(optimizer_factory)
    num_variables = size(Q, 1)

    @variable(model, x[1:num_variables], Bin)

    objective = sum(Q[i, j] * x[i] * x[j] for i in 1:num_variables, j in 1:num_variables)
    if sense == :Min
        @objective(model, Min, objective)
    else
        @objective(model, Max, objective)
    end

    return model, x
end

function assert_solution_consistency(model, x; Q=TEST_Q)
    @test result_count(model) >= 1

    for result in 1:result_count(model)
        assignment = round.(Int, value.(x; result=result))
        expected_objective = assignment' * Q * assignment
        observed_objective = objective_value(model; result=result)

        @test isapprox(observed_objective, expected_objective; atol=1.0e-6)
    end
end

function configure_solver!(model, optimizer_module; max_iterations=5, number_of_reads=64)
    set_attribute(model, optimizer_module.MaximumIterations(), max_iterations)
    set_attribute(model, optimizer_module.NumberOfReads(), number_of_reads)
end

function solve_locally_without_runtime_service(
    optimizer_factory,
    optimizer_module;
    Q=TEST_Q,
    sense=:Min,
    fixed_variables=Pair{Int,Int}[],
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
            model, x = build_model(optimizer_factory; Q=Q, sense=sense)
            for (index, value) in fixed_variables
                fix(x[index], value; force=true)
            end
            configure_solver!(model, optimizer_module)

            optimize!(model)
            assert_solution_consistency(model, x; Q=Q)
        finally
            QiskitOpt._runtime_service_builder[] = previous_builder
        end
    end
end

function run_qubodrivers_suite(optimizer_module)
    QUBODrivers.test(optimizer_module.Optimizer) do model
        QUBODrivers.MOI.set(model, optimizer_module.MaximumIterations(), 5)
        QUBODrivers.MOI.set(model, optimizer_module.NumberOfReads(), 64)
    end
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
        set_attribute(model, QAOA.IBMBackend(), "ibm_fez")
        set_attribute(model, QAOA.MaximumIterations(), 1)

        @test_throws ErrorException optimize!(model)
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
            set_attribute(model, VQE.IBMBackend(), "ibm_fez")
            set_attribute(model, VQE.IsLocal(), true)
            set_attribute(model, VQE.MaximumIterations(), 2)
            set_attribute(model, VQE.NumberOfReads(), 32)

            optimize!(model)
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

@testset "QUBODrivers compatibility suite" begin
    run_qubodrivers_suite(QAOA)
    run_qubodrivers_suite(VQE)
end
