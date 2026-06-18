using Test
using QiskitOpt
using QiskitOpt: QAOA, QUBODrivers, QUBOTools, VQE

include(joinpath(@__DIR__, "..", "examples", "ds_mfg_qubo", "DSMFGQUBOExample.jl"))

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

function sample_locally_without_runtime_service(
    optimizer_factory,
    optimizer_module,
    configure!::Function = (_, _) -> nothing;
    max_iterations=1,
    number_of_reads=16,
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
            model, _ = build_model(optimizer_factory)
            configure_solver!(
                model,
                optimizer_module;
                max_iterations=max_iterations,
                number_of_reads=number_of_reads,
            )
            configure!(model, optimizer_module)
            MOI.optimize!(model)
            return QUBOTools.solution(MOI.get(model, MOI.RawSolver()))
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

function expected_maxcut_value(statevector, edges)
    probabilities = QiskitOpt.PythonCall.pyconvert(
        Dict{String,Float64},
        statevector.probabilities_dict(),
    )

    expected_value = 0.0
    for (bitstring, probability) in probabilities
        bits = reverse(Int[digit == '1' for digit in bitstring])
        expected_value += probability * sum(bits[i] != bits[j] for (i, j) in edges)
    end

    return expected_value
end

@testset "DS-MFG example cached-data smoke" begin
    base_dir = joinpath(@__DIR__, "..", "examples", "ds_mfg_qubo")
    instance = DSMFGQUBOExample.load_instance(base_dir)

    @test instance.n_variables == 5
    @test instance.flow_indices == [1, 2, 3]
    @test instance.auxiliary_indices == [4, 5]
    @test DSMFGQUBOExample.qubo_energy(instance, [1, 1, 0, 1, 1]) ≈ 2.2
    @test DSMFGQUBOExample.projected_objective(instance, [1, 1, 0, 0, 0]) ≈ 2.2

    pool = DSMFGQUBOExample.load_solution_pool(instance, base_dir)
    observations = DSMFGQUBOExample.load_cached_distributions(base_dir)
    rows = DSMFGQUBOExample.annotate_distributions(instance, observations, pool)

    @test length(pool) == 3
    @test length(observations) == 12
    @test sum(row.reads for row in observations if row.algorithm == "QAOA") == 128
    @test sum(row.reads for row in observations if row.algorithm == "VQE") == 128
    @test any(row -> row.algorithm == "QAOA" && row.bitstring == "11011" && row.match == "global_optimum", rows)

    aux_mismatch = only(row for row in rows if row.algorithm == "QAOA" && row.bitstring == "11010")
    @test aux_mismatch.match == "projection_global_optimum"
    @test aux_mismatch.raw_qubo_energy > aux_mismatch.repaired_raw_qubo_energy
    @test aux_mismatch.repaired_bitstring == "11011"

    mktempdir() do dir
        summary_path = joinpath(dir, "summary.csv")
        svg_path = joinpath(dir, "distribution.svg")
        DSMFGQUBOExample.write_distribution_summary_csv(summary_path, rows)
        DSMFGQUBOExample.write_distribution_svg(svg_path, rows, instance, pool)

        summary = read(summary_path, String)
        @test occursin("algorithm,bitstring,reads", summary)
        @test occursin("projection_global_optimum", summary)

        svg = read(svg_path, String)
        @test occursin("<svg", svg)
        @test occursin("QAOA", svg)
        @test occursin("VQE", svg)
        @test occursin("Raw energy includes QUBO penalties", svg)
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

@testset "Aer backend configuration helpers" begin
    default_config = QiskitOpt.AerBackendConfig()
    @test QiskitOpt.aer_backend_options(default_config) == (method="matrix_product_state",)

    custom_config = QiskitOpt.AerBackendConfig(
        method="statevector",
        precision="single",
        max_parallel_threads=2,
        mps_omp_threads=1,
        mps_truncation_threshold=1.0e-6,
        mps_max_bond_dimension=32,
        mps_sample_measure_algorithm="mps_apply_measure",
        seed_simulator=1234,
        seed_transpiler=5678,
    )
    options = QiskitOpt.aer_backend_options(custom_config)
    @test options.method == "statevector"
    @test options.precision == "single"
    @test options.max_parallel_threads == 2
    @test options.mps_omp_threads == 1
    @test options.matrix_product_state_truncation_threshold == 1.0e-6
    @test options.matrix_product_state_max_bond_dimension == 32
    @test !(:mps_truncation_threshold in keys(options))
    @test !(:mps_max_bond_dimension in keys(options))
    @test options.mps_sample_measure_algorithm == "mps_apply_measure"
    @test options.seed_simulator == 1234
    @test !(:seed_transpiler in keys(options))

    smoke_config = QiskitOpt.AerBackendConfig(
        method="matrix_product_state",
        precision="single",
        max_parallel_threads=2,
        mps_omp_threads=1,
        mps_truncation_threshold=1.0e-6,
        mps_max_bond_dimension=32,
        mps_sample_measure_algorithm="mps_apply_measure",
        seed_simulator=1234,
        seed_transpiler=5678,
    )
    @test !isnothing(QiskitOpt.local_aer_backend(smoke_config))
    @test QiskitOpt.backend_version(QiskitOpt.default_local_backend()) ==
          QiskitOpt._python_module_version(QiskitOpt.qiskit_aer())
    @test QiskitOpt.backend_version(QiskitOpt.default_local_backend()) != "2"

    metadata = QiskitOpt.empty_metadata(
        "QAOA",
        "aer_simulator",
        "local";
        backend_version="0.17.0",
        backend_config=custom_config,
        backend_config_source="aer_attributes",
        number_of_reads=48,
        optimizer_iterations=2,
        optimizer_evaluations=3,
        optimizer_number_of_reads=16,
        seed_transpiler=5678,
    )
    metadata["time"]["effective"] = 1.0e-6
    metadata["time"]["total"] = 2.0e-6
    @test QUBODrivers.validate_metadata(metadata) == String[]
    @test metadata["algorithm"]["name"] == "QAOA"
    @test metadata["backend_name"] == "aer_simulator"
    @test metadata["backend"]["name"] == "aer_simulator"
    @test metadata["backend"]["version"] == "0.17.0"
    @test metadata["execution"]["mode"] == "local"
    @test metadata["status"] == "locally_solved"
    @test metadata["reads"]["number_of_reads"] == 48
    @test metadata["reads"]["final_number_of_reads"] == 48
    @test metadata["optimizer"]["iterations"] == 2
    @test metadata["optimizer"]["evaluations"] == 3
    @test metadata["optimizer"]["number_of_reads"] == 16
    @test metadata["backend_configuration"]["source"] == "aer_attributes"
    @test metadata["backend_configuration"]["method"] == "statevector"
    @test metadata["backend_configuration"]["precision"] == "single"
    @test metadata["seeds"]["simulator"] == 1234
    @test metadata["seeds"]["transpiler"] == 5678
    @test metadata["seeds"]["optimizer"] === nothing

    @test QiskitOpt.effective_seed(nothing, 73001, 1) == QiskitOpt.derived_seed(73001, 1)
    @test QiskitOpt.effective_seed(999, 73001, 1) == 999

    opaque_metadata = QiskitOpt.empty_metadata(
        "QAOA",
        "custom_backend",
        "local";
        backend_config_source="user_backend_factory",
        seed_transpiler=5678,
    )
    @test opaque_metadata["backend_configuration"] == Dict{String,Any}(
        "source" => "user_backend_factory",
    )
    @test !haskey(opaque_metadata["backend_configuration"], "method")
    @test opaque_metadata["seeds"]["simulator"] === nothing
    @test opaque_metadata["seeds"]["transpiler"] == 5678

    cloud_metadata = QiskitOpt.empty_metadata("QAOA", "ibm_backend", "cloud"; number_of_reads=48)
    cloud_metadata["time"]["effective"] = 1.0e-6
    cloud_metadata["time"]["total"] = 2.0e-6
    @test QUBODrivers.validate_metadata(cloud_metadata) == String[]
    @test cloud_metadata["status"] == "cloud_solved"

    factory = QiskitOpt.local_aer_backend_factory(custom_config)
    @test factory.config.seed_simulator == 1234
    @test QiskitOpt.local_aer_backend_factory().config.method == "matrix_product_state"

    sentinel_backend = Ref(:backend)
    selected = QiskitOpt.configured_local_backend(() -> sentinel_backend[], default_config)
    @test selected.backend === :backend
    @test selected.config === nothing
    @test selected.source == "user_backend_factory"
end

@testset "Initial parameter helpers" begin
    @test QAOA.parameter_names(3; number_of_layers=2) == ["β[0]", "β[1]", "γ[0]", "γ[1]"]
    @test QAOA.parameter_count(3; number_of_layers=2) == 4
    @test QAOA.fixed_angle_initial_parameters(number_of_layers=2) == [0.555, 0.293, -0.488, -0.898]
    @test QAOA.fixed_angle_guarantee(number_of_layers=2) == 0.7559
    @test QAOA.fixed_angle_initial_parameters(number_of_layers=1; gamma_sign=1) == [0.393, 0.616]
    @test_throws ArgumentError QAOA.fixed_angle_initial_parameters(number_of_layers=6)
    @test_throws ArgumentError QAOA.fixed_angle_guarantee(number_of_layers=6)

    @test QAOA.linear_ramp_initial_parameters(number_of_layers=4) ≈ [
        0.35,
        0.2625,
        0.175,
        0.0875,
        -0.1875,
        -0.375,
        -0.5625,
        -0.75,
    ]
    @test QAOA.linear_ramp_initial_parameters(
        number_of_layers=3;
        delta_beta=0.6,
        delta_gamma=1.2,
        gamma_sign=1,
    ) ≈ [0.6, 0.4, 0.2, 0.4, 0.8, 1.2]

    @test QAOA.tqa_initial_parameters(number_of_layers=3) ≈ [0.5, 0.25, 0.0, -0.25, -0.5, -0.75]
    @test QAOA.tqa_initial_parameters(number_of_layers=4; delta_t=0.8, gamma_sign=1) ≈ [
        0.6,
        0.4,
        0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
    ]

    prior_qaoa_parameters = [0.2, 0.4, -0.6, -1.0]
    @test QAOA.interpolated_initial_parameters(prior_qaoa_parameters) ≈ [0.2, 0.3, 0.4, -0.6, -0.8, -1.0]
    @test QAOA.interpolated_initial_parameters(prior_qaoa_parameters; gamma_sign=-1) ≈ [0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

    @test_throws ArgumentError QAOA.linear_ramp_initial_parameters(number_of_layers=0)
    @test_throws ArgumentError QAOA.tqa_initial_parameters(number_of_layers=0)
    @test_throws ArgumentError QAOA.interpolated_initial_parameters(Float64[])
    @test_throws ArgumentError QAOA.interpolated_initial_parameters([1.0, 2.0, 3.0])
    @test_throws ArgumentError QAOA.interpolated_initial_parameters(Any[1.0, "not-real"])

    qaoa_cost_operator = QiskitOpt.qiskit().quantum_info.SparsePauliOp.from_list([
        ("ZI", 1.0),
        ("IZ", 1.0),
        ("ZZ", 1.0),
    ])
    for number_of_layers in 1:3
        live_names = QiskitOpt.qiskit_parameter_names(
            QiskitOpt.qiskit().circuit.library.QAOAAnsatz(
                qaoa_cost_operator,
                reps=number_of_layers,
            ),
        )
        @test QAOA.parameter_names(2; number_of_layers=number_of_layers) == live_names
        @test QAOA.parameter_names(qaoa_cost_operator; number_of_layers=number_of_layers) == live_names
    end

    k4_edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)]
    k4_maxcut_operator = QiskitOpt.qiskit().quantum_info.SparsePauliOp.from_list([
        ("IIZZ", 0.5),
        ("IZIZ", 0.5),
        ("ZIIZ", 0.5),
        ("IZZI", 0.5),
        ("ZIZI", 0.5),
        ("ZZII", 0.5),
    ])
    for number_of_layers in 1:3
        negative_gamma_parameters = QAOA.fixed_angle_initial_parameters(
            number_of_layers=number_of_layers,
            gamma_sign=-1,
        )
        positive_gamma_parameters = QAOA.fixed_angle_initial_parameters(
            number_of_layers=number_of_layers,
            gamma_sign=1,
        )
        negative_gamma_state = QiskitOpt.qiskit().quantum_info.Statevector.from_instruction(
            QiskitOpt.qiskit().circuit.library.QAOAAnsatz(
                k4_maxcut_operator,
                reps=number_of_layers,
            ).assign_parameters(negative_gamma_parameters),
        )
        positive_gamma_state = QiskitOpt.qiskit().quantum_info.Statevector.from_instruction(
            QiskitOpt.qiskit().circuit.library.QAOAAnsatz(
                k4_maxcut_operator,
                reps=number_of_layers,
            ).assign_parameters(positive_gamma_parameters),
        )
        negative_gamma_ratio = expected_maxcut_value(negative_gamma_state, k4_edges) / 4
        positive_gamma_ratio = expected_maxcut_value(positive_gamma_state, k4_edges) / 4
        @test negative_gamma_ratio >= QAOA.fixed_angle_guarantee(number_of_layers=number_of_layers)
        @test negative_gamma_ratio > positive_gamma_ratio
    end

    qaoa_seeded = QAOA.random_initial_parameters(number_of_layers=2; seed=1234)
    @test length(qaoa_seeded) == 4
    @test qaoa_seeded == QAOA.random_initial_parameters(number_of_layers=2; seed=1234)
    @test qaoa_seeded != QAOA.random_initial_parameters(number_of_layers=2; seed=1235)

    vqe_names = VQE.parameter_names(n_variables=3)
    @test length(vqe_names) == VQE.parameter_count(n_variables=3)
    @test VQE.parameter_count(n_variables=3) == 24
    @test first(vqe_names) == "θ[0]"
    @test last(vqe_names) == "θ[23]"

    vqe_seeded = VQE.random_initial_parameters(n_variables=3; seed=73001)
    @test length(vqe_seeded) == 24
    @test vqe_seeded == VQE.random_initial_parameters(n_variables=3; seed=73001)
    @test vqe_seeded != VQE.random_initial_parameters(n_variables=3; seed=73002)
end

@testset "Fixed-parameter QAOA circuit export" begin
    model, _ = build_model(
        QAOA.Optimizer;
        Q=[
            -1.0 2.0
            2.0 -1.0
        ],
        L=[0.5, -0.25],
        scale=2.0,
        offset=1.5,
    )
    qaoa_sampler = QAOA._fixed_parameter_sampler(model)
    n, linear, quadratic, qubo_scale, qubo_offset = QUBOTools.qubo(qaoa_sampler, :dense)
    @test n == 2

    parameters = [0.23, -0.41]
    circuit, metadata = QAOA.fixed_parameter_circuit(
        model;
        parameters=parameters,
        reps=1,
        measure=true,
    )

    @test metadata["algorithm"]["mode"] == "fixed_parameter_circuit"
    @test metadata["qaoa"]["number_of_layers"] == 1
    @test metadata["parameters"]["input_order"] == "beta_then_gamma"
    @test metadata["parameters"]["qiskit_order"] == "beta_then_gamma"
    @test metadata["parameters"]["parameter_names"] == ["β[0]", "γ[0]"]
    @test metadata["parameters"]["values_order"] == "qiskit_order"
    @test metadata["parameters"]["values_aligned_to"] == "parameter_names"
    @test metadata["parameters"]["values"] == parameters
    @test metadata["variables"]["order"] == ["1", "2"]
    @test metadata["measurement"]["enabled"]
    @test metadata["objective"]["sense"] == "min"
    @test metadata["objective"]["scale"] == qubo_scale
    @test metadata["objective"]["offset"] == qubo_offset
    @test metadata["objective"]["qiskit_minimization_sign"] == 1
    @test metadata["circuit"]["num_qubits"] == 2
    @test metadata["circuit"]["num_clbits"] == 2
    @test metadata["circuit"]["operations"]["measure"] == 2

    previous_builder = QiskitOpt._runtime_service_builder[]
    QiskitOpt._runtime_service_builder[] = (; kwargs...) -> error("runtime service invoked")
    try
        withenv(
            "QISKIT_IBM_TOKEN" => "secret-token-issue-45",
            "QISKIT_IBM_INSTANCE" => "secret-instance-issue-45",
            "QISKIT_IBM_CHANNEL" => "ibm_quantum_platform",
        ) do
            unsafe_metadata = merge(
                metadata,
                Dict{String,Any}(
                    "token" => "secret-token-issue-45",
                    "instance" => "secret-instance-issue-45",
                    "account_file" => "/tmp/secret-account.json",
                ),
            )

            handoff = QAOA.ibm_runtime_handoff(
                circuit;
                fixed_metadata=unsafe_metadata,
                backend="ibm_fez",
                shots=4096,
                transpiler_seed=73001,
            )

            @test handoff.job === nothing
            @test handoff.transpiled_circuit === nothing
            @test handoff.metadata["runtime_handoff"]["mode"] == "dry_run"
            @test handoff.metadata["runtime_handoff"]["status"] == "dry_run"
            @test handoff.metadata["runtime_handoff"]["backend"] == "ibm_fez"
            @test handoff.metadata["runtime_handoff"]["shots"] == 4096
            @test handoff.metadata["runtime_handoff"]["transpiler_seed"] == 73001
            @test handoff.metadata["runtime_handoff"]["instance_configured"] == true
            @test handoff.metadata["runtime_handoff"]["credentials_recorded"] == false
            @test handoff.metadata["fixed_parameter_circuit"]["parameters"]["parameter_names"] == ["β[0]", "γ[0]"]
            @test handoff.metadata["fixed_parameter_circuit"]["objective"]["value_convention"] ==
                  "QUBOTools.value(bits, linear, quadratic, scale, offset)"
            @test handoff.metadata["count_scoring"]["bit_conversion"] == "QAOA.count_key_bits"
            @test haskey(handoff.metadata["packages"], "qiskit")
            @test !haskey(handoff.metadata["fixed_parameter_circuit"], "token")
            @test !haskey(handoff.metadata["fixed_parameter_circuit"], "instance")
            @test !haskey(handoff.metadata["fixed_parameter_circuit"], "account_file")

            handoff_text = repr(handoff.metadata)
            @test !occursin("secret-token-issue-45", handoff_text)
            @test !occursin("secret-instance-issue-45", handoff_text)
            @test !occursin("/tmp/secret-account.json", handoff_text)

            saved_account_token = "saved-account-token-issue-45"
            secret_crn = "secret-crn-issue-45"
            QiskitOpt._runtime_service_builder[] = (; kwargs...) -> error(
                "auth failed token=$(saved_account_token) " *
                "instance=secret-instance-issue-45 " *
                "account_file=/tmp/secret-account.json " *
                "crn:$(secret_crn)",
            )

            runtime_error = try
                QAOA.ibm_runtime_handoff(
                    circuit;
                    fixed_metadata=unsafe_metadata,
                    backend="ibm_fez",
                    shots=4096,
                    transpiler_seed=73001,
                    dry_run=false,
                )
                nothing
            catch err
                err
            end

            @test runtime_error isa QAOA.RuntimeHandoffError
            @test runtime_error.cause !== nothing
            @test runtime_error.metadata["runtime_handoff"]["status"] == "failed_before_submission"
            failure_message = runtime_error.metadata["runtime_handoff"]["failure"]["message"]
            @test occursin("[redacted]", failure_message)
            for secret in (
                saved_account_token,
                "secret-instance-issue-45",
                "/tmp/secret-account.json",
                secret_crn,
            )
                @test !occursin(secret, failure_message)
                @test !occursin(secret, repr(runtime_error.metadata))
                @test !occursin(secret, sprint(showerror, runtime_error))
            end
        end
    finally
        QiskitOpt._runtime_service_builder[] = previous_builder
    end

    @test_throws ArgumentError QAOA.ibm_runtime_handoff(circuit; backend="", shots=1024)
    @test_throws ArgumentError QAOA.ibm_runtime_handoff(circuit; backend="ibm_fez", shots=0)
    @test_throws ArgumentError QAOA.ibm_runtime_handoff(
        circuit.remove_final_measurements(inplace=false);
        backend="ibm_fez",
        shots=1024,
    )

    max_model, _ = build_model(
        QAOA.Optimizer;
        Q=[
            -1.0 2.0
            2.0 -1.0
        ],
        sense=:Max,
    )
    _, max_metadata = QAOA.fixed_parameter_circuit(
        max_model;
        parameters=parameters,
        reps=1,
        measure=false,
    )
    @test max_metadata["objective"]["sense"] == "max"
    @test max_metadata["objective"]["qiskit_minimization_sign"] == -1

    unmeasured = circuit.remove_final_measurements(inplace=false)
    exported_probabilities = QiskitOpt.PythonCall.pyconvert(
        Dict{String,Float64},
        QiskitOpt.qiskit().quantum_info.Statevector.from_instruction(unmeasured).probabilities_dict(),
    )

    reference = QiskitOpt.qiskit().circuit.library.QAOAAnsatz(
        QiskitOpt.quadratic_program(qaoa_sampler)[0],
        reps=1,
    ).assign_parameters(QiskitOpt.numpy().array(parameters))
    reference_probabilities = QiskitOpt.PythonCall.pyconvert(
        Dict{String,Float64},
        QiskitOpt.qiskit().quantum_info.Statevector.from_instruction(reference).probabilities_dict(),
    )

    @test Set(keys(exported_probabilities)) == Set(keys(reference_probabilities))
    for key in keys(reference_probabilities)
        @test exported_probabilities[key] ≈ reference_probabilities[key]
    end

    uniform_circuit, uniform_metadata = QAOA.fixed_parameter_circuit(
        model;
        parameters=[0.0, 0.0],
        reps=1,
        measure=false,
    )
    uniform_probabilities = QiskitOpt.PythonCall.pyconvert(
        Dict{String,Float64},
        QiskitOpt.qiskit().quantum_info.Statevector.from_instruction(uniform_circuit).probabilities_dict(),
    )
    @test uniform_metadata["measurement"]["enabled"] == false
    @test uniform_metadata["circuit"]["num_clbits"] == 0
    @test Set(keys(uniform_probabilities)) == Set(["00", "01", "10", "11"])
    for probability in values(uniform_probabilities)
        @test probability ≈ 0.25
    end

    reordered_circuit, reordered_metadata = QAOA.fixed_parameter_circuit(
        model;
        parameters=[-0.41, 0.23],
        reps=1,
        parameter_order=:gamma_then_beta,
        measure=false,
    )
    reordered_probabilities = QiskitOpt.PythonCall.pyconvert(
        Dict{String,Float64},
        QiskitOpt.qiskit().quantum_info.Statevector.from_instruction(reordered_circuit).probabilities_dict(),
    )
    @test reordered_metadata["parameters"]["input_order"] == "gamma_then_beta"
    @test reordered_metadata["parameters"]["values_order"] == "qiskit_order"
    @test reordered_metadata["parameters"]["values_aligned_to"] == "parameter_names"
    @test reordered_metadata["parameters"]["values"] == parameters
    @test Set(keys(reordered_probabilities)) == Set(keys(exported_probabilities))
    for key in keys(exported_probabilities)
        @test reordered_probabilities[key] ≈ exported_probabilities[key]
    end

    @test QAOA.count_key_bits("10") == [0, 1]
    @test QAOA.count_key_bitstring("10") == "01"
    @test QUBOTools.value(QAOA.count_key_bits("10"), linear, quadratic, qubo_scale, qubo_offset) ≈
          QUBOTools.value([0, 1], linear, quadratic, qubo_scale, qubo_offset)

    MOI.set(model, QAOA.NumberOfLayers(), 1)
    _, default_layer_metadata = QAOA.fixed_parameter_circuit(
        model;
        parameters=parameters,
        measure=false,
    )
    @test default_layer_metadata["qaoa"]["number_of_layers"] == 1

    @test_throws ArgumentError QAOA.fixed_parameter_circuit(model; parameters=[0.0], reps=1)
    @test_throws ArgumentError QAOA.fixed_parameter_circuit(
        model;
        parameters=parameters,
        reps=1,
        number_of_layers=2,
    )
    @test_throws ArgumentError QAOA.fixed_parameter_circuit(
        model;
        parameters=parameters,
        reps=1,
        parameter_order=:interleaved,
    )
end

@testset "Initial parameter validation fails before backend execution" begin
    previous_builder = QiskitOpt._runtime_service_builder[]
    QiskitOpt._runtime_service_builder[] = (; kwargs...) -> error("runtime service invoked")

    try
        qaoa_model, _ = build_model(QAOA.Optimizer)
        MOI.set(qaoa_model, QAOA.IBMBackend(), "ibm_fez")
        MOI.set(qaoa_model, QAOA.InitialParameters(), [0.0])
        @test_throws ArgumentError MOI.optimize!(qaoa_model)

        vqe_model, _ = build_model(VQE.Optimizer)
        MOI.set(vqe_model, VQE.IBMBackend(), "ibm_fez")
        MOI.set(vqe_model, VQE.InitialParameters(), [0.0])
        @test_throws ArgumentError MOI.optimize!(vqe_model)
    finally
        QiskitOpt._runtime_service_builder[] = previous_builder
    end
end

@testset "Aer backend configuration is recorded in SampleSet metadata" begin
    for (optimizer_factory, optimizer_module, algorithm) in (
        (QAOA.Optimizer, QAOA, "QAOA"),
        (VQE.Optimizer, VQE, "VQE"),
    )
        initial_parameters = if optimizer_module === QAOA
            QAOA.fixed_angle_initial_parameters(number_of_layers=1)
        else
            VQE.random_initial_parameters(n_variables=3; seed=73001)
        end
        initial_parameter_names = if optimizer_module === QAOA
            QAOA.parameter_names(3; number_of_layers=1)
        else
            VQE.parameter_names(n_variables=3)
        end
        initial_parameter_source = optimizer_module === QAOA ? QAOA.FIXED_ANGLE_SOURCE : "random_seed_73001"
        number_of_reads = 16

        sampleset = sample_locally_without_runtime_service(
            optimizer_factory,
            optimizer_module,
            (model, module_) -> begin
                MOI.set(model, module_.AerPrecision(), "single")
                MOI.set(model, module_.AerMaxParallelThreads(), 1)
                MOI.set(model, module_.AerSeedSimulator(), 1234)
                MOI.set(model, module_.TranspilerSeed(), 5678)
                MOI.set(model, module_.InitialParameters(), initial_parameters)
                MOI.set(model, module_.InitialParameterSource(), initial_parameter_source)
            end;
            max_iterations=1,
            number_of_reads=number_of_reads,
        )
        metadata = QUBOTools.metadata(sampleset)
        @test startswith(metadata["origin"], "$(algorithm) @ ")
        @test metadata["execution_mode"] == "local"
        @test metadata["execution"]["mode"] == "local"
        @test metadata["algorithm"]["name"] == algorithm
        @test metadata["backend_name"] == metadata["backend"]["name"]
        @test metadata["backend"] isa Dict{String,Any}
        @test metadata["backend"]["name"] isa String
        @test haskey(metadata["backend"], "version")
        @test metadata["reads"]["number_of_reads"] == number_of_reads
        @test metadata["reads"]["final_number_of_reads"] == number_of_reads
        @test metadata["optimizer"]["number_of_reads"] == number_of_reads
        @test metadata["optimizer"]["evaluations"] isa Union{Integer,Nothing}
        @test metadata["time"]["effective"] > 0
        @test metadata["time"]["total"] > 0
        @test QUBODrivers.validate_metadata(metadata) == String[]
        @test metadata["backend_configuration"]["source"] == "aer_attributes"
        @test metadata["backend_configuration"]["method"] == "matrix_product_state"
        @test metadata["backend_configuration"]["precision"] == "single"
        @test metadata["backend_configuration"]["max_parallel_threads"] == 1
        @test metadata["seeds"]["simulator"] == 1234
        @test metadata["seeds"]["transpiler"] == 5678
        @test metadata["initial_parameters"]["source"] == initial_parameter_source
        @test metadata["initial_parameters"]["parameter_names"] == initial_parameter_names
        @test metadata["initial_parameters"]["values"] == initial_parameters
        if optimizer_module === QAOA
            @test metadata["pass_manager"]["source"] == "default_preset"
            @test metadata["pass_manager"]["optimization_level"] == 3
        else
            @test !haskey(metadata, "pass_manager")
        end
    end
end

@testset "Standard QUBODrivers RandomSeed is recorded and overridden explicitly" begin
    for (optimizer_factory, optimizer_module) in ((QAOA.Optimizer, QAOA), (VQE.Optimizer, VQE))
        sampler = optimizer_factory()
        @test MOI.supports(sampler, QUBODrivers.RandomSeed())
        @test QUBODrivers.supports_seed(typeof(sampler))
        MOI.set(sampler, QUBODrivers.RandomSeed(), 73001)
        @test MOI.get(sampler, QUBODrivers.RandomSeed()) == 73001

        sampleset = sample_locally_without_runtime_service(
            optimizer_factory,
            optimizer_module,
            (model, module_) -> begin
                MOI.set(model, QUBODrivers.RandomSeed(), 73001)
                MOI.set(model, module_.AerSeedSimulator(), 1234)
                MOI.set(model, module_.TranspilerSeed(), 5678)
            end;
            max_iterations=1,
            number_of_reads=8,
        )
        metadata = QUBOTools.metadata(sampleset)

        @test metadata["seeds"]["sampler"] == 73001
        @test metadata["seeds"]["simulator"] == 1234
        @test metadata["seeds"]["transpiler"] == 5678
    end
end

@testset "QAOA custom pass-manager factory" begin
    factory_calls = NamedTuple[]
    pass_manager_runs = Ref(0)

    function factory(backend; optimization_level=3, seed_transpiler=nothing)
        push!(
            factory_calls,
            (
                backend=backend,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
            ),
        )
        inner = QiskitOpt.preset_pass_manager(
            backend;
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        return (
            run = circuit -> begin
                pass_manager_runs[] += 1
                return inner.run(circuit)
            end,
        )
    end

    sampleset = sample_locally_without_runtime_service(
        QAOA.Optimizer,
        QAOA,
        (model, module_) -> begin
            MOI.set(model, module_.PassManagerFactory(), factory)
            MOI.set(model, module_.TranspilerSeed(), 4321)
        end;
        max_iterations=1,
        number_of_reads=16,
    )
    metadata = QUBOTools.metadata(sampleset)

    @test length(factory_calls) == 1
    @test factory_calls[1].optimization_level == 3
    @test factory_calls[1].seed_transpiler == 4321
    @test pass_manager_runs[] == 2
    @test metadata["pass_manager"]["source"] == "custom_factory"
    @test metadata["pass_manager"]["optimization_level"] == 3
    @test !isdefined(VQE, :PassManagerFactory)
end

@testset "Final sampling reads use generic QUBODrivers attribute" begin
    for (optimizer_factory, optimizer_module) in ((QAOA.Optimizer, QAOA), (VQE.Optimizer, VQE))
        fallback_number_of_reads = 32
        fallback_sampleset = sample_locally_without_runtime_service(
            optimizer_factory,
            optimizer_module;
            max_iterations=1,
            number_of_reads=fallback_number_of_reads,
        )
        @test sum(QUBOTools.reads(sample) for sample in fallback_sampleset) == fallback_number_of_reads

        final_number_of_reads = 48
        sampleset = sample_locally_without_runtime_service(
            optimizer_factory,
            optimizer_module,
            (model, _) -> MOI.set(model, QUBODrivers.FinalNumberOfReads(), final_number_of_reads);
            max_iterations=1,
            number_of_reads=8,
        )

        @test sum(QUBOTools.reads(sample) for sample in sampleset) == final_number_of_reads
    end
end

@testset "Initial parameter metadata can be disabled" begin
    sampleset = sample_locally_without_runtime_service(
        QAOA.Optimizer,
        QAOA,
        (model, module_) -> begin
            MOI.set(model, module_.InitialParameters(), QAOA.fixed_angle_initial_parameters(number_of_layers=1))
            MOI.set(model, module_.RecordInitialParameters(), false)
        end;
        max_iterations=1,
        number_of_reads=8,
    )
    @test !haskey(QUBOTools.metadata(sampleset), "initial_parameters")
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
