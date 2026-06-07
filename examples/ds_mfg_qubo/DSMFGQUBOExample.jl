module DSMFGQUBOExample

using Printf
using QiskitOpt
using QiskitOpt: QAOA, QUBODrivers, QUBOTools, VQE

const MOI = QUBODrivers.MOI

export AnnotatedObservation,
    DistributionObservation,
    PoolEntry,
    QUBOInstance,
    VariableInfo,
    annotate_distributions,
    bitstring_from_bits,
    bits_from_string,
    build_model,
    load_cached_distributions,
    load_instance,
    load_solution_pool,
    projected_objective,
    qubo_energy,
    repair_bits,
    run_example,
    solve_with_qiskitopt,
    write_distribution_observations,
    write_distribution_summary_csv,
    write_distribution_svg

struct VariableInfo
    index::Int
    label::String
    role::Symbol
    coefficient::Float64
    projected_coefficient::Float64
    repair_source::Union{Nothing,Int}
end

struct QUBOInstance
    name::String
    n_variables::Int
    scale::Float64
    offset::Float64
    variables::Vector{VariableInfo}
    linear::Vector{Float64}
    quadratic::Matrix{Float64}
    flow_indices::Vector{Int}
    auxiliary_indices::Vector{Int}
    repair_sources::Dict{Int,Int}
end

struct PoolEntry
    rank::Int
    flow_bits::String
    aux_bits::String
    full_bits::String
    projected_objective::Float64
    raw_qubo_energy::Float64
    match::String
end

struct DistributionObservation
    algorithm::String
    bitstring::String
    reads::Int
    source::String
end

struct AnnotatedObservation
    algorithm::String
    bitstring::String
    reads::Int
    probability::Float64
    raw_qubo_energy::Float64
    projected_objective::Float64
    repaired_raw_qubo_energy::Float64
    match::String
    flow_bits::String
    aux_bits::String
    repaired_bitstring::String
end

function _data_dir(base_dir::AbstractString)
    return joinpath(base_dir, "data")
end

function _read_simple_csv(path::AbstractString)
    rows = String[]
    for line in readlines(path)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue
        push!(rows, stripped)
    end
    isempty(rows) && error("CSV file is empty: $(path)")

    header = strip.(split(first(rows), ","; keepempty=true))
    records = Vector{Dict{String,String}}()
    for (line_number, line) in enumerate(rows[2:end])
        fields = strip.(split(line, ","; keepempty=true))
        if length(fields) != length(header)
            error("CSV field count mismatch in $(path) at data row $(line_number)")
        end
        push!(records, Dict{String,String}(header .=> fields))
    end
    return records
end

function _required(row::Dict{String,String}, key::AbstractString)
    haskey(row, key) || error("missing CSV field: $(key)")
    return row[key]
end

function _parse_int(row::Dict{String,String}, key::AbstractString)
    return parse(Int, _required(row, key))
end

function _parse_float(row::Dict{String,String}, key::AbstractString)
    return parse(Float64, _required(row, key))
end

function _parse_optional_int(row::Dict{String,String}, key::AbstractString)
    value = _required(row, key)
    return isempty(value) ? nothing : parse(Int, value)
end

function _algorithm_name(algorithm)
    name = uppercase(String(algorithm))
    name in ("QAOA", "VQE") || throw(ArgumentError("unsupported algorithm: $(algorithm)"))
    return name
end

function load_instance(base_dir::AbstractString = @__DIR__)
    data_dir = _data_dir(base_dir)
    scalar_rows = _read_simple_csv(joinpath(data_dir, "scalars.csv"))
    length(scalar_rows) == 1 || error("scalars.csv must contain exactly one data row")

    scalars = only(scalar_rows)
    n_variables = _parse_int(scalars, "n")
    scale = _parse_float(scalars, "scale")
    offset = _parse_float(scalars, "offset")

    variable_rows = _read_simple_csv(joinpath(data_dir, "linear.csv"))
    variables = VariableInfo[]
    for row in variable_rows
        role = Symbol(_required(row, "role"))
        role in (:flow, :auxiliary) || error("unsupported variable role: $(role)")
        push!(
            variables,
            VariableInfo(
                _parse_int(row, "index"),
                _required(row, "label"),
                role,
                _parse_float(row, "coefficient"),
                _parse_float(row, "projected_coefficient"),
                _parse_optional_int(row, "repair_source"),
            ),
        )
    end
    sort!(variables; by = variable -> variable.index)

    if [variable.index for variable in variables] != collect(1:n_variables)
        error("linear.csv variable indices must be contiguous from 1 to n")
    end

    linear = [variable.coefficient for variable in variables]
    quadratic = zeros(Float64, n_variables, n_variables)
    for row in _read_simple_csv(joinpath(data_dir, "quadratic.csv"))
        i = _parse_int(row, "row")
        j = _parse_int(row, "column")
        1 <= i <= n_variables || error("quadratic row index out of range: $(i)")
        1 <= j <= n_variables || error("quadratic column index out of range: $(j)")
        quadratic[i, j] += _parse_float(row, "coefficient")
    end

    flow_indices = [variable.index for variable in variables if variable.role == :flow]
    auxiliary_indices = [variable.index for variable in variables if variable.role == :auxiliary]
    repair_sources = Dict{Int,Int}()
    for variable in variables
        if !isnothing(variable.repair_source)
            source = variable.repair_source::Int
            source in flow_indices || error("repair source must be a flow variable index")
            repair_sources[variable.index] = source
        end
    end

    return QUBOInstance(
        "compact_ds_mfg_flow_example",
        n_variables,
        scale,
        offset,
        variables,
        linear,
        quadratic,
        flow_indices,
        auxiliary_indices,
        repair_sources,
    )
end

function bits_from_string(bitstring::AbstractString)
    bits = Int[]
    for digit in bitstring
        digit == '0' && push!(bits, 0)
        digit == '1' && push!(bits, 1)
        digit in ('0', '1') || throw(ArgumentError("bitstring must contain only 0 and 1"))
    end
    return bits
end

function bitstring_from_bits(bits::AbstractVector{<:Integer})
    return join(Int.(bits))
end

function _check_bits(instance::QUBOInstance, bits::AbstractVector{<:Integer})
    length(bits) == instance.n_variables ||
        throw(ArgumentError("expected $(instance.n_variables) bits, got $(length(bits))"))
    all(bit -> bit == 0 || bit == 1, bits) || throw(ArgumentError("bits must be binary"))
    return Int.(bits)
end

function qubo_energy(instance::QUBOInstance, bits::AbstractVector{<:Integer})
    checked_bits = _check_bits(instance, bits)
    raw = instance.offset + sum(instance.linear .* checked_bits)
    for i in 1:instance.n_variables, j in 1:instance.n_variables
        raw += instance.quadratic[i, j] * checked_bits[i] * checked_bits[j]
    end
    return instance.scale * raw
end

function projected_objective(instance::QUBOInstance, bits::AbstractVector{<:Integer})
    checked_bits = _check_bits(instance, bits)
    return sum(
        instance.variables[i].projected_coefficient * checked_bits[i] for i in instance.flow_indices
    )
end

function repair_bits(instance::QUBOInstance, bits::AbstractVector{<:Integer})
    repaired = _check_bits(instance, bits)
    for (aux_index, source_index) in instance.repair_sources
        repaired[aux_index] = repaired[source_index]
    end
    return repaired
end

function _flow_bits(instance::QUBOInstance, bits::AbstractVector{<:Integer})
    checked_bits = _check_bits(instance, bits)
    return bitstring_from_bits(checked_bits[instance.flow_indices])
end

function _aux_bits(instance::QUBOInstance, bits::AbstractVector{<:Integer})
    checked_bits = _check_bits(instance, bits)
    return bitstring_from_bits(checked_bits[instance.auxiliary_indices])
end

function build_model(instance::QUBOInstance, optimizer_factory)
    model = MOI.instantiate(optimizer_factory; with_cache_type = Float64)
    variables = MOI.add_variables(model, instance.n_variables)
    for variable in variables
        MOI.add_constraint(model, variable, MOI.ZeroOne())
    end

    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:instance.n_variables, j in 1:instance.n_variables
        coefficient = instance.quadratic[i, j]
        iszero(coefficient) && continue
        moi_coefficient = (i == j ? 2.0 : 1.0) * instance.scale * coefficient
        push!(quadratic_terms, MOI.ScalarQuadraticTerm(moi_coefficient, variables[i], variables[j]))
    end

    linear_terms = MOI.ScalarAffineTerm{Float64}[]
    for i in 1:instance.n_variables
        coefficient = instance.linear[i]
        iszero(coefficient) && continue
        push!(linear_terms, MOI.ScalarAffineTerm(instance.scale * coefficient, variables[i]))
    end

    objective = MOI.ScalarQuadraticFunction(
        quadratic_terms,
        linear_terms,
        instance.scale * instance.offset,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)

    return model, variables
end

function load_solution_pool(instance::QUBOInstance, base_dir::AbstractString = @__DIR__)
    pool = PoolEntry[]
    for row in _read_simple_csv(joinpath(_data_dir(base_dir), "solution_pool.csv"))
        full_bits = _required(row, "full_bits")
        bits = bits_from_string(full_bits)
        if !isapprox(qubo_energy(instance, bits), _parse_float(row, "raw_qubo_energy"); atol = 1.0e-8)
            error("solution_pool.csv raw QUBO energy is inconsistent for $(full_bits)")
        end
        push!(
            pool,
            PoolEntry(
                _parse_int(row, "rank"),
                _required(row, "flow_bits"),
                _required(row, "aux_bits"),
                full_bits,
                _parse_float(row, "projected_objective"),
                _parse_float(row, "raw_qubo_energy"),
                _required(row, "match"),
            ),
        )
    end
    sort!(pool; by = entry -> entry.rank)
    return pool
end

function load_cached_distributions(base_dir::AbstractString = @__DIR__)
    observations = DistributionObservation[]
    for row in _read_simple_csv(joinpath(_data_dir(base_dir), "cached_distributions.csv"))
        push!(
            observations,
            DistributionObservation(
                _algorithm_name(_required(row, "algorithm")),
                _required(row, "bitstring"),
                _parse_int(row, "reads"),
                _required(row, "source"),
            ),
        )
    end
    return observations
end

function _match_label(pool::Vector{PoolEntry}, full_bits::String, flow_bits::String)
    for entry in pool
        entry.full_bits == full_bits && return entry.match
    end
    for entry in pool
        entry.flow_bits == flow_bits && return "projection_" * entry.match
    end
    return "not_in_solution_pool"
end

function annotate_distributions(
    instance::QUBOInstance,
    observations::Vector{DistributionObservation},
    pool::Vector{PoolEntry},
)
    totals = Dict{String,Int}()
    for observation in observations
        totals[observation.algorithm] = get(totals, observation.algorithm, 0) + observation.reads
    end

    annotated = AnnotatedObservation[]
    for observation in observations
        bits = bits_from_string(observation.bitstring)
        repaired = repair_bits(instance, bits)
        flow_bits = _flow_bits(instance, bits)
        push!(
            annotated,
            AnnotatedObservation(
                observation.algorithm,
                observation.bitstring,
                observation.reads,
                observation.reads / totals[observation.algorithm],
                qubo_energy(instance, bits),
                projected_objective(instance, bits),
                qubo_energy(instance, repaired),
                _match_label(pool, observation.bitstring, flow_bits),
                flow_bits,
                _aux_bits(instance, bits),
                bitstring_from_bits(repaired),
            ),
        )
    end
    return annotated
end

function _configure_qaoa!(model, instance::QUBOInstance; number_of_reads::Integer, maximum_iterations::Integer)
    MOI.set(model, QAOA.NumberOfReads(), number_of_reads)
    MOI.set(model, QAOA.MaximumIterations(), maximum_iterations)
    MOI.set(model, QAOA.NumberOfLayers(), 1)
    MOI.set(model, QAOA.InitialParameters(), QAOA.fixed_angle_initial_parameters(number_of_layers = 1))
    MOI.set(model, QAOA.InitialParameterSource(), QAOA.FIXED_ANGLE_SOURCE)
    MOI.set(model, QAOA.AerPrecision(), "single")
    MOI.set(model, QAOA.AerMaxParallelThreads(), 1)
    MOI.set(model, QAOA.AerSeedSimulator(), 73001)
    MOI.set(model, QAOA.TranspilerSeed(), 73001)
    return model
end

function _configure_vqe!(model, instance::QUBOInstance; number_of_reads::Integer, maximum_iterations::Integer)
    MOI.set(model, VQE.NumberOfReads(), number_of_reads)
    MOI.set(model, VQE.MaximumIterations(), maximum_iterations)
    MOI.set(model, VQE.InitialParameters(), VQE.random_initial_parameters(n_variables = instance.n_variables, seed = 73001))
    MOI.set(model, VQE.InitialParameterSource(), "random_seed_73001")
    MOI.set(model, VQE.AerPrecision(), "single")
    MOI.set(model, VQE.AerMaxParallelThreads(), 1)
    MOI.set(model, VQE.AerSeedSimulator(), 73001)
    MOI.set(model, VQE.TranspilerSeed(), 73001)
    return model
end

function solve_with_qiskitopt(
    instance::QUBOInstance,
    algorithm;
    number_of_reads::Integer = 128,
    maximum_iterations::Integer = 10,
)
    name = _algorithm_name(algorithm)
    optimizer_factory = name == "QAOA" ? QAOA.Optimizer : VQE.Optimizer
    model, _ = build_model(instance, optimizer_factory)
    if name == "QAOA"
        _configure_qaoa!(model, instance; number_of_reads, maximum_iterations)
    else
        _configure_vqe!(model, instance; number_of_reads, maximum_iterations)
    end

    MOI.optimize!(model)
    sampleset = QUBOTools.solution(MOI.get(model, MOI.RawSolver()))
    observations = DistributionObservation[]
    for sample in sampleset
        bits = Int.(QUBOTools.state(sample))
        push!(
            observations,
            DistributionObservation(
                name,
                bitstring_from_bits(bits),
                QUBOTools.reads(sample),
                "local_aer",
            ),
        )
    end
    sort!(observations; by = row -> (-row.reads, row.bitstring))
    return observations
end

function write_distribution_observations(path::AbstractString, observations::Vector{DistributionObservation})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "algorithm,bitstring,reads,source")
        for row in observations
            println(io, join((row.algorithm, row.bitstring, row.reads, row.source), ","))
        end
    end
    return path
end

function _fmt(value::Real)
    return @sprintf("%.10g", value)
end

function write_distribution_summary_csv(path::AbstractString, rows::Vector{AnnotatedObservation})
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "algorithm,bitstring,reads,probability,raw_qubo_energy,projected_objective,repaired_raw_qubo_energy,match,flow_bits,aux_bits,repaired_bitstring",
        )
        for row in sort(rows; by = row -> (row.algorithm, -row.probability, row.bitstring))
            println(
                io,
                join(
                    (
                        row.algorithm,
                        row.bitstring,
                        row.reads,
                        _fmt(row.probability),
                        _fmt(row.raw_qubo_energy),
                        _fmt(row.projected_objective),
                        _fmt(row.repaired_raw_qubo_energy),
                        row.match,
                        row.flow_bits,
                        row.aux_bits,
                        row.repaired_bitstring,
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function _xml_escape(text)
    escaped = replace(String(text), "&" => "&amp;")
    escaped = replace(escaped, "<" => "&lt;")
    escaped = replace(escaped, ">" => "&gt;")
    escaped = replace(escaped, "\"" => "&quot;")
    return escaped
end

function _bar_color(algorithm::AbstractString)
    algorithm == "QAOA" && return "#2f6f9f"
    algorithm == "VQE" && return "#c45824"
    return "#555555"
end

function write_distribution_svg(
    path::AbstractString,
    rows::Vector{AnnotatedObservation},
    instance::QUBOInstance,
    pool::Vector{PoolEntry};
    title::AbstractString = "QAOA/VQE cached sample distributions",
)
    isempty(rows) && error("no distribution rows to plot")
    mkpath(dirname(path))

    ordered = sort(rows; by = row -> (row.algorithm, -row.probability, row.bitstring))
    width = 980
    height = 430
    left = 70
    right = 30
    top = 55
    bottom = 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_probability = maximum(row.probability for row in ordered)
    slot = plot_width / length(ordered)
    bar_width = max(16.0, min(42.0, slot * 0.62))
    optimum = first(pool)

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(width)" height="$(height)" viewBox="0 0 $(width) $(height)">""")
        println(io, """<rect width="100%" height="100%" fill="#ffffff"/>""")
        println(io, """<text x="$(left)" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#222222">$(_xml_escape(title))</text>""")
        println(io, """<text x="$(left)" y="50" font-family="Arial, sans-serif" font-size="12" fill="#555555">Instance: $(_xml_escape(instance.name)); best full bits $(optimum.full_bits), projected objective $(_fmt(optimum.projected_objective))</text>""")
        println(io, """<line x1="$(left)" y1="$(top + plot_height)" x2="$(width - right)" y2="$(top + plot_height)" stroke="#222222" stroke-width="1"/>""")
        println(io, """<line x1="$(left)" y1="$(top)" x2="$(left)" y2="$(top + plot_height)" stroke="#222222" stroke-width="1"/>""")
        println(io, """<text x="18" y="$(top + 12)" font-family="Arial, sans-serif" font-size="12" fill="#555555">prob.</text>""")

        for (i, row) in enumerate(ordered)
            x = left + (i - 0.5) * slot - bar_width / 2
            bar_height = row.probability / max_probability * (plot_height - 16)
            y = top + plot_height - bar_height
            color = _bar_color(row.algorithm)
            println(io, """<rect x="$(_fmt(x))" y="$(_fmt(y))" width="$(_fmt(bar_width))" height="$(_fmt(bar_height))" fill="$(color)" rx="2"/>""")
            println(io, """<text x="$(_fmt(x + bar_width / 2))" y="$(_fmt(y - 5))" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333333">$(_fmt(row.probability))</text>""")
            println(io, """<text x="$(_fmt(x + bar_width / 2))" y="$(top + plot_height + 18)" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#222222">$(_xml_escape(row.bitstring))</text>""")
            println(io, """<text x="$(_fmt(x + bar_width / 2))" y="$(top + plot_height + 34)" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#555555">$(_xml_escape(row.algorithm))</text>""")
            println(io, """<text x="$(_fmt(x + bar_width / 2))" y="$(top + plot_height + 50)" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#555555">raw $(_fmt(row.raw_qubo_energy))</text>""")
            println(io, """<text x="$(_fmt(x + bar_width / 2))" y="$(top + plot_height + 65)" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#555555">proj $(_fmt(row.projected_objective))</text>""")
        end

        legend_y = height - 22
        println(io, """<rect x="$(left)" y="$(legend_y - 11)" width="11" height="11" fill="$(_bar_color("QAOA"))"/>""")
        println(io, """<text x="$(left + 17)" y="$(legend_y - 2)" font-family="Arial, sans-serif" font-size="12" fill="#333333">QAOA</text>""")
        println(io, """<rect x="$(left + 75)" y="$(legend_y - 11)" width="11" height="11" fill="$(_bar_color("VQE"))"/>""")
        println(io, """<text x="$(left + 92)" y="$(legend_y - 2)" font-family="Arial, sans-serif" font-size="12" fill="#333333">VQE</text>""")
        println(io, """<text x="$(left + 150)" y="$(legend_y - 2)" font-family="Arial, sans-serif" font-size="12" fill="#555555">Raw energy includes QUBO penalties and auxiliary bits; projected objective uses flow bits.</text>""")
        println(io, "</svg>")
    end
    return path
end

function run_example(;
    base_dir::AbstractString = @__DIR__,
    output_dir::AbstractString = joinpath(base_dir, "output"),
    rerun::Bool = false,
    number_of_reads::Integer = 128,
    maximum_iterations::Integer = 10,
)
    instance = load_instance(base_dir)
    pool = load_solution_pool(instance, base_dir)
    observations = if rerun
        vcat(
            solve_with_qiskitopt(instance, "QAOA"; number_of_reads, maximum_iterations),
            solve_with_qiskitopt(instance, "VQE"; number_of_reads, maximum_iterations),
        )
    else
        load_cached_distributions(base_dir)
    end

    mkpath(output_dir)
    if rerun
        write_distribution_observations(joinpath(output_dir, "sample_distributions.csv"), observations)
    end
    rows = annotate_distributions(instance, observations, pool)
    summary_path = write_distribution_summary_csv(joinpath(output_dir, "distribution_summary.csv"), rows)
    svg_path = write_distribution_svg(joinpath(output_dir, "distribution_comparison.svg"), rows, instance, pool)
    return (instance = instance, pool = pool, observations = observations, rows = rows, summary_path = summary_path, svg_path = svg_path)
end

end
