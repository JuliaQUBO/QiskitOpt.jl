#!/usr/bin/env julia

include(joinpath(@__DIR__, "DSMFGQUBOExample.jl"))

function _env_flag(name::AbstractString)
    value = lowercase(get(ENV, name, "false"))
    return value in ("1", "true", "yes", "on")
end

function main(args = ARGS)
    rerun = _env_flag("QISKITOPT_DSMFG_RERUN") || "--rerun" in args
    output_dir = get(ENV, "QISKITOPT_DSMFG_OUTPUT_DIR", joinpath(@__DIR__, "output"))
    result = DSMFGQUBOExample.run_example(; base_dir = @__DIR__, output_dir, rerun)

    println("DS-MFG QUBO example")
    println("  source: ", rerun ? "local Aer simulation" : "cached distributions")
    println("  summary: ", result.summary_path)
    println("  plot: ", result.svg_path)
    println("  top sampled states:")

    rows = sort(result.rows; by = row -> -row.probability)
    for row in rows[1:min(6, length(rows))]
        println(
            "    ",
            row.algorithm,
            " ",
            row.bitstring,
            ": p=",
            round(row.probability; digits = 3),
            ", raw=",
            round(row.raw_qubo_energy; digits = 3),
            ", projected=",
            round(row.projected_objective; digits = 3),
            ", match=",
            row.match,
        )
    end
    return result
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
