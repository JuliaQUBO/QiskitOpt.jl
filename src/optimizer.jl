# ============================================================================
#optimizer.jl — Pure-Julia derivative-free classical optimizer
#
# Implements a deterministic, derivative-free Hooke-Jeeves / compass trust-region
# optimizer designed for variational quantum objectives (QAOA, VQE, PCE) under
# both exact statevector/MPS evaluation and finite-shot sampling noise.
# ============================================================================

"""
    NativeOptimizerResult

Result returned by [`minimize_objective`](@ref), mirroring the fields queried by
`optimizer_result_metadata` (`x`, `fun`, `success`, `status`, `message`, `nit`, `nfev`).
"""
struct NativeOptimizerResult
    x::Vector{Float64}
    fun::Float64
    success::Bool
    status::Int
    message::String
    nit::Int
    nfev::Int
end

"""
    minimize_objective(f, x0::AbstractVector{<:Real}; max_iter::Integer = 15, rhobeg::Real = 0.5, rhoend::Real = 1.0e-4) -> NativeOptimizerResult

Minimize a scalar objective `f(x)` starting from `x0` using a deterministic
derivative-free trust-region pattern search (Hooke-Jeeves / compass search).

Handles edge cases cleanly:
- `isempty(x0)` (`d == 0`): evaluates `f(Float64[])` once and returns `nit = 0`, `nfev = 1`, `success = true`.
- `max_iter <= 1`: evaluates `f(x0)` once and returns `nit = max(0, Int(max_iter))`, `nfev = 1`.
- `max_iter > 1`: performs at most `max_iter` total function evaluations so high-dimensional
  ansätze (e.g. `8n`-parameter `EfficientSU2`) remain bounded by `max_iter`.
"""
function minimize_objective(
    f::Function,
    x0::AbstractVector{<:Real};
    max_iter::Integer = 15,
    rhobeg::Real = 0.5,
    rhoend::Real = 1.0e-4,
)
    d = length(x0)
    x_best = Float64[Float64(v) for v in x0]
    f_best = Float64(f(copy(x_best)))
    nfev = 1

    if d == 0
        return NativeOptimizerResult(
            x_best,
            f_best,
            true,
            0,
            "Optimization terminated successfully (0 parameters).",
            0,
            nfev,
        )
    end

    if max_iter <= 0
        return NativeOptimizerResult(
            x_best,
            f_best,
            false,
            2,
            "Maximum number of iterations has been exceeded.",
            0,
            nfev,
        )
    end

    if max_iter == 1
        return NativeOptimizerResult(
            x_best,
            f_best,
            false,
            2,
            "Maximum number of function evaluations has been exceeded.",
            1,
            nfev,
        )
    end

    rho = Float64(rhobeg)
    rho_min = Float64(rhoend)
    signs = ones(Float64, d)
    nit = 0
    converged = false

    while nfev < max_iter
        if rho < rho_min
            converged = true
            break
        end

        x_sweep_start = copy(x_best)
        improved_in_sweep = false

        # On the first sweep when d >= 2, probe a simultaneous multi-coordinate
        # rotation so two-qubit correlator objectives (e.g. PCE <XX>, <YY> starting
        # at θ = 0) escape the single-qubit product-state stationary manifold.
        if nit == 0 && d >= 2 && nfev < max_iter
            nit += 1
            x_diag = Vector{Float64}(undef, d)
            @inbounds for i in 1:d
                x_diag[i] = x_best[i] + (isodd(i) ? 1.5 : -1.5) * rho
            end
            f_diag = Float64(f(copy(x_diag)))
            nfev += 1
            if f_diag < f_best
                x_best = x_diag
                f_best = f_diag
                improved_in_sweep = true
            end
        end

        for coord in 1:d
            nfev >= max_iter && break
            nit += 1

            # Try primary direction along coordinate `coord`
            dir = signs[coord]
            x_try = copy(x_best)
            x_try[coord] += dir * rho
            f_try = Float64(f(copy(x_try)))
            nfev += 1

            if f_try < f_best
                x_best = x_try
                f_best = f_try
                improved_in_sweep = true
                continue
            end

            # If budget remains, try opposite direction along `coord`
            if nfev < max_iter
                x_opp = copy(x_best)
                x_opp[coord] -= dir * rho
                f_opp = Float64(f(copy(x_opp)))
                nfev += 1

                if f_opp < f_best
                    x_best = x_opp
                    f_best = f_opp
                    signs[coord] = -dir
                    improved_in_sweep = true
                else
                    signs[coord] = -dir
                end
            else
                signs[coord] = -dir
            end
        end

        nfev >= max_iter && break

        if improved_in_sweep
            # Attempt a Hooke-Jeeves pattern step along the net displacement
            x_pat = Vector{Float64}(undef, d)
            @inbounds for i in 1:d
                x_pat[i] = x_best[i] + (x_best[i] - x_sweep_start[i])
            end
            f_pat = Float64(f(copy(x_pat)))
            nfev += 1
            if f_pat < f_best
                x_best = x_pat
                f_best = f_pat
            else
                rho *= 0.5
            end
        else
            rho *= 0.5
        end
    end

    if rho < rho_min
        converged = true
    end

    status = converged ? 0 : 2
    message = if converged
        "Optimization terminated successfully."
    else
        "Maximum number of function evaluations has been exceeded."
    end

    return NativeOptimizerResult(
        x_best,
        f_best,
        converged,
        status,
        message,
        max(nit, 1),
        nfev,
    )
end

