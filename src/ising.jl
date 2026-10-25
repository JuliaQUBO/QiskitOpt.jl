const _LibQiskit = Qiskit.C.LibQiskit

"""
    PauliTerm(coeff, qubits, paulis)

Single tensor-product Pauli string term `coeff * ⊗_{k} P_{paulis[k]}^{(qubits[k])}`
with 1-indexed `qubits` sorted in ascending order and `paulis` in `(:X, :Y, :Z)`.
"""
struct PauliTerm
    coeff::Float64
    qubits::Vector{Int}
    paulis::Vector{Symbol}

    function PauliTerm(
        coeff::Real,
        qubits::AbstractVector{<:Integer},
        paulis::AbstractVector{Symbol},
    )
        length(qubits) == length(paulis) ||
            throw(ArgumentError("PauliTerm qubits and paulis must have the same length."))
        filtered_q = Int[]
        filtered_p = Symbol[]
        for (q_raw, p) in zip(qubits, paulis)
            q = Int(q_raw)
            q >= 1 || throw(ArgumentError("PauliTerm qubit indices must be >= 1; got $q."))
            if p === :I
                continue
            elseif p === :X || p === :Y || p === :Z
                push!(filtered_q, q)
                push!(filtered_p, p)
            else
                throw(ArgumentError("Unsupported Pauli symbol: $p (expected :I, :X, :Y, or :Z)."))
            end
        end
        if length(filtered_q) > 1
            perm = sortperm(filtered_q)
            filtered_q = filtered_q[perm]
            filtered_p = filtered_p[perm]
            for k in 2:length(filtered_q)
                filtered_q[k] == filtered_q[k - 1] &&
                    throw(ArgumentError("Duplicate qubit index $(filtered_q[k]) in PauliTerm."))
            end
        end
        return new(Float64(coeff), filtered_q, filtered_p)
    end
end

"""
    IsingHamiltonian

Native Julia representation of an Ising / Pauli Hamiltonian on `num_qubits` qubits:
`offset * I + ∑_i linear[i] Z_i + ∑_{i < j} quadratic[(i, j)] Z_i Z_j`
(plus any general non-identity `PauliTerm`s in `terms`).
"""
struct IsingHamiltonian
    num_qubits::Int
    offset::Float64
    linear::Vector{Float64}
    quadratic::Dict{Tuple{Int,Int},Float64}
    terms::Vector{PauliTerm}

    function IsingHamiltonian(
        num_qubits::Integer,
        offset::Real,
        linear::AbstractVector{<:Real},
        quadratic::AbstractDict{Tuple{Int,Int},<:Real},
        terms::AbstractVector{PauliTerm} = PauliTerm[];
        atol::Real = 1e-15,
    )
        n = Int(num_qubits)
        n >= 0 || throw(ArgumentError("num_qubits must be non-negative; got $n."))
        length(linear) == n ||
            throw(ArgumentError("linear coefficient vector length ($(length(linear))) must match num_qubits ($n)."))

        lin = Float64[abs(Float64(v)) > atol ? Float64(v) : 0.0 for v in linear]
        quad = Dict{Tuple{Int,Int},Float64}()
        for ((i_raw, j_raw), v_raw) in quadratic
            i, j = Int(i_raw), Int(j_raw)
            (1 <= i <= n && 1 <= j <= n && i != j) ||
                throw(ArgumentError("Invalid quadratic index pair ($i, $j) for $n qubits."))
            key = i < j ? (i, j) : (j, i)
            val = get(quad, key, 0.0) + Float64(v_raw)
            if abs(val) > atol
                quad[key] = val
            else
                delete!(quad, key)
            end
        end

        total_offset = Float64(offset)
        all_terms = PauliTerm[]
        if isempty(terms)
            for i in 1:n
                if !iszero(lin[i])
                    push!(all_terms, PauliTerm(lin[i], [i], [:Z]))
                end
            end
            for (i, j) in sort!(collect(keys(quad)))
                val = quad[(i, j)]
                if !iszero(val)
                    push!(all_terms, PauliTerm(val, [i, j], [:Z, :Z]))
                end
            end
        else
            for term in terms
                for q in term.qubits
                    1 <= q <= n ||
                        throw(ArgumentError("PauliTerm qubit $q exceeds num_qubits ($n)."))
                end
                if isempty(term.qubits)
                    total_offset += term.coeff
                elseif abs(term.coeff) > atol
                    push!(all_terms, term)
                end
            end
        end

        return new(n, total_offset, lin, quad, all_terms)
    end
end

function is_diagonal_ising(h::IsingHamiltonian)::Bool
    return all(term -> all(==(:Z), term.paulis), h.terms)
end

"""
    qubo_to_ising(n, L, Q, sense = MOI.MinSense; atol = 1e-15) -> IsingHamiltonian

Lower a sense-signed QUBO objective over binary variables `x ∈ {0, 1}^n` using
`x_i = (1 - Z_i) / 2` and `x_i^2 = x_i`:
- `w_i = sign * L_i + sign * Q_{i,i}`
- `w_{i,j} = sign * (Q_{i,j} + Q_{j,i})` for `1 <= i < j <= n`
- `offset = 1/2 * ∑_i w_i + 1/4 * ∑_{i < j} w_{i,j}`
- `h_i = -1/2 * w_i - 1/4 * ∑_{j != i} w_{min(i,j), max(i,j)}`
- `J_{i,j} = 1/4 * w_{i,j}` for `1 <= i < j <= n`
"""
function qubo_to_ising(
    n::Integer,
    L::AbstractDict,
    Q::AbstractDict,
    sense_pos::MOI.OptimizationSense;
    atol::Real = 1e-15,
)::IsingHamiltonian
    return qubo_to_ising(n, L, Q; sense = sense_pos, atol = atol)
end

function qubo_to_ising(
    n::Integer,
    L::AbstractDict,
    Q::AbstractDict;
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
    atol::Real = 1e-15,
)::IsingHamiltonian
    num_qubits = Int(n)
    num_qubits >= 0 || throw(ArgumentError("Number of variables must be non-negative."))
    sign = sense == MOI.MIN_SENSE ? 1.0 : -1.0

    w_linear = zeros(Float64, num_qubits)
    for (i_raw, v) in L
        i = Int(i_raw)
        1 <= i <= num_qubits || throw(ArgumentError("Linear index $i out of bounds for $num_qubits qubits."))
        w_linear[i] += sign * Float64(v)
    end

    w_quad = Dict{Tuple{Int,Int},Float64}()
    for ((i_raw, j_raw), v) in Q
        i, j = Int(i_raw), Int(j_raw)
        (1 <= i <= num_qubits && 1 <= j <= num_qubits) ||
            throw(ArgumentError("Quadratic index ($i, $j) out of bounds for $num_qubits qubits."))
        coeff = sign * Float64(v)
        if i == j
            w_linear[i] += coeff
        else
            key = i < j ? (i, j) : (j, i)
            w_quad[key] = get(w_quad, key, 0.0) + coeff
        end
    end

    offset = 0.5 * sum(w_linear; init = 0.0) + 0.25 * sum(values(w_quad); init = 0.0)
    h_linear = -0.5 .* w_linear
    J_quad = Dict{Tuple{Int,Int},Float64}()

    for ((i, j), wij) in w_quad
        if abs(wij) > atol
            h_linear[i] -= 0.25 * wij
            h_linear[j] -= 0.25 * wij
            J_quad[(i, j)] = 0.25 * wij
        end
    end

    return IsingHamiltonian(num_qubits, offset, h_linear, J_quad; atol = atol)
end

function qubo_to_ising(
    n::Integer,
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real},
    sense_pos::MOI.OptimizationSense;
    atol::Real = 1e-15,
)::IsingHamiltonian
    return qubo_to_ising(n, L, Q; sense = sense_pos, atol = atol)
end

function qubo_to_ising(
    n::Integer,
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real};
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
    atol::Real = 1e-15,
)::IsingHamiltonian
    num_qubits = Int(n)
    num_qubits >= 0 || throw(ArgumentError("Number of variables must be non-negative."))
    length(L) == num_qubits || throw(ArgumentError("L length ($(length(L))) must match n ($num_qubits)."))
    size(Q) == (num_qubits, num_qubits) ||
        throw(ArgumentError("Q size $(size(Q)) must match ($num_qubits, $num_qubits)."))

    L_dict = Dict{Int,Float64}(i => Float64(L[i]) for i in 1:num_qubits if !iszero(L[i]))
    Q_dict = Dict{Tuple{Int,Int},Float64}()
    for i in 1:num_qubits, j in 1:num_qubits
        if !iszero(Q[i, j])
            Q_dict[(i, j)] = Float64(Q[i, j])
        end
    end
    return qubo_to_ising(num_qubits, L_dict, Q_dict; sense = sense, atol = atol)
end

function qubo_to_ising(
    sampler::QUBODrivers.AbstractSampler;
    atol::Real = 1e-15,
)::IsingHamiltonian
    n, L, Q, _, _ = QUBOTools.qubo(sampler, :dict)
    sense = MOI.get(sampler, MOI.ObjectiveSense())
    return qubo_to_ising(n, L, Q, sense; atol = atol)
end

"""
    evaluate_ising_bitstring(h::IsingHamiltonian, x::AbstractVector{<:Real}; include_offset::Bool = true)

Evaluate a diagonal Ising Hamiltonian on a binary assignment `x ∈ {0, 1}^n` using `z_i = 1 - 2x_i ∈ {+1, -1}`.
"""
function evaluate_ising_bitstring(
    h::IsingHamiltonian,
    x::AbstractVector{<:Real};
    include_offset::Bool = true,
)::Float64
    length(x) == h.num_qubits ||
        throw(ArgumentError("Bitstring length ($(length(x))) must match num_qubits ($(h.num_qubits))."))
    total = include_offset ? h.offset : 0.0
    for term in h.terms
        prod_z = 1.0
        for (q, p) in zip(term.qubits, term.paulis)
            p === :Z ||
                throw(ArgumentError("evaluate_ising_bitstring requires diagonal :Z terms; got $p on qubit $q."))
            prod_z *= (1.0 - 2.0 * Float64(x[q]))
        end
        total += term.coeff * prod_z
    end
    return total
end

function _symbol_to_bitterm(p::Symbol)::_LibQiskit.QkBitTerm
    if p === :X
        return _LibQiskit.QkBitTerm_X
    elseif p === :Y
        return _LibQiskit.QkBitTerm_Y
    elseif p === :Z
        return _LibQiskit.QkBitTerm_Z
    else
        throw(ArgumentError("Unsupported Pauli symbol for SparseObservable: $p"))
    end
end

function _bitterm_to_symbol(bt::_LibQiskit.QkBitTerm)::Symbol
    if bt == _LibQiskit.QkBitTerm_X
        return :X
    elseif bt == _LibQiskit.QkBitTerm_Y
        return :Y
    elseif bt == _LibQiskit.QkBitTerm_Z
        return :Z
    else
        label = Char(_LibQiskit.qk_bitterm_label(bt))
        label == 'X' && return :X
        label == 'Y' && return :Y
        label == 'Z' && return :Z
        label == 'I' && return :I
        throw(ArgumentError("Unsupported QkBitTerm label '$label' ($bt) in SparseObservable."))
    end
end

"""
    to_sparse_observable(h::IsingHamiltonian; include_offset::Bool = false) -> Qiskit.SparseObservable

Construct a `Qiskit.SparseObservable` from an `IsingHamiltonian` via `LibQiskit.qk_obs_add_term`
using 0-based `UInt32` qubit indices and `GC.@preserve`.
"""
function to_sparse_observable(
    h::IsingHamiltonian;
    include_offset::Bool = false,
)::Qiskit.SparseObservable
    obs = Qiskit.SparseObservable(h.num_qubits)
    if include_offset && !iszero(h.offset)
        c_id = _LibQiskit.QkObsTerm(
            _LibQiskit.QkComplex64(h.offset, 0.0),
            Csize_t(0),
            Ptr{_LibQiskit.QkBitTerm}(C_NULL),
            Ptr{UInt32}(C_NULL),
            UInt32(h.num_qubits),
        )
        GC.@preserve obs begin
            code = _LibQiskit.qk_obs_add_term(obs.ptr, Ref(c_id))
            code == _LibQiskit.QkExitCode_Success ||
                error("LibQiskit.qk_obs_add_term failed with exit code $code.")
        end
    end

    for term in h.terms
        iszero(term.coeff) && continue
        len = length(term.qubits)
        bit_terms = _LibQiskit.QkBitTerm[_symbol_to_bitterm(p) for p in term.paulis]
        indices0 = UInt32[UInt32(q - 1) for q in term.qubits]
        c_term = _LibQiskit.QkObsTerm(
            _LibQiskit.QkComplex64(term.coeff, 0.0),
            Csize_t(len),
            len == 0 ? Ptr{_LibQiskit.QkBitTerm}(C_NULL) : pointer(bit_terms),
            len == 0 ? Ptr{UInt32}(C_NULL) : pointer(indices0),
            UInt32(h.num_qubits),
        )
        GC.@preserve obs bit_terms indices0 begin
            code = _LibQiskit.qk_obs_add_term(obs.ptr, Ref(c_term))
            code == _LibQiskit.QkExitCode_Success ||
                error("LibQiskit.qk_obs_add_term failed with exit code $code.")
        end
    end
    return obs
end

"""
    parse_sparse_observable(obs::Qiskit.SparseObservable; offset::Real = 0.0, atol::Real = 1e-12) -> IsingHamiltonian

Convert a `Qiskit.SparseObservable` into an `IsingHamiltonian` by reading terms via `LibQiskit.qk_obs_term`.
"""
function parse_sparse_observable(
    obs::Qiskit.SparseObservable;
    offset::Real = 0.0,
    atol::Real = 1e-12,
)::IsingHamiltonian
    num_qubits = Qiskit.qk_obs_num_qubits(obs)
    num_terms = Qiskit.qk_obs_num_terms(obs)
    total_offset = Float64(offset)

    merged_coeffs = Dict{Tuple{Vector{Int},Vector{Symbol}},Float64}()
    term_order = Vector{Tuple{Vector{Int},Vector{Symbol}}}()

    for idx in 0:(num_terms - 1)
        term_ref = Ref(
            _LibQiskit.QkObsTerm(
                _LibQiskit.QkComplex64(0.0, 0.0),
                Csize_t(0),
                Ptr{_LibQiskit.QkBitTerm}(C_NULL),
                Ptr{UInt32}(C_NULL),
                UInt32(0),
            ),
        )
        GC.@preserve obs begin
            code = _LibQiskit.qk_obs_term(obs.ptr, UInt64(idx), term_ref)
            code == _LibQiskit.QkExitCode_Success ||
                error("LibQiskit.qk_obs_term failed at index $idx with exit code $code.")
            raw = term_ref[]
            abs(Float64(raw.coeff.im)) <= atol ||
                throw(ArgumentError("Observable term at index $idx has non-zero imaginary coefficient $(raw.coeff.im)."))
            coeff = Float64(raw.coeff.re)
            len = Int(raw.len)
            if len == 0
                total_offset += coeff
            else
                qubits = Int[Int(unsafe_load(raw.indices, k)) + 1 for k in 1:len]
                paulis = Symbol[_bitterm_to_symbol(unsafe_load(raw.bit_terms, k)) for k in 1:len]
                pt = PauliTerm(coeff, qubits, paulis)
                if isempty(pt.qubits)
                    total_offset += pt.coeff
                else
                    key = (pt.qubits, pt.paulis)
                    if !haskey(merged_coeffs, key)
                        push!(term_order, key)
                        merged_coeffs[key] = pt.coeff
                    else
                        merged_coeffs[key] += pt.coeff
                    end
                end
            end
        end
    end

    linear = zeros(Float64, num_qubits)
    quadratic = Dict{Tuple{Int,Int},Float64}()
    terms = PauliTerm[]

    for key in term_order
        coeff = merged_coeffs[key]
        abs(coeff) <= 1e-15 && continue
        qs, ps = key
        push!(terms, PauliTerm(coeff, qs, ps))
        if length(qs) == 1 && ps[1] === :Z
            linear[qs[1]] += coeff
        elseif length(qs) == 2 && ps[1] === :Z && ps[2] === :Z
            pair = (qs[1], qs[2])
            quadratic[pair] = get(quadratic, pair, 0.0) + coeff
        end
    end

    return IsingHamiltonian(num_qubits, total_offset, linear, quadratic, terms)
end

"""
    apply_layout_to_hamiltonian(h::IsingHamiltonian, layout_perm::AbstractVector{<:Integer}; num_physical_qubits::Integer = max(h.num_qubits, isempty(layout_perm) ? 0 : maximum(layout_perm))) -> IsingHamiltonian

Remap logical qubit indices `q` of `h` to physical qubit indices `layout_perm[q]`.
"""
function apply_layout_to_hamiltonian(
    h::IsingHamiltonian,
    layout::Qiskit.TranspileLayout;
    num_physical_qubits::Union{Nothing,Integer} = nothing,
)::IsingHamiltonian
    perm = transpile_layout_final_permutation(layout)
    n_phys = num_physical_qubits === nothing ?
        max(h.num_qubits, Int(_LibQiskit.qk_transpile_layout_num_output_qubits(layout.ptr)), isempty(perm) ? 0 : maximum(perm)) :
        Int(num_physical_qubits)
    return apply_layout_to_hamiltonian(h, perm; num_physical_qubits = n_phys)
end

function apply_layout_to_hamiltonian(
    h::IsingHamiltonian,
    layout_perm::AbstractVector{<:Integer};
    num_physical_qubits::Integer = max(h.num_qubits, isempty(layout_perm) ? 0 : maximum(layout_perm)),
)::IsingHamiltonian
    length(layout_perm) >= h.num_qubits ||
        throw(ArgumentError("layout_perm length ($(length(layout_perm))) is smaller than Hamiltonian num_qubits ($(h.num_qubits))."))
    n_phys = Int(num_physical_qubits)
    linear = zeros(Float64, n_phys)
    quadratic = Dict{Tuple{Int,Int},Float64}()
    mapped_terms = PauliTerm[]

    for term in h.terms
        mapped_q = Int[Int(layout_perm[q]) for q in term.qubits]
        pt = PauliTerm(term.coeff, mapped_q, term.paulis)
        push!(mapped_terms, pt)
        if length(pt.qubits) == 1 && pt.paulis[1] === :Z
            linear[pt.qubits[1]] += pt.coeff
        elseif length(pt.qubits) == 2 && pt.paulis[1] === :Z && pt.paulis[2] === :Z
            pair = (pt.qubits[1], pt.qubits[2])
            quadratic[pair] = get(quadratic, pair, 0.0) + pt.coeff
        end
    end

    return IsingHamiltonian(n_phys, h.offset, linear, quadratic, mapped_terms)
end

function _coerce_custom_observable(observable, num_qubits::Int, default_offset::Float64)
    throw(
        ArgumentError(
            "Unsupported Observable type $(typeof(observable)). Expected `nothing`, `Qiskit.SparseObservable`, `IsingHamiltonian`, or `AbstractVector{PauliTerm}`.",
        ),
    )
end

function coerce_ising_hamiltonian(
    observable,
    default_hamiltonian::IsingHamiltonian;
    offset_override = nothing,
)::IsingHamiltonian
    if observable === nothing
        if offset_override === nothing
            return default_hamiltonian
        else
            return IsingHamiltonian(
                default_hamiltonian.num_qubits,
                Float64(offset_override),
                default_hamiltonian.linear,
                default_hamiltonian.quadratic,
                default_hamiltonian.terms,
            )
        end
    elseif observable isa IsingHamiltonian
        observable.num_qubits == default_hamiltonian.num_qubits ||
            throw(ArgumentError("Observable num_qubits ($(observable.num_qubits)) does not match model variable count ($(default_hamiltonian.num_qubits))."))
        if offset_override === nothing
            return observable
        else
            return IsingHamiltonian(
                observable.num_qubits,
                Float64(offset_override),
                observable.linear,
                observable.quadratic,
                observable.terms,
            )
        end
    elseif observable isa Qiskit.SparseObservable
        obs_qubits = Qiskit.qk_obs_num_qubits(observable)
        obs_qubits == default_hamiltonian.num_qubits ||
            throw(ArgumentError("SparseObservable num_qubits ($obs_qubits) does not match model variable count ($(default_hamiltonian.num_qubits))."))
        base_offset = offset_override === nothing ? default_hamiltonian.offset : Float64(offset_override)
        return parse_sparse_observable(observable; offset = base_offset)
    elseif observable isa AbstractVector{PauliTerm}
        base_offset = offset_override === nothing ? default_hamiltonian.offset : Float64(offset_override)
        n = default_hamiltonian.num_qubits
        linear = zeros(Float64, n)
        quadratic = Dict{Tuple{Int,Int},Float64}()
        for term in observable
            for q in term.qubits
                1 <= q <= n || throw(ArgumentError("PauliTerm qubit $q exceeds model variable count ($n)."))
            end
            if length(term.qubits) == 1 && term.paulis[1] === :Z
                linear[term.qubits[1]] += term.coeff
            elseif length(term.qubits) == 2 && term.paulis[1] === :Z && term.paulis[2] === :Z
                pair = (term.qubits[1], term.qubits[2])
                quadratic[pair] = get(quadratic, pair, 0.0) + term.coeff
            end
        end
        return IsingHamiltonian(n, base_offset, linear, quadratic, collect(PauliTerm, observable))
    else
        base_offset = offset_override === nothing ? default_hamiltonian.offset : Float64(offset_override)
        return _coerce_custom_observable(observable, default_hamiltonian.num_qubits, base_offset)
    end
end
