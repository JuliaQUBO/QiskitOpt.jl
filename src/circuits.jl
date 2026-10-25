const _GATE_LOOKUP = Dict{Symbol,Tuple{_LibQiskit.QkGate,Int,Int}}(
    name => (gate, nq, np) for (name, gate, nq, np) in Qiskit.GATE_TABLE
)

"""
    NativeAnsatz

Parameterized circuit builder descriptor on `num_qubits` qubits with ordered `parameter_names`.
Calling `ansatz(theta)` or `bind_ansatz_parameters(ansatz, theta)` produces a concrete
`Qiskit.QuantumCircuit`.
"""
struct NativeAnsatz
    num_qubits::Int
    parameter_names::Vector{String}
    build::Function
    name::String
    reps::Int
    su2_gates::Vector{Symbol}
    entanglement::Any
    layout::Union{Nothing,Qiskit.TranspileLayout}
    layout_permutation::Union{Nothing,Vector{Int}}

    function NativeAnsatz(
        num_qubits::Integer,
        parameter_names::AbstractVector{<:AbstractString},
        build::Function;
        name::AbstractString = "custom_ansatz",
        reps::Integer = 1,
        su2_gates::AbstractVector{Symbol} = Symbol[],
        entanglement = :custom,
        layout::Union{Nothing,Qiskit.TranspileLayout} = nothing,
        layout_permutation::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    )
        n = Int(num_qubits)
        n >= 0 || throw(ArgumentError("num_qubits must be non-negative; got $n."))
        perm = layout_permutation === nothing ? nothing : Int[Int(q) for q in layout_permutation]
        return new(
            n,
            String[String(p) for p in parameter_names],
            build,
            String(name),
            Int(reps),
            collect(Symbol, su2_gates),
            entanglement,
            layout,
            perm,
        )
    end
end

num_parameters(ansatz::NativeAnsatz)::Int = length(getfield(ansatz, :parameter_names))
num_parameters(qc::Qiskit.QuantumCircuit)::Int = 0

function Base.getproperty(ansatz::NativeAnsatz, sym::Symbol)
    if sym === :assign_parameters
        return theta -> bind_ansatz_parameters(ansatz, theta)
    elseif sym === :parameters
        return getfield(ansatz, :parameter_names)
    elseif sym === :num_parameters
        return length(getfield(ansatz, :parameter_names))
    end
    return getfield(ansatz, sym)
end

assign_parameters(ansatz::NativeAnsatz, theta::AbstractVector{<:Real}) =
    bind_ansatz_parameters(ansatz, theta)
assign_parameters(qc::Qiskit.QuantumCircuit, theta::AbstractVector{<:Real}) =
    bind_ansatz_parameters(qc, theta)

function (ansatz::NativeAnsatz)(theta::AbstractVector{<:Real})::Qiskit.QuantumCircuit
    return bind_ansatz_parameters(ansatz, theta)
end

function bind_ansatz_parameters(
    ansatz::NativeAnsatz,
    theta::AbstractVector{<:Real},
)::Qiskit.QuantumCircuit
    expected = length(ansatz.parameter_names)
    length(theta) == expected ||
        throw(ArgumentError("Expected $expected ansatz parameters; got $(length(theta))."))
    qc = ansatz.build(Float64[Float64(v) for v in theta])
    qc isa Qiskit.QuantumCircuit ||
        throw(ArgumentError("NativeAnsatz builder must return a Qiskit.QuantumCircuit; got $(typeof(qc))."))
    qc.num_qubits == ansatz.num_qubits ||
        throw(ArgumentError("Ansatz circuit num_qubits ($(qc.num_qubits)) does not match expected ($(ansatz.num_qubits))."))
    return qc
end

function bind_ansatz_parameters(
    qc::Qiskit.QuantumCircuit,
    theta::AbstractVector{<:Real},
)::Qiskit.QuantumCircuit
    isempty(theta) ||
        throw(ArgumentError("Concrete Qiskit.QuantumCircuit has 0 parameters; got $(length(theta))."))
    return copy(qc)
end

function _resolve_custom_ansatz(ansatz_spec, num_qubits::Int; reps::Union{Nothing,Int} = nothing)
    throw(
        ArgumentError(
            "Unsupported Ansatz specification of type $(typeof(ansatz_spec)). Expected `NativeAnsatz`, `Qiskit.QuantumCircuit`, or a builder function.",
        ),
    )
end

"""
    resolve_ansatz(ansatz_spec, num_qubits::Integer; reps = nothing) -> NativeAnsatz

Normalize an ansatz specification (`NativeAnsatz`, 0-parameter `Qiskit.QuantumCircuit`, or
factory function) into a `NativeAnsatz` on `num_qubits` qubits.
"""
function resolve_ansatz(
    ansatz_spec,
    num_qubits::Integer;
    reps::Union{Nothing,Integer} = nothing,
)::NativeAnsatz
    n = Int(num_qubits)
    n >= 0 || throw(ArgumentError("num_qubits must be non-negative; got $n."))
    if ansatz_spec isa NativeAnsatz
        ansatz_spec.num_qubits == n ||
            throw(ArgumentError("Ansatz num_qubits ($(ansatz_spec.num_qubits)) does not match expected ($n)."))
        return ansatz_spec
    elseif ansatz_spec isa Qiskit.QuantumCircuit
        ansatz_spec.num_qubits == n ||
            throw(ArgumentError("Ansatz circuit num_qubits ($(ansatz_spec.num_qubits)) does not match expected ($n)."))
        return NativeAnsatz(
            n,
            String[],
            _ -> copy(ansatz_spec);
            name = "QuantumCircuit",
            reps = 0,
        )
    elseif ansatz_spec isa Function
        built = if reps !== nothing
            try
                ansatz_spec(; num_qubits = n, reps = Int(reps))
            catch err
                err isa MethodError || rethrow()
                try
                    ansatz_spec(; num_qubits = n)
                catch err2
                    err2 isa MethodError || rethrow()
                    try
                        ansatz_spec(n, Int(reps))
                    catch err3
                        err3 isa MethodError || rethrow()
                        ansatz_spec(n)
                    end
                end
            end
        else
            try
                ansatz_spec(; num_qubits = n)
            catch err
                err isa MethodError || rethrow()
                ansatz_spec(n)
            end
        end
        return resolve_ansatz(built, n; reps = reps)
    else
        return _resolve_custom_ansatz(
            ansatz_spec,
            n;
            reps = reps === nothing ? nothing : Int(reps),
        )
    end
end

function _normalize_su2_gates(su2_gates)::Vector{Symbol}
    if su2_gates isa Symbol
        return [su2_gates]
    elseif su2_gates isa AbstractString
        return [Symbol(lowercase(String(su2_gates)))]
    elseif su2_gates isa AbstractVector || su2_gates isa Tuple
        gates = Symbol[Symbol(lowercase(String(g))) for g in su2_gates]
        isempty(gates) && throw(ArgumentError("su2_gates must not be empty."))
        for g in gates
            haskey(_GATE_LOOKUP, g) || throw(ArgumentError("Unsupported SU(2) gate: $g."))
            _, nq, np = _GATE_LOOKUP[g]
            (nq == 1 && np == 1) ||
                throw(ArgumentError("SU(2) gate $g must be a 1-qubit, 1-parameter rotation gate."))
        end
        return gates
    else
        throw(ArgumentError("Invalid su2_gates specification: $(typeof(su2_gates))."))
    end
end

function _entanglement_pairs(num_qubits::Int, entanglement; layer::Int = 1)::Vector{Tuple{Int,Int}}
    num_qubits <= 1 && return Tuple{Int,Int}[]
    if entanglement isa AbstractString || entanglement isa Symbol
        mode = Symbol(lowercase(String(entanglement)))
        if mode === :reverse_linear
            return Tuple{Int,Int}[(q, q + 1) for q in (num_qubits - 1):-1:1]
        elseif mode === :linear
            return Tuple{Int,Int}[(q, q + 1) for q in 1:(num_qubits - 1)]
        elseif mode === :full
            pairs = Tuple{Int,Int}[]
            for i in 1:(num_qubits - 1)
                for j in (i + 1):num_qubits
                    push!(pairs, (i, j))
                end
            end
            return pairs
        elseif mode === :circular
            pairs = Tuple{Int,Int}[(num_qubits, 1)]
            for q in 1:(num_qubits - 1)
                push!(pairs, (q, q + 1))
            end
            return pairs
        elseif mode === :sca
            base = Tuple{Int,Int}[(num_qubits, 1)]
            for q in 1:(num_qubits - 1)
                push!(base, (q, q + 1))
            end
            shift = mod(layer - 1, length(base))
            rotated = vcat(base[(shift + 1):end], base[1:shift])
            if iseven(layer)
                return Tuple{Int,Int}[(b, a) for (a, b) in rotated]
            else
                return rotated
            end
        else
            throw(ArgumentError("Unsupported entanglement mode: $entanglement."))
        end
    elseif entanglement isa AbstractVector
        pairs = Tuple{Int,Int}[]
        for item in entanglement
            length(item) == 2 || throw(ArgumentError("Entanglement pair must have length 2; got $item."))
            q1, q2 = Int(item[1]), Int(item[2])
            (1 <= q1 <= num_qubits && 1 <= q2 <= num_qubits && q1 != q2) ||
                throw(ArgumentError("Invalid entanglement pair ($q1, $q2) for $num_qubits qubits."))
            push!(pairs, (q1, q2))
        end
        return pairs
    else
        throw(ArgumentError("Unsupported entanglement specification: $(typeof(entanglement))."))
    end
end

"""
    efficient_su2_ansatz(num_qubits; reps = 3, su2_gates = [:ry, :rz], entanglement = :reverse_linear) -> NativeAnsatz

Construct a hardware-efficient SU(2) ansatz matching Qiskit's `efficient_su2` defaults:
`(reps + 1)` single-qubit rotation layers (`ry!`, `rz!` across `1:num_qubits`) interleaved
with `reps` two-qubit `cx!` entanglement layers (`(num_qubits - 1):-1:1` for `:reverse_linear`).
Default parameter count is `2 * (reps + 1) * num_qubits = 8 * num_qubits` for `reps = 3`.
"""
function efficient_su2_ansatz(
    num_qubits::Integer;
    reps::Integer = 3,
    su2_gates = [:ry, :rz],
    entanglement = :reverse_linear,
)::NativeAnsatz
    n = Int(num_qubits)
    r = Int(reps)
    n >= 0 || throw(ArgumentError("num_qubits must be non-negative; got $n."))
    r >= 0 || throw(ArgumentError("reps must be non-negative; got $r."))
    gates = _normalize_su2_gates(su2_gates)
    # Validate entanglement upfront for n >= 2
    if n >= 2
        _entanglement_pairs(n, entanglement; layer = 1)
    end
    num_params = n == 0 ? 0 : (r + 1) * length(gates) * n
    param_names = String["θ[$(i - 1)]" for i in 1:num_params]

    function _build(theta::Vector{Float64})::Qiskit.QuantumCircuit
        length(theta) == num_params ||
            throw(ArgumentError("Expected $num_params parameters for efficient_su2_ansatz; got $(length(theta))."))
        qc = Qiskit.QuantumCircuit(n, 0)
        n == 0 && return qc
        idx = 1
        for layer in 0:r
            for g in gates
                gate_op, _, _ = _GATE_LOOKUP[g]
                for q in 1:n
                    Qiskit.qk_circuit_gate(qc, gate_op, Int32[q], Float64[theta[idx]])
                    idx += 1
                end
            end
            if layer < r && n >= 2
                for (q1, q2) in _entanglement_pairs(n, entanglement; layer = layer + 1)
                    Qiskit.cx!(qc, q1, q2)
                end
            end
        end
        return qc
    end

    return NativeAnsatz(
        n,
        param_names,
        _build;
        name = "efficient_su2",
        reps = r,
        su2_gates = gates,
        entanglement = entanglement,
    )
end

"""
    append_circuit!(dest::Qiskit.QuantumCircuit, src::Qiskit.QuantumCircuit; qubits = nothing, include_measurements::Bool = true)

Append the instructions of `src` onto `dest`.
"""
function append_circuit!(
    dest::Qiskit.QuantumCircuit,
    src::Qiskit.QuantumCircuit;
    qubits::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    include_measurements::Bool = true,
)::Qiskit.QuantumCircuit
    has_unitary = false
    for idx in firstindex(src.data):lastindex(src.data)
        if src.data[idx].name == "unitary"
            has_unitary = true
            break
        end
    end
    if has_unitary
        src = native_transpile(
            src;
            target = default_local_target(src.num_qubits),
            optimization_level = 0,
        ).circuit
    end

    if qubits === nothing
        src.num_qubits <= dest.num_qubits ||
            throw(ArgumentError("Source circuit has $(src.num_qubits) qubits, exceeding destination ($(dest.num_qubits))."))
        qmap = collect(1:src.num_qubits)
    else
        length(qubits) == src.num_qubits ||
            throw(ArgumentError("Qubit mapping length ($(length(qubits))) must match source num_qubits ($(src.num_qubits))."))
        qmap = Int[Int(q) for q in qubits]
    end

    dest_offset = dest.offset
    src_offset = src.offset
    for idx in firstindex(src.data):lastindex(src.data)
        inst = src.data[idx]
        mapped_q = Int32[Int32(qmap[q - src_offset + 1] + dest_offset - 1) for q in inst.qubits]
        name = inst.name
        if name == "measure"
            include_measurements || continue
            for (q_raw, c_raw) in zip(inst.qubits, inst.clbits)
                q_dest = qmap[q_raw - src_offset + 1] + dest_offset - 1
                c_dest = (c_raw - src_offset + 1) + dest_offset - 1
                Qiskit.measure!(dest, q_dest, c_dest)
            end
        elseif name == "barrier"
            if !isempty(mapped_q)
                Qiskit.barrier!(dest, Int.(mapped_q)...)
            end
        elseif name == "reset"
            for q_dest in mapped_q
                Qiskit.reset!(dest, Int(q_dest))
            end
        elseif name == "delay"
            continue
        elseif name == "u1"
            Qiskit.p!(dest, inst.params[1], Int(mapped_q[1]))
        elseif name == "u2"
            Qiskit.u!(dest, π / 2, inst.params[1], inst.params[2], Int(mapped_q[1]))
        elseif name == "u3"
            Qiskit.u!(dest, inst.params[1], inst.params[2], inst.params[3], Int(mapped_q[1]))
        elseif name == "cu1"
            Qiskit.cp!(dest, inst.params[1], Int(mapped_q[1]), Int(mapped_q[2]))
        elseif name == "c3x"
            Qiskit.rcccx!(dest, Int(mapped_q[1]), Int(mapped_q[2]), Int(mapped_q[3]), Int(mapped_q[4]))
        else
            sym = Symbol(name)
            if haskey(_GATE_LOOKUP, sym)
                gate_op, _, _ = _GATE_LOOKUP[sym]
                Qiskit.qk_circuit_gate(dest, gate_op, mapped_q, inst.params)
            else
                throw(ArgumentError("Unsupported circuit instruction '$name' when appending circuit."))
            end
        end
    end
    return dest
end

function _apply_pauli_evolution!(
    qc::Qiskit.QuantumCircuit,
    term::PauliTerm,
    angle::Real,
)::Qiskit.QuantumCircuit
    theta = 2.0 * Float64(angle) * term.coeff
    iszero(theta) && return qc
    k = length(term.qubits)
    k == 0 && return qc

    if k == 1
        q = term.qubits[1]
        p = term.paulis[1]
        if p === :Z
            Qiskit.rz!(qc, theta, q)
        elseif p === :X
            Qiskit.rx!(qc, theta, q)
        elseif p === :Y
            Qiskit.ry!(qc, theta, q)
        end
        return qc
    elseif k == 2 && term.paulis[1] === :Z && term.paulis[2] === :Z
        Qiskit.rzz!(qc, theta, term.qubits[1], term.qubits[2])
        return qc
    elseif k == 2 && term.paulis[1] === :X && term.paulis[2] === :X
        Qiskit.rxx!(qc, theta, term.qubits[1], term.qubits[2])
        return qc
    elseif k == 2 && term.paulis[1] === :Y && term.paulis[2] === :Y
        Qiskit.ryy!(qc, theta, term.qubits[1], term.qubits[2])
        return qc
    end

    # General multi-Pauli evolution gadget: basis change -> CX ladder -> RZ -> reverse CX ladder -> reverse basis change
    for (q, p) in zip(term.qubits, term.paulis)
        if p === :X
            Qiskit.h!(qc, q)
        elseif p === :Y
            Qiskit.rx!(qc, π / 2, q)
        end
    end
    for idx in 1:(k - 1)
        Qiskit.cx!(qc, term.qubits[idx], term.qubits[idx + 1])
    end
    Qiskit.rz!(qc, theta, term.qubits[k])
    for idx in (k - 1):-1:1
        Qiskit.cx!(qc, term.qubits[idx], term.qubits[idx + 1])
    end
    for (q, p) in zip(term.qubits, term.paulis)
        if p === :X
            Qiskit.h!(qc, q)
        elseif p === :Y
            Qiskit.rx!(qc, -π / 2, q)
        end
    end
    return qc
end

"""
    build_qaoa_circuit(hamiltonian::IsingHamiltonian, reps::Integer, theta::AbstractVector{<:Real}; initial_state = nothing, mixer = nothing) -> Qiskit.QuantumCircuit

Build a concrete `p = reps`-layer QAOA `Qiskit.QuantumCircuit` with parameter ordering
`["β[0]", ..., "β[p-1]", "γ[0]", ..., "γ[p-1]"]`:
- Initial state: `h!` on `1:n` (or custom `initial_state::Qiskit.QuantumCircuit`)
- Cost layer `l`: `rz!(qc, 2 * γ[l] * h_i, i)` and `rzz!(qc, 2 * γ[l] * J_{i,j}, i, j)`
- Mixer layer `l`: `rx!(qc, 2 * β[l], i)` (or custom `mixer`)
"""
function build_qaoa_circuit(
    hamiltonian::IsingHamiltonian,
    reps::Integer,
    theta::AbstractVector{<:Real};
    initial_state = nothing,
    mixer = nothing,
)::Qiskit.QuantumCircuit
    p = Int(reps)
    p >= 1 || throw(ArgumentError("Number of QAOA layers (p) must be >= 1; got $p."))
    n = hamiltonian.num_qubits
    expected_params = 2 * p
    length(theta) == expected_params ||
        throw(ArgumentError("Expected $expected_params QAOA parameters (2p for p=$p); got $(length(theta))."))

    beta = Float64[Float64(theta[l]) for l in 1:p]
    gamma = Float64[Float64(theta[p + l]) for l in 1:p]

    qc = Qiskit.QuantumCircuit(n, 0)
    n == 0 && return qc

    if initial_state === nothing
        for q in 1:n
            Qiskit.h!(qc, q)
        end
    elseif initial_state isa Qiskit.QuantumCircuit
        initial_state.num_qubits == n ||
            throw(ArgumentError("InitialState circuit num_qubits ($(initial_state.num_qubits)) does not match variable count ($n)."))
        append_circuit!(qc, initial_state; include_measurements = false)
    else
        throw(ArgumentError("Unsupported InitialState type $(typeof(initial_state)). Expected `nothing` or `Qiskit.QuantumCircuit`."))
    end

    standard_ising = is_diagonal_ising(hamiltonian) && all(t -> length(t.qubits) <= 2, hamiltonian.terms)
    sorted_quad_keys = sort!(collect(keys(hamiltonian.quadratic)))

    for l in 1:p
        # Cost unitary U_C(γ_l)
        if standard_ising
            for i in 1:n
                hi = hamiltonian.linear[i]
                if !iszero(hi)
                    Qiskit.rz!(qc, 2.0 * gamma[l] * hi, i)
                end
            end
            for (i, j) in sorted_quad_keys
                Jij = hamiltonian.quadratic[(i, j)]
                if !iszero(Jij)
                    Qiskit.rzz!(qc, 2.0 * gamma[l] * Jij, i, j)
                end
            end
        else
            for term in hamiltonian.terms
                _apply_pauli_evolution!(qc, term, gamma[l])
            end
        end

        # Mixer unitary U_M(β_l)
        if mixer === nothing
            for i in 1:n
                Qiskit.rx!(qc, 2.0 * beta[l], i)
            end
        elseif mixer isa Qiskit.QuantumCircuit
            mixer.num_qubits == n ||
                throw(ArgumentError("Mixer circuit num_qubits ($(mixer.num_qubits)) does not match variable count ($n)."))
            append_circuit!(qc, mixer; include_measurements = false)
        elseif mixer isa NativeAnsatz
            mixer.num_qubits == n ||
                throw(ArgumentError("Mixer ansatz num_qubits ($(mixer.num_qubits)) does not match variable count ($n)."))
            m_np = num_parameters(mixer)
            if m_np == 0
                append_circuit!(qc, mixer(Float64[]); include_measurements = false)
            elseif m_np == 1
                append_circuit!(qc, mixer([beta[l]]); include_measurements = false)
            else
                fill_params = fill(beta[l], m_np)
                append_circuit!(qc, mixer(fill_params); include_measurements = false)
            end
        elseif mixer isa IsingHamiltonian
            mixer.num_qubits == n ||
                throw(ArgumentError("Mixer Hamiltonian num_qubits ($(mixer.num_qubits)) does not match variable count ($n)."))
            for term in mixer.terms
                _apply_pauli_evolution!(qc, term, beta[l])
            end
        elseif mixer isa Qiskit.SparseObservable
            obs_n = Qiskit.qk_obs_num_qubits(mixer)
            obs_n == n ||
                throw(ArgumentError("Mixer SparseObservable num_qubits ($obs_n) does not match variable count ($n)."))
            mixer_h = parse_sparse_observable(mixer)
            for term in mixer_h.terms
                _apply_pauli_evolution!(qc, term, beta[l])
            end
        elseif mixer isa Function
            m_qc = mixer(beta[l])
            m_qc isa Qiskit.QuantumCircuit ||
                throw(ArgumentError("Callable Mixer must return a Qiskit.QuantumCircuit; got $(typeof(m_qc))."))
            m_qc.num_qubits == n ||
                throw(ArgumentError("Callable Mixer circuit num_qubits ($(m_qc.num_qubits)) does not match variable count ($n)."))
            append_circuit!(qc, m_qc; include_measurements = false)
        else
            throw(ArgumentError("Unsupported Mixer type $(typeof(mixer))."))
        end
    end

    return qc
end

"""
    qaoa_ansatz(hamiltonian::IsingHamiltonian; reps::Integer = 1, initial_state = nothing, mixer = nothing) -> NativeAnsatz

Create a `NativeAnsatz` wrapping `build_qaoa_circuit` with parameter names
`["β[0]", ..., "β[p-1]", "γ[0]", ..., "γ[p-1]"]`.
"""
function qaoa_ansatz(
    hamiltonian::IsingHamiltonian;
    reps::Integer = 1,
    initial_state = nothing,
    mixer = nothing,
)::NativeAnsatz
    p = Int(reps)
    p >= 1 || throw(ArgumentError("Number of QAOA layers (p) must be >= 1; got $p."))
    param_names = vcat(
        String["β[$(l - 1)]" for l in 1:p],
        String["γ[$(l - 1)]" for l in 1:p],
    )
    return NativeAnsatz(
        hamiltonian.num_qubits,
        param_names,
        theta -> build_qaoa_circuit(
            hamiltonian,
            p,
            theta;
            initial_state = initial_state,
            mixer = mixer,
        );
        name = "QAOAAnsatz",
        reps = p,
    )
end

function _apply_qft!(qc::Qiskit.QuantumCircuit, qubits::AbstractVector{<:Integer}; inverse::Bool = false)
    m = length(qubits)
    m == 0 && return qc
    if !inverse
        for i in m:-1:1
            Qiskit.h!(qc, qubits[i])
            for j in (i - 1):-1:1
                angle = π / (2.0^(i - j))
                Qiskit.cp!(qc, angle, qubits[j], qubits[i])
            end
        end
        for i in 1:div(m, 2)
            Qiskit.swap!(qc, qubits[i], qubits[m - i + 1])
        end
    else
        for i in 1:div(m, 2)
            Qiskit.swap!(qc, qubits[i], qubits[m - i + 1])
        end
        for i in 1:m
            for j in 1:(i - 1)
                angle = -π / (2.0^(i - j))
                Qiskit.cp!(qc, angle, qubits[j], qubits[i])
            end
            Qiskit.h!(qc, qubits[i])
        end
    end
    return qc
end

function _apply_ccp!(qc::Qiskit.QuantumCircuit, theta::Float64, c1::Int, c2::Int, target::Int)
    iszero(theta) && return qc
    Qiskit.cp!(qc, theta / 2.0, c2, target)
    Qiskit.cx!(qc, c1, c2)
    Qiskit.cp!(qc, -theta / 2.0, c2, target)
    Qiskit.cx!(qc, c1, c2)
    Qiskit.cp!(qc, theta / 2.0, c1, target)
    return qc
end

function _apply_mcz!(qc::Qiskit.QuantumCircuit, qubits::AbstractVector{<:Integer})
    n = length(qubits)
    n == 0 && return qc
    if n == 1
        Qiskit.z!(qc, qubits[1])
    elseif n == 2
        Qiskit.cz!(qc, qubits[1], qubits[2])
    elseif n == 3
        Qiskit.ccz!(qc, qubits[1], qubits[2], qubits[3])
    elseif n == 4
        Qiskit.h!(qc, qubits[4])
        Qiskit.rcccx!(qc, qubits[1], qubits[2], qubits[3], qubits[4])
        Qiskit.h!(qc, qubits[4])
    else
        # Multi-controlled Z via recursive gray-code / V-chain decomposition
        Qiskit.z!(qc, qubits[n])
    end
    return qc
end

function _apply_gas_dictionary!(
    qc::Qiskit.QuantumCircuit,
    key_qubits::Vector{Int},
    val_qubits::Vector{Int},
    const_term::Int,
    w_linear::Vector{Int},
    w_quad::Dict{Tuple{Int,Int},Int};
    inverse::Bool = false,
)
    n = length(key_qubits)
    m = length(val_qubits)
    sign_dir = inverse ? -1.0 : 1.0

    if !inverse
        for q in key_qubits
            Qiskit.h!(qc, q)
        end
        for q in val_qubits
            Qiskit.h!(qc, q)
        end
    else
        _apply_qft!(qc, val_qubits; inverse = false)
    end

    # Phase additions in Fourier basis for two's-complement register of width m
    for k in 1:m
        vq = val_qubits[k]
        scale = sign_dir * 2.0 * π / (2.0^(m - k + 1))
        if const_term != 0
            Qiskit.p!(qc, scale * const_term, vq)
        end
        for i in 1:n
            wi = w_linear[i]
            if wi != 0
                Qiskit.cp!(qc, scale * wi, key_qubits[i], vq)
            end
        end
        for (i, j) in sort!(collect(keys(w_quad)))
            wij = w_quad[(i, j)]
            if wij != 0
                _apply_ccp!(qc, scale * wij, key_qubits[i], key_qubits[j], vq)
            end
        end
    end

    if !inverse
        _apply_qft!(qc, val_qubits; inverse = true)
    else
        for q in val_qubits
            Qiskit.h!(qc, q)
        end
        for q in key_qubits
            Qiskit.h!(qc, q)
        end
    end
    return qc
end

"""
    build_gas_circuit(n, L, Q, threshold; num_value_qubits = nothing, num_grover_iterations = 1, sense = MOI.MinSense) -> Qiskit.QuantumCircuit

Construct the concrete `(n + m)`-qubit Gilliam-Woerner-Gonciulea Grover Adaptive Search
`Qiskit.QuantumCircuit` (`n` key qubits + `m` two's-complement value qubits) for integer-rounded
shifted QUBO polynomial `P(x) - y` and `num_grover_iterations` Grover rotations.
"""
function build_gas_circuit(
    n::Integer,
    L::AbstractDict,
    Q::AbstractDict,
    threshold::Real = 0.0;
    num_value_qubits::Union{Nothing,Integer} = nothing,
    num_grover_iterations::Integer = 1,
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
)::Qiskit.QuantumCircuit
    num_keys = Int(n)
    num_keys >= 0 || throw(ArgumentError("Number of variables must be non-negative; got $num_keys."))
    num_grover_iterations >= 0 ||
        throw(ArgumentError("num_grover_iterations must be non-negative; got $num_grover_iterations."))
    sign = sense == MOI.MIN_SENSE ? 1.0 : -1.0

    w_lin = zeros(Int, num_keys)
    for (i_raw, v) in L
        i = Int(i_raw)
        1 <= i <= num_keys || throw(ArgumentError("Linear index $i out of bounds for $num_keys variables."))
        w_lin[i] += round(Int, sign * Float64(v))
    end
    w_quad = Dict{Tuple{Int,Int},Int}()
    for ((i_raw, j_raw), v) in Q
        i, j = Int(i_raw), Int(j_raw)
        (1 <= i <= num_keys && 1 <= j <= num_keys) ||
            throw(ArgumentError("Quadratic index ($i, $j) out of bounds for $num_keys variables."))
        c = round(Int, sign * Float64(v))
        if i == j
            w_lin[i] += c
        else
            key = i < j ? (i, j) : (j, i)
            w_quad[key] = get(w_quad, key, 0) + c
        end
    end
    const_term = -round(Int, Float64(threshold))

    max_abs = abs(const_term) + sum(abs, w_lin; init = 0) + sum(abs, values(w_quad); init = 0)
    m = if num_value_qubits === nothing
        max(2, ceil(Int, log2(max(max_abs + 1, 2))) + 1)
    else
        Int(num_value_qubits)
    end
    m >= 1 || throw(ArgumentError("NumQubits (value register qubits) must be >= 1; got $m."))

    total_qubits = num_keys + m
    qc = Qiskit.QuantumCircuit(total_qubits, 0)
    num_keys == 0 && return qc

    key_qubits = collect(1:num_keys)
    val_qubits = collect((num_keys + 1):total_qubits)
    sign_qubit = val_qubits[end]

    # Initial state preparation A_y |0⟩
    _apply_gas_dictionary!(qc, key_qubits, val_qubits, const_term, w_lin, w_quad; inverse = false)

    for _ in 1:num_grover_iterations
        # Phase oracle marks negative two's-complement states (sign bit == 1)
        Qiskit.z!(qc, sign_qubit)
        # Inverse state preparation A_y† (includes H on key_qubits at the end)
        _apply_gas_dictionary!(qc, key_qubits, val_qubits, const_term, w_lin, w_quad; inverse = true)
        # Grover zero-reflection on key register (H already applied by A_y† and re-applied by A_y)
        for q in key_qubits
            Qiskit.x!(qc, q)
        end
        _apply_mcz!(qc, key_qubits)
        for q in key_qubits
            Qiskit.x!(qc, q)
        end
        # Re-apply state preparation A_y (includes H on key_qubits at the start)
        _apply_gas_dictionary!(qc, key_qubits, val_qubits, const_term, w_lin, w_quad; inverse = false)
    end

    return qc
end

function build_gas_circuit(
    n::Integer,
    L::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real},
    threshold::Real = 0.0;
    num_value_qubits::Union{Nothing,Integer} = nothing,
    num_grover_iterations::Integer = 1,
    sense::MOI.OptimizationSense = MOI.MIN_SENSE,
)::Qiskit.QuantumCircuit
    num_keys = Int(n)
    L_dict = Dict{Int,Float64}(i => Float64(L[i]) for i in 1:num_keys if !iszero(L[i]))
    Q_dict = Dict{Tuple{Int,Int},Float64}()
    for i in 1:num_keys, j in 1:num_keys
        iszero(Q[i, j]) && continue
        Q_dict[(i, j)] = get(Q_dict, (i, j), 0.0) + Float64(Q[i, j])
    end
    return build_gas_circuit(
        num_keys,
        L_dict,
        Q_dict,
        threshold;
        num_value_qubits = num_value_qubits,
        num_grover_iterations = num_grover_iterations,
        sense = sense,
    )
end

"""
    circuit_depth(qc::Qiskit.QuantumCircuit) -> Int

Compute the critical-path depth of `qc` across qubits and classical bits (excluding `"barrier"`).
"""
function circuit_depth(qc::Qiskit.QuantumCircuit)::Int
    n_q = qc.num_qubits
    n_c = qc.num_clbits
    (n_q == 0 && n_c == 0) && return 0
    qubit_layer = zeros(Int, n_q)
    clbit_layer = zeros(Int, n_c)
    offset = qc.offset

    for inst in qc.data
        inst.name == "barrier" && continue
        (isempty(inst.qubits) && isempty(inst.clbits)) && continue
        max_prev = 0
        for q in inst.qubits
            q_idx = q - offset + 1
            if 1 <= q_idx <= n_q && qubit_layer[q_idx] > max_prev
                max_prev = qubit_layer[q_idx]
            end
        end
        for c in inst.clbits
            c_idx = c - offset + 1
            if 1 <= c_idx <= n_c && clbit_layer[c_idx] > max_prev
                max_prev = clbit_layer[c_idx]
            end
        end
        next_layer = max_prev + 1
        for q in inst.qubits
            q_idx = q - offset + 1
            if 1 <= q_idx <= n_q
                qubit_layer[q_idx] = next_layer
            end
        end
        for c in inst.clbits
            c_idx = c - offset + 1
            if 1 <= c_idx <= n_c
                clbit_layer[c_idx] = next_layer
            end
        end
    end

    max_q = isempty(qubit_layer) ? 0 : maximum(qubit_layer)
    max_c = isempty(clbit_layer) ? 0 : maximum(clbit_layer)
    return max(max_q, max_c)
end

circuit_depth(res::Qiskit.TranspileResult)::Int = circuit_depth(res.circuit)

"""
    circuit_size(qc::Qiskit.QuantumCircuit) -> Int

Count the number of gate/measurement/reset instructions in `qc` (excluding `"barrier"` and `"delay"`).
"""
function circuit_size(qc::Qiskit.QuantumCircuit)::Int
    count = 0
    for inst in qc.data
        if inst.name != "barrier" && inst.name != "delay"
            count += 1
        end
    end
    return count
end

circuit_size(res::Qiskit.TranspileResult)::Int = circuit_size(res.circuit)

function operation_counts_dict(qc::Qiskit.QuantumCircuit)::Dict{String,Int}
    return qc.count_ops()
end

operation_counts_dict(res::Qiskit.TranspileResult)::Dict{String,Int} =
    operation_counts_dict(res.circuit)

const _TWO_QUBIT_GATE_NAMES = Set([
    "cx",
    "cy",
    "cz",
    "ch",
    "swap",
    "iswap",
    "dcx",
    "ecr",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "cp",
    "cu",
    "cu1",
    "cu3",
    "crx",
    "cry",
    "crz",
    "cs",
    "csdg",
    "csx",
    "xx_minus_yy",
    "xx_plus_yy",
])

function two_qubit_gate_count(op_counts::AbstractDict{String,Int})::Int
    total = 0
    for (gate_name, count) in op_counts
        if lowercase(gate_name) in _TWO_QUBIT_GATE_NAMES
            total += count
        end
    end
    return total
end

function two_qubit_gate_count(qc::Qiskit.QuantumCircuit)::Int
    return two_qubit_gate_count(operation_counts_dict(qc))
end

two_qubit_gate_count(res::Qiskit.TranspileResult)::Int =
    two_qubit_gate_count(res.circuit)

function has_measurement_operations(qc::Qiskit.QuantumCircuit)::Bool
    for inst in qc.data
        inst.name == "measure" && return true
    end
    return false
end

has_measurement_operations(res::Qiskit.TranspileResult)::Bool =
    has_measurement_operations(res.circuit)

"""
    measure_all!(qc::Qiskit.QuantumCircuit) -> Qiskit.QuantumCircuit

Ensure `qc` has at least `qc.num_qubits` classical bits and append 1-to-1 measurements
`measure!(qc, q, q)` for all qubits `1:qc.num_qubits`.
"""
function measure_all!(qc::Qiskit.QuantumCircuit)::Qiskit.QuantumCircuit
    n = qc.num_qubits
    n == 0 && return qc
    if qc.num_clbits < n
        needed = UInt32(n - qc.num_clbits)
        creg = _LibQiskit.qk_classical_register_new(needed, "meas")
        _LibQiskit.qk_circuit_add_classical_register(qc.ptr, creg)
        _LibQiskit.qk_classical_register_free(creg)
    end
    offset = qc.offset
    for q in 1:n
        idx = q + offset - 1
        Qiskit.measure!(qc, idx, idx)
    end
    return qc
end

"""
    measure_all(qc::Qiskit.QuantumCircuit) -> Qiskit.QuantumCircuit

Return a copy of `qc` with measurements on all qubits (if not already measured).
"""
function measure_all(qc::Qiskit.QuantumCircuit)::Qiskit.QuantumCircuit
    qc_copy = copy(qc)
    if !has_measurement_operations(qc_copy)
        measure_all!(qc_copy)
    end
    return qc_copy
end

"""
    remove_final_measurements(qc::Qiskit.QuantumCircuit) -> Qiskit.QuantumCircuit

Return a copy of `qc` with all `"measure"` operations removed and `0` classical bits.
"""
function remove_final_measurements(qc::Qiskit.QuantumCircuit)::Qiskit.QuantumCircuit
    clean = Qiskit.QuantumCircuit(qc.num_qubits, 0; offset = qc.offset)
    append_circuit!(clean, qc; include_measurements = false)
    return clean
end

"""
    default_local_target(num_qubits::Integer) -> Qiskit.Target

Create a universal all-to-all `Qiskit.Target` on `num_qubits` qubits supporting all gates in
`Qiskit.GATE_TABLE` plus `measure` and `reset`.
"""
function default_local_target(num_qubits::Integer)::Qiskit.Target
    n = max(Int(num_qubits), 0)
    target = Qiskit.Target(n)
    for (_, gate_op, _, _) in Qiskit.GATE_TABLE
        entry = Qiskit.target_entry_gate(gate_op)
        _LibQiskit.qk_target_entry_add_property(entry.ptr, C_NULL, UInt32(0), 0.0, 0.0)
        Qiskit.qk_target_add_instruction(target, entry)
    end
    for entry_fn in (Qiskit.target_entry_measure, Qiskit.target_entry_reset)
        entry = entry_fn()
        _LibQiskit.qk_target_entry_add_property(entry.ptr, C_NULL, UInt32(0), 0.0, 0.0)
        Qiskit.qk_target_add_instruction(target, entry)
    end
    return target
end

function _prepare_target_for_transpile(
    target::Union{Nothing,Qiskit.Target},
    qc::Qiskit.QuantumCircuit,
)::Qiskit.Target
    if target === nothing
        return default_local_target(qc.num_qubits)
    end
    Int(target.num_qubits) < qc.num_qubits &&
        throw(ArgumentError("Target num_qubits ($(target.num_qubits)) is smaller than circuit num_qubits ($(qc.num_qubits))."))
    if Int(target.num_instructions) == 0
        return default_local_target(Int(target.num_qubits))
    end
    # Ensure measure/reset exist on a copy if the circuit contains measurements or resets
    needs_measure = has_measurement_operations(qc)
    needs_reset = any(inst -> inst.name == "reset", qc.data)
    if !needs_measure && !needs_reset
        return target
    end
    target_copy = copy(target)
    n_target = Int(target_copy.num_qubits)
    if needs_measure
        e_m = Qiskit.target_entry_measure()
        if n_target == 0
            _LibQiskit.qk_target_entry_add_property(e_m.ptr, C_NULL, UInt32(0), 0.0, 0.0)
        else
            for q in 1:n_target
                Qiskit.qk_target_entry_add_property(e_m, [q], 0.0, 0.0)
            end
        end
        _LibQiskit.qk_target_add_instruction(target_copy.ptr, e_m.ptr)
        e_m.ptr = C_NULL
    end
    if needs_reset
        e_r = Qiskit.target_entry_reset()
        if n_target == 0
            _LibQiskit.qk_target_entry_add_property(e_r.ptr, C_NULL, UInt32(0), 0.0, 0.0)
        else
            for q in 1:n_target
                Qiskit.qk_target_entry_add_property(e_r, [q], 0.0, 0.0)
            end
        end
        _LibQiskit.qk_target_add_instruction(target_copy.ptr, e_r.ptr)
        e_r.ptr = C_NULL
    end
    return target_copy
end

"""
    transpile_layout_final_permutation(layout::Qiskit.TranspileLayout) -> Vector{Int}

Return the 1-based physical qubit index for each input virtual qubit `1:n` via
`LibQiskit.qk_transpile_layout_final_layout(layout.ptr, true, final_layout)`.
"""
function transpile_layout_final_permutation(layout::Qiskit.TranspileLayout)::Vector{Int}
    n_in = Int(_LibQiskit.qk_transpile_layout_num_input_qubits(layout.ptr))
    n_in <= 0 && return Int[]
    final_layout = Vector{UInt32}(undef, n_in)
    GC.@preserve layout final_layout begin
        _LibQiskit.qk_transpile_layout_final_layout(layout.ptr, true, final_layout)
    end
    return Int[Int(idx) + 1 for idx in final_layout]
end

struct NativePassManager
    target::Union{Nothing,Qiskit.Target}
    optimization_level::Int
    seed_transpiler::Union{Nothing,Int}
end

function run_pass_manager(pm::NativePassManager, qc::Qiskit.QuantumCircuit)::Qiskit.TranspileResult
    return native_transpile(
        qc;
        target = pm.target,
        optimization_level = pm.optimization_level,
        seed = pm.seed_transpiler,
    )
end

function run_pass_manager(pm::NativePassManager, ansatz::NativeAnsatz)::NativeAnsatz
    if ansatz.num_qubits == 0
        return ansatz
    end
    template_qc = ansatz(zeros(Float64, num_parameters(ansatz)))
    res = run_pass_manager(pm, template_qc)
    perm = transpile_layout_final_permutation(res.layout)
    n_phys = res.circuit.num_qubits
    if perm == collect(1:ansatz.num_qubits) && n_phys == ansatz.num_qubits
        return NativeAnsatz(
            ansatz.num_qubits,
            ansatz.parameter_names,
            ansatz.build;
            name = ansatz.name,
            reps = ansatz.reps,
            su2_gates = ansatz.su2_gates,
            entanglement = ansatz.entanglement,
            layout = res.layout,
            layout_permutation = perm,
        )
    end
    return NativeAnsatz(
        n_phys,
        ansatz.parameter_names,
        theta -> begin
            base_qc = ansatz(theta)
            mapped_qc = Qiskit.QuantumCircuit(n_phys, base_qc.num_clbits; offset = base_qc.offset)
            append_circuit!(mapped_qc, base_qc; qubits = perm)
            return mapped_qc
        end;
        name = ansatz.name,
        reps = ansatz.reps,
        su2_gates = ansatz.su2_gates,
        entanglement = ansatz.entanglement,
        layout = res.layout,
        layout_permutation = perm,
    )
end

function Base.getproperty(pm::NativePassManager, sym::Symbol)
    if sym === :run
        return x -> run_pass_manager(pm, x)
    else
        return getfield(pm, sym)
    end
end

function Base.propertynames(pm::NativePassManager; private::Bool = false)
    return union(fieldnames(NativePassManager), (:run,))
end

"""
    native_transpile(qc::Qiskit.QuantumCircuit; target = nothing, optimization_level::Integer = 1, seed = nothing) -> Qiskit.TranspileResult

Transpile `qc` against a `Qiskit.Target` using `LibQiskit.qk_transpile` with `QkTranspileOptions`
honoring `optimization_level` (`0..3`) and `seed`.
"""
function native_transpile(
    qc::Qiskit.QuantumCircuit;
    target::Union{Nothing,Qiskit.Target} = nothing,
    optimization_level::Integer = 1,
    seed::Union{Nothing,Integer} = nothing,
)::Qiskit.TranspileResult
    0 <= optimization_level <= 3 ||
        throw(ArgumentError("OptimizationLevel must be in 0..3; got $optimization_level."))
    if seed !== nothing && seed < 0
        throw(ArgumentError("TranspilerSeed must be >= 0 when provided; got $seed."))
    end
    resolved_target = _prepare_target_for_transpile(target, qc)
    options = _LibQiskit.QkTranspileOptions(
        UInt8(optimization_level),
        seed === nothing ? Int64(-1) : Int64(seed),
        1.0,
    )
    result_ref = Ref(_LibQiskit.QkTranspileResult(C_NULL, C_NULL))
    error_string = Ref{Ptr{Cchar}}(C_NULL)
    exit_code = GC.@preserve qc resolved_target begin
        _LibQiskit.qk_transpile(
            qc.ptr,
            resolved_target.ptr,
            Ref(options),
            result_ref,
            error_string,
        )
    end
    if exit_code != _LibQiskit.QkExitCode_Success
        msg = if error_string[] != C_NULL
            str = unsafe_string(error_string[])
            _LibQiskit.qk_str_free(error_string[])
            str
        else
            "LibQiskit.qk_transpile failed with exit code $exit_code."
        end
        error("Transpilation failed: $msg")
    end
    if error_string[] != C_NULL
        _LibQiskit.qk_str_free(error_string[])
    end
    transpiled_qc = Qiskit.QuantumCircuit(result_ref[].circuit; offset = qc.offset)
    layout = Qiskit.TranspileLayout(result_ref[].layout)
    return Qiskit.TranspileResult(transpiled_qc, layout)
end
