# Pure-Julia local statevector, MPS, and seeded shot simulator for Qiskit.QuantumCircuit

function _apply_1q_gate!(
    psi::Vector{Complex{T}},
    n::Int,
    q::Int,
    u00::Complex{T},
    u01::Complex{T},
    u10::Complex{T},
    u11::Complex{T},
) where {T<:AbstractFloat}
    step = 1 << (q - 1)
    block = step << 1
    total = 1 << n
    @inbounds for base in 0:block:(total - 1)
        for offset in 0:(step - 1)
            i0 = base + offset + 1
            i1 = i0 + step
            a0 = psi[i0]
            a1 = psi[i1]
            psi[i0] = u00 * a0 + u01 * a1
            psi[i1] = u10 * a0 + u11 * a1
        end
    end
    return psi
end

function _gate_1q_matrix(name::String, params::Vector{Float64}, ::Type{T}) where {T<:AbstractFloat}
    C = Complex{T}
    z = zero(C)
    o = one(C)
    im_c = C(0, 1)
    inv_sqrt2 = T(1 / sqrt(2.0))

    if name == "id"
        return (o, z, z, o)
    elseif name == "x"
        return (z, o, o, z)
    elseif name == "y"
        return (z, -im_c, im_c, z)
    elseif name == "z"
        return (o, z, z, -o)
    elseif name == "h"
        s = C(inv_sqrt2, 0)
        return (s, s, s, -s)
    elseif name == "s"
        return (o, z, z, im_c)
    elseif name == "sdg"
        return (o, z, z, -im_c)
    elseif name == "t"
        return (o, z, z, cis(T(π / 4)))
    elseif name == "tdg"
        return (o, z, z, cis(T(-π / 4)))
    elseif name == "sx"
        p = C(T(0.5), T(0.5))
        m = C(T(0.5), T(-0.5))
        return (p, m, m, p)
    elseif name == "sxdg"
        p = C(T(0.5), T(0.5))
        m = C(T(0.5), T(-0.5))
        return (m, p, p, m)
    elseif name == "p" || name == "u1"
        λ = T(params[1])
        return (o, z, z, cis(λ))
    elseif name == "rx"
        θ = T(params[1])
        c = C(cos(θ / 2), 0)
        s = C(0, -sin(θ / 2))
        return (c, s, s, c)
    elseif name == "ry"
        θ = T(params[1])
        c = C(cos(θ / 2), 0)
        s = C(sin(θ / 2), 0)
        return (c, -s, s, c)
    elseif name == "rz"
        θ = T(params[1])
        return (cis(-θ / 2), z, z, cis(θ / 2))
    elseif name == "r"
        θ = T(params[1])
        ϕ = T(params[2])
        c = C(cos(θ / 2), 0)
        s = T(sin(θ / 2))
        return (c, -im_c * cis(-ϕ) * s, -im_c * cis(ϕ) * s, c)
    elseif name == "u" || name == "u3"
        θ = T(params[1])
        ϕ = T(params[2])
        λ = T(params[3])
        c = cos(θ / 2)
        s = sin(θ / 2)
        return (C(c, 0), -cis(λ) * s, cis(ϕ) * s, cis(ϕ + λ) * c)
    elseif name == "u2"
        ϕ = T(params[1])
        λ = T(params[2])
        return (
            C(inv_sqrt2, 0),
            -cis(λ) * inv_sqrt2,
            cis(ϕ) * inv_sqrt2,
            cis(ϕ + λ) * inv_sqrt2,
        )
    else
        throw(ArgumentError("Unsupported 1-qubit gate '$name'."))
    end
end

function _apply_2q_matrix!(
    psi::Vector{Complex{T}},
    n::Int,
    q1::Int,
    q2::Int,
    M::AbstractMatrix{Complex{T}},
) where {T<:AbstractFloat}
    m1 = 1 << (q1 - 1)
    m2 = 1 << (q2 - 1)
    total = 1 << n
    @inbounds for idx0 in 0:(total - 1)
        if (idx0 & m1) == 0 && (idx0 & m2) == 0
            i00 = idx0 + 1
            i01 = (idx0 | m2) + 1
            i10 = (idx0 | m1) + 1
            i11 = (idx0 | m1 | m2) + 1
            a00 = psi[i00]
            a01 = psi[i01]
            a10 = psi[i10]
            a11 = psi[i11]
            psi[i00] = M[1, 1] * a00 + M[1, 2] * a01 + M[1, 3] * a10 + M[1, 4] * a11
            psi[i01] = M[2, 1] * a00 + M[2, 2] * a01 + M[2, 3] * a10 + M[2, 4] * a11
            psi[i10] = M[3, 1] * a00 + M[3, 2] * a01 + M[3, 3] * a10 + M[3, 4] * a11
            psi[i11] = M[4, 1] * a00 + M[4, 2] * a01 + M[4, 3] * a10 + M[4, 4] * a11
        end
    end
    return psi
end

function _gate_2q_matrix(name::String, params::Vector{Float64}, ::Type{T}) where {T<:AbstractFloat}
    C = Complex{T}
    z = zero(C)
    o = one(C)
    im_c = C(0, 1)
    inv_sqrt2 = T(1 / sqrt(2.0))

    # Basis order: |b1=0, b2=0⟩, |b1=0, b2=1⟩, |b1=1, b2=0⟩, |b1=1, b2=1⟩
    if name == "cx"
        return C[
            o z z z
            z o z z
            z z z o
            z z o z
        ]
    elseif name == "cy"
        return C[
            o z z z
            z o z z
            z z z -im_c
            z z im_c z
        ]
    elseif name == "cz"
        return C[
            o z z z
            z o z z
            z z o z
            z z z -o
        ]
    elseif name == "ch"
        s = C(inv_sqrt2, 0)
        return C[
            o z z z
            z o z z
            z z s s
            z z s -s
        ]
    elseif name == "swap"
        return C[
            o z z z
            z z o z
            z o z z
            z z z o
        ]
    elseif name == "iswap"
        return C[
            o z z z
            z z im_c z
            z im_c z z
            z z z o
        ]
    elseif name == "dcx"
        # cx(1, 2) followed by cx(2, 1)
        return C[
            o z z z
            z z o z
            z z z o
            z o z z
        ]
    elseif name == "ecr"
        s = C(inv_sqrt2, 0)
        s_im = C(0, inv_sqrt2)
        return C[
            z z s s_im
            z z s_im s
            s -s_im z z
            -s_im s z z
        ]
    elseif name == "cp" || name == "cu1"
        λ = T(params[1])
        return C[
            o z z z
            z o z z
            z z o z
            z z z cis(λ)
        ]
    elseif name == "crx"
        u00, u01, u10, u11 = _gate_1q_matrix("rx", params, T)
        return C[
            o z z z
            z o z z
            z z u00 u01
            z z u10 u11
        ]
    elseif name == "cry"
        u00, u01, u10, u11 = _gate_1q_matrix("ry", params, T)
        return C[
            o z z z
            z o z z
            z z u00 u01
            z z u10 u11
        ]
    elseif name == "crz"
        u00, u01, u10, u11 = _gate_1q_matrix("rz", params, T)
        return C[
            o z z z
            z o z z
            z z u00 u01
            z z u10 u11
        ]
    elseif name == "cs"
        return C[
            o z z z
            z o z z
            z z o z
            z z z im_c
        ]
    elseif name == "csdg"
        return C[
            o z z z
            z o z z
            z z o z
            z z z -im_c
        ]
    elseif name == "csx"
        u00, u01, u10, u11 = _gate_1q_matrix("sx", params, T)
        return C[
            o z z z
            z o z z
            z z u00 u01
            z z u10 u11
        ]
    elseif name == "cu"
        θ, ϕ, λ, γ = T(params[1]), T(params[2]), T(params[3]), T(params[4])
        ph = cis(γ)
        u00, u01, u10, u11 = _gate_1q_matrix("u", Float64[θ, ϕ, λ], T)
        return C[
            o z z z
            z o z z
            z z (ph * u00) (ph * u01)
            z z (ph * u10) (ph * u11)
        ]
    elseif name == "cu3"
        u00, u01, u10, u11 = _gate_1q_matrix("u3", params, T)
        return C[
            o z z z
            z o z z
            z z u00 u01
            z z u10 u11
        ]
    elseif name == "rxx"
        θ = T(params[1])
        c = C(cos(θ / 2), 0)
        s = C(0, -sin(θ / 2))
        return C[
            c z z s
            z c s z
            z s c z
            s z z c
        ]
    elseif name == "ryy"
        θ = T(params[1])
        c = C(cos(θ / 2), 0)
        s = C(0, sin(θ / 2))
        return C[
            c z z s
            z c -s z
            z -s c z
            s z z c
        ]
    elseif name == "rzz"
        θ = T(params[1])
        em = cis(-θ / 2)
        ep = cis(θ / 2)
        return C[
            em z z z
            z ep z z
            z z ep z
            z z z em
        ]
    elseif name == "rzx"
        θ = T(params[1])
        c = C(cos(θ / 2), 0)
        s = C(0, -sin(θ / 2))
        return C[
            c s z z
            s c z z
            z z c -s
            z z -s c
        ]
    elseif name == "xx_minus_yy"
        θ = T(params[1])
        β = length(params) >= 2 ? T(params[2]) : zero(T)
        c = C(cos(θ / 2), 0)
        s = T(sin(θ / 2))
        return C[
            c z z (-im_c * cis(-β) * s)
            z o z z
            z z o z
            (-im_c * cis(β) * s) z z c
        ]
    elseif name == "xx_plus_yy"
        θ = T(params[1])
        β = length(params) >= 2 ? T(params[2]) : zero(T)
        c = C(cos(θ / 2), 0)
        s = T(sin(θ / 2))
        return C[
            o z z z
            z c (-im_c * cis(-β) * s) z
            z (-im_c * cis(β) * s) c z
            z z z o
        ]
    else
        throw(ArgumentError("Unsupported 2-qubit gate '$name'."))
    end
end

function _apply_reset!(
    psi::Vector{Complex{T}},
    n::Int,
    q::Int,
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    mask = 1 << (q - 1)
    total = 1 << n
    p1 = zero(T)
    @inbounds for idx0 in 0:(total - 1)
        if (idx0 & mask) != 0
            p1 += abs2(psi[idx0 + 1])
        end
    end
    p0 = one(T) - p1
    if rand(rng, T) < p1 && p1 > eps(T)
        norm_inv = one(T) / sqrt(p1)
        @inbounds for idx0 in 0:(total - 1)
            if (idx0 & mask) == 0
                i0 = idx0 + 1
                i1 = (idx0 | mask) + 1
                psi[i0] = psi[i1] * norm_inv
                psi[i1] = zero(Complex{T})
            end
        end
    else
        norm_inv = p0 > eps(T) ? one(T) / sqrt(p0) : one(T)
        @inbounds for idx0 in 0:(total - 1)
            if (idx0 & mask) == 0
                psi[idx0 + 1] *= norm_inv
            else
                psi[idx0 + 1] = zero(Complex{T})
            end
        end
    end
    return psi
end

function _apply_instruction_dense!(
    psi::Vector{Complex{T}},
    n::Int,
    name::String,
    qubits::Vector{Int},
    params::Vector{Float64},
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    if name == "barrier" || name == "delay" || name == "measure"
        return psi
    elseif name == "reset"
        for q in qubits
            _apply_reset!(psi, n, q, rng)
        end
        return psi
    end

    nq = length(qubits)
    if nq == 1
        u00, u01, u10, u11 = _gate_1q_matrix(name, params, T)
        return _apply_1q_gate!(psi, n, qubits[1], u00, u01, u10, u11)
    elseif nq == 2
        q1, q2 = qubits[1], qubits[2]
        if name == "rzz"
            θ = T(params[1])
            em = cis(-θ / 2)
            ep = cis(θ / 2)
            m1 = 1 << (q1 - 1)
            m2 = 1 << (q2 - 1)
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                same = (((idx0 & m1) != 0) == ((idx0 & m2) != 0))
                psi[idx0 + 1] *= (same ? em : ep)
            end
            return psi
        elseif name == "cx"
            m1 = 1 << (q1 - 1)
            m2 = 1 << (q2 - 1)
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                if (idx0 & m1) != 0 && (idx0 & m2) == 0
                    i0 = idx0 + 1
                    i1 = (idx0 | m2) + 1
                    psi[i0], psi[i1] = psi[i1], psi[i0]
                end
            end
            return psi
        elseif name == "cz"
            m1 = 1 << (q1 - 1)
            m2 = 1 << (q2 - 1)
            both = m1 | m2
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                if (idx0 & both) == both
                    psi[idx0 + 1] = -psi[idx0 + 1]
                end
            end
            return psi
        end
        M = _gate_2q_matrix(name, params, T)
        return _apply_2q_matrix!(psi, n, q1, q2, M)
    elseif nq == 3
        q1, q2, q3 = qubits[1], qubits[2], qubits[3]
        if name == "ccx"
            m1 = 1 << (q1 - 1)
            m2 = 1 << (q2 - 1)
            m3 = 1 << (q3 - 1)
            c_mask = m1 | m2
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                if (idx0 & c_mask) == c_mask && (idx0 & m3) == 0
                    i0 = idx0 + 1
                    i1 = (idx0 | m3) + 1
                    psi[i0], psi[i1] = psi[i1], psi[i0]
                end
            end
            return psi
        elseif name == "ccz"
            m_all = (1 << (q1 - 1)) | (1 << (q2 - 1)) | (1 << (q3 - 1))
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                if (idx0 & m_all) == m_all
                    psi[idx0 + 1] = -psi[idx0 + 1]
                end
            end
            return psi
        elseif name == "cswap"
            m1 = 1 << (q1 - 1)
            m2 = 1 << (q2 - 1)
            m3 = 1 << (q3 - 1)
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                if (idx0 & m1) != 0 && (idx0 & m2) == 0 && (idx0 & m3) != 0
                    i01 = idx0 + 1
                    i10 = ((idx0 & ~m3) | m2) + 1
                    psi[i01], psi[i10] = psi[i10], psi[i01]
                end
            end
            return psi
        elseif name == "rccx"
            _apply_instruction_dense!(psi, n, "u", [q3], [π / 2, π / 4, -π], rng)
            _apply_instruction_dense!(psi, n, "cx", [q2, q3], Float64[], rng)
            _apply_instruction_dense!(psi, n, "p", [q3], [-π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q1, q3], Float64[], rng)
            _apply_instruction_dense!(psi, n, "p", [q3], [π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q2, q3], Float64[], rng)
            _apply_instruction_dense!(psi, n, "u", [q3], [π / 2, 0.0, 3π / 4], rng)
            return psi
        end
    elseif nq == 4 && (name == "rcccx" || name == "c3x")
        q1, q2, q3, q4 = qubits[1], qubits[2], qubits[3], qubits[4]
        if name == "c3x"
            c_mask = (1 << (q1 - 1)) | (1 << (q2 - 1)) | (1 << (q3 - 1))
            m4 = 1 << (q4 - 1)
            total = 1 << n
            @inbounds for idx0 in 0:(total - 1)
                if (idx0 & c_mask) == c_mask && (idx0 & m4) == 0
                    i0 = idx0 + 1
                    i1 = (idx0 | m4) + 1
                    psi[i0], psi[i1] = psi[i1], psi[i0]
                end
            end
            return psi
        else
            _apply_instruction_dense!(psi, n, "u", [q4], [π / 2, π / 4, -π], rng)
            _apply_instruction_dense!(psi, n, "cx", [q3, q4], Float64[], rng)
            _apply_instruction_dense!(psi, n, "u", [q4], [π / 2, 0.0, 3π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q1, q4], Float64[], rng)
            _apply_instruction_dense!(psi, n, "p", [q4], [π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q2, q4], Float64[], rng)
            _apply_instruction_dense!(psi, n, "p", [q4], [-π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q1, q4], Float64[], rng)
            _apply_instruction_dense!(psi, n, "p", [q4], [π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q2, q4], Float64[], rng)
            _apply_instruction_dense!(psi, n, "u", [q4], [π / 2, π / 4, 3π / 4], rng)
            _apply_instruction_dense!(psi, n, "cx", [q3, q4], Float64[], rng)
            _apply_instruction_dense!(psi, n, "u", [q4], [π / 2, 0.0, 3π / 4], rng)
            return psi
        end
    end

    throw(ArgumentError("Unsupported circuit instruction '$name' on $nq qubits."))
end

function _ensure_decomposed_unitaries(qc::Qiskit.QuantumCircuit)::Qiskit.QuantumCircuit
    has_unitary = any(inst -> inst.name == "unitary", qc.data)
    has_unitary || return qc
    return native_transpile(
        qc;
        target = default_local_target(qc.num_qubits),
        optimization_level = 0,
    ).circuit
end

function _resolve_float_type(precision::Union{Nothing,AbstractString})::DataType
    if precision === nothing || lowercase(String(precision)) == "double"
        return Float64
    elseif lowercase(String(precision)) == "single"
        return Float32
    else
        throw(ArgumentError("Unsupported AerPrecision: $(repr(precision)) (expected \"single\" or \"double\")."))
    end
end

"""
    simulate_statevector(qc::Qiskit.QuantumCircuit; precision = "double", seed = nothing) -> Vector{Complex}

Simulate `qc` from `|0...0⟩` and return the final `2^n` statevector.
"""
function simulate_statevector(
    qc::Qiskit.QuantumCircuit;
    precision::Union{Nothing,AbstractString} = "double",
    seed::Union{Nothing,Integer} = nothing,
)
    T = _resolve_float_type(precision)
    exec_qc = _ensure_decomposed_unitaries(qc)
    n = exec_qc.num_qubits
    n > 24 && throw(ArgumentError("Dense statevector simulation requires n <= 24; got $n. Use MPS simulation for larger circuits."))
    psi = zeros(Complex{T}, 1 << n)
    psi[1] = one(Complex{T})
    n == 0 && return psi

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
    offset = exec_qc.offset
    for inst in exec_qc.data
        qubits1 = Int[q - offset + 1 for q in inst.qubits]
        _apply_instruction_dense!(psi, n, inst.name, qubits1, inst.params, rng)
    end
    return psi
end

"""
    statevector_probabilities_dict(qc::Qiskit.QuantumCircuit; precision = "double") -> Dict{String,Float64}

Return Qiskit-style MSB-first bitstring (`c_n...c_1`) probabilities for `qc`.
"""
function statevector_probabilities_dict(
    qc::Qiskit.QuantumCircuit;
    precision::Union{Nothing,AbstractString} = "double",
)::Dict{String,Float64}
    psi = simulate_statevector(qc; precision = precision)
    n = qc.num_qubits
    out = Dict{String,Float64}()
    for idx0 in 0:(length(psi) - 1)
        p = Float64(abs2(psi[idx0 + 1]))
        p > 0.0 || continue
        key = string(idx0, base = 2, pad = max(1, n))
        out[key] = get(out, key, 0.0) + p
    end
    return out
end

# ==============================================================================
# Matrix Product State (MPS) Simulator for large circuits (e.g. 36 qubits)
# ==============================================================================

mutable struct MatrixProductState{T<:AbstractFloat}
    tensors::Vector{Array{Complex{T},3}}
    max_bond_dim::Int
    trunc_thresh::Float64
end

function MatrixProductState(
    num_qubits::Int;
    T::Type{F} = Float64,
    max_bond_dim::Union{Nothing,Integer} = nothing,
    trunc_thresh::Union{Nothing,Real} = nothing,
) where {F<:AbstractFloat}
    chi = max_bond_dim === nothing ? 16 : max(1, Int(max_bond_dim))
    thresh = trunc_thresh === nothing ? 1e-10 : Float64(trunc_thresh)
    tensors = Vector{Array{Complex{F},3}}(undef, num_qubits)
    for q in 1:num_qubits
        A = zeros(Complex{F}, 1, 2, 1)
        A[1, 1, 1] = one(Complex{F})
        tensors[q] = A
    end
    return MatrixProductState{F}(tensors, chi, thresh)
end

function mps_amplitude(
    mps::MatrixProductState{T},
    bits::AbstractVector{<:Integer},
)::Complex{T} where {T<:AbstractFloat}
    n = length(mps.tensors)
    n == 0 && return one(Complex{T})
    v = ones(Complex{T}, 1)
    for q in 1:n
        A = mps.tensors[q]
        s = Int(bits[q]) + 1
        dL, _, dR = size(A)
        v_next = zeros(Complex{T}, dR)
        @inbounds for r in 1:dR, l in 1:dL
            v_next[r] += v[l] * A[l, s, r]
        end
        v = v_next
    end
    return v[1]
end

function _mps_apply_1q!(
    mps::MatrixProductState{T},
    q::Int,
    u00::Complex{T},
    u01::Complex{T},
    u10::Complex{T},
    u11::Complex{T},
) where {T<:AbstractFloat}
    A = mps.tensors[q]
    dL, _, dR = size(A)
    @inbounds for r in 1:dR, l in 1:dL
        a0 = A[l, 1, r]
        a1 = A[l, 2, r]
        A[l, 1, r] = u00 * a0 + u01 * a1
        A[l, 2, r] = u10 * a0 + u11 * a1
    end
    return mps
end

function _mps_apply_adjacent_2q!(
    mps::MatrixProductState{T},
    q::Int,
    M::AbstractMatrix{Complex{T}},
) where {T<:AbstractFloat}
    A1 = mps.tensors[q]
    A2 = mps.tensors[q + 1]
    dL, _, chi = size(A1)
    _, _, dR = size(A2)

    # Contract A1 and A2 over internal bond chi -> theta(dL, 2, 2, dR)
    theta = zeros(Complex{T}, dL, 2, 2, dR)
    @inbounds for r in 1:dR, s2 in 1:2, s1 in 1:2, l in 1:dL
        acc = zero(Complex{T})
        for m in 1:chi
            acc += A1[l, s1, m] * A2[m, s2, r]
        end
        theta[l, s1, s2, r] = acc
    end

    # Apply 4x4 gate M in basis (00, 01, 10, 11) -> (s1=1..2, s2=1..2)
    mat = Matrix{Complex{T}}(undef, 2 * dL, 2 * dR)
    @inbounds for r in 1:dR, l in 1:dL
        t00 = theta[l, 1, 1, r]
        t01 = theta[l, 1, 2, r]
        t10 = theta[l, 2, 1, r]
        t11 = theta[l, 2, 2, r]

        u00 = M[1, 1] * t00 + M[1, 2] * t01 + M[1, 3] * t10 + M[1, 4] * t11
        u01 = M[2, 1] * t00 + M[2, 2] * t01 + M[2, 3] * t10 + M[2, 4] * t11
        u10 = M[3, 1] * t00 + M[3, 2] * t01 + M[3, 3] * t10 + M[3, 4] * t11
        u11 = M[4, 1] * t00 + M[4, 2] * t01 + M[4, 3] * t10 + M[4, 4] * t11

        mat[l, r] = u00
        mat[l, dR + r] = u01
        mat[dL + l, r] = u10
        mat[dL + l, dR + r] = u11
    end

    F = LinearAlgebra.svd(mat)
    S = F.S
    k = min(length(S), mps.max_bond_dim)
    while k > 1 && S[k] <= mps.trunc_thresh
        k -= 1
    end
    norm_factor = sqrt(sum(abs2, S) / max(sum(abs2, @view S[1:k]), eps(T)))

    new_A1 = Array{Complex{T},3}(undef, dL, 2, k)
    new_A2 = Array{Complex{T},3}(undef, k, 2, dR)
    @inbounds for m in 1:k
        for l in 1:dL
            new_A1[l, 1, m] = F.U[l, m]
            new_A1[l, 2, m] = F.U[dL + l, m]
        end
        sm = Complex{T}(S[m] * norm_factor)
        for r in 1:dR
            new_A2[m, 1, r] = sm * F.Vt[m, r]
            new_A2[m, 2, r] = sm * F.Vt[m, dR + r]
        end
    end

    mps.tensors[q] = new_A1
    mps.tensors[q + 1] = new_A2
    return mps
end

function _mps_apply_2q!(
    mps::MatrixProductState{T},
    q1::Int,
    q2::Int,
    M::AbstractMatrix{Complex{T}},
) where {T<:AbstractFloat}
    q1 == q2 && return mps
    if q1 > q2
        # Permute qubit 1 and qubit 2 in M: swap rows (2, 3) and cols (2, 3)
        perm = [1, 3, 2, 4]
        M_swapped = M[perm, perm]
        return _mps_apply_2q!(mps, q2, q1, M_swapped)
    end
    if q2 == q1 + 1
        return _mps_apply_adjacent_2q!(mps, q1, M)
    end
    swap_M = _gate_2q_matrix("swap", Float64[], T)
    for k in q1:(q2 - 2)
        _mps_apply_adjacent_2q!(mps, k, swap_M)
    end
    _mps_apply_adjacent_2q!(mps, q2 - 1, M)
    for k in (q2 - 2):-1:q1
        _mps_apply_adjacent_2q!(mps, k, swap_M)
    end
    return mps
end

function _mps_right_canonicalize!(mps::MatrixProductState{T}) where {T<:AbstractFloat}
    n = length(mps.tensors)
    for q in n:-1:2
        A = mps.tensors[q]
        dL, _, dR = size(A)
        mat = Matrix{Complex{T}}(undef, dL, 2 * dR)
        @inbounds for r in 1:dR, l in 1:dL
            mat[l, r] = A[l, 1, r]
            mat[l, dR + r] = A[l, 2, r]
        end
        # LQ decomposition via QR of adjoint
        F = LinearAlgebra.qr( adjoint(mat) )
        Q_thin = Matrix(F.Q)
        k = size(Q_thin, 2)
        Q_adj = adjoint(Q_thin)
        L_mat = adjoint(Matrix(F.R))

        new_A = Array{Complex{T},3}(undef, k, 2, dR)
        @inbounds for r in 1:dR, m in 1:k
            new_A[m, 1, r] = Q_adj[m, r]
            new_A[m, 2, r] = Q_adj[m, dR + r]
        end
        mps.tensors[q] = new_A

        A_prev = mps.tensors[q - 1]
        dL_prev, _, _ = size(A_prev)
        new_prev = zeros(Complex{T}, dL_prev, 2, k)
        @inbounds for m in 1:k, s in 1:2, l in 1:dL_prev
            acc = zero(Complex{T})
            for c in 1:dL
                acc += A_prev[l, s, c] * L_mat[c, m]
            end
            new_prev[l, s, m] = acc
        end
        mps.tensors[q - 1] = new_prev
    end
    return mps
end

function _mps_apply_gate_seq!(mps::MatrixProductState{T}, seq) where {T<:AbstractFloat}
    for (gname, gq, gp) in seq
        if length(gq) == 1
            u00, u01, u10, u11 = _gate_1q_matrix(gname, gp, T)
            _mps_apply_1q!(mps, gq[1], u00, u01, u10, u11)
        else
            M = _gate_2q_matrix(gname, gp, T)
            _mps_apply_2q!(mps, gq[1], gq[2], M)
        end
    end
    return mps
end

function _ccx_1q2q_seq(q1::Int, q2::Int, q3::Int)
    return (
        ("h", [q3], Float64[]),
        ("cx", [q2, q3], Float64[]),
        ("tdg", [q3], Float64[]),
        ("cx", [q1, q3], Float64[]),
        ("t", [q3], Float64[]),
        ("cx", [q2, q3], Float64[]),
        ("tdg", [q3], Float64[]),
        ("cx", [q1, q3], Float64[]),
        ("t", [q2], Float64[]),
        ("t", [q3], Float64[]),
        ("h", [q3], Float64[]),
        ("cx", [q1, q2], Float64[]),
        ("t", [q1], Float64[]),
        ("tdg", [q2], Float64[]),
        ("cx", [q1, q2], Float64[]),
    )
end

function _mps_reset_1q!(
    mps::MatrixProductState{T},
    q::Int,
    rng::AbstractRNG,
) where {T<:AbstractFloat}
    _mps_right_canonicalize!(mps)
    n = length(mps.tensors)
    for k in 1:(q - 1)
        A = mps.tensors[k]
        dL, _, dR = size(A)
        mat = reshape(A, 2 * dL, dR)
        F = qr(mat)
        Q_mat = Matrix(F.Q)
        R_mat = Matrix(F.R)
        new_bond = size(Q_mat, 2)
        mps.tensors[k] = reshape(Q_mat, dL, 2, new_bond)
        B = mps.tensors[k + 1]
        _, _, dR2 = size(B)
        B_mat = R_mat * reshape(B, dR, 2 * dR2)
        mps.tensors[k + 1] = reshape(B_mat, new_bond, 2, dR2)
    end
    Aq = mps.tensors[q]
    p0 = sum(abs2, @view Aq[:, 1, :])
    p1 = sum(abs2, @view Aq[:, 2, :])
    tot = max(p0 + p1, eps(T))
    if rand(rng, T) * tot < p1
        # Measured |1⟩ -> project onto |1⟩ and flip to |0⟩
        scale = inv(sqrt(max(p1, eps(T))))
        Aq[:, 1, :] .= Aq[:, 2, :] .* scale
        Aq[:, 2, :] .= zero(Complex{T})
    else
        # Measured |0⟩ -> project onto |0⟩
        scale = inv(sqrt(max(p0, eps(T))))
        Aq[:, 1, :] .*= scale
        Aq[:, 2, :] .= zero(Complex{T})
    end
    return mps
end

function simulate_mps(
    qc::Qiskit.QuantumCircuit;
    precision::Union{Nothing,AbstractString} = "double",
    max_bond_dimension::Union{Nothing,Integer} = nothing,
    truncation_threshold::Union{Nothing,Real} = nothing,
    seed::Union{Nothing,Integer} = nothing,
)
    T = _resolve_float_type(precision)
    exec_qc = _ensure_decomposed_unitaries(qc)
    n = exec_qc.num_qubits
    mps = MatrixProductState(
        n;
        T = T,
        max_bond_dim = max_bond_dimension,
        trunc_thresh = truncation_threshold,
    )
    n == 0 && return mps
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
    offset = exec_qc.offset

    for inst in exec_qc.data
        name = inst.name
        (name == "barrier" || name == "delay" || name == "measure") && continue
        qubits1 = Int[q - offset + 1 for q in inst.qubits]
        nq = length(qubits1)
        if name == "reset" && nq == 1
            _mps_reset_1q!(mps, qubits1[1], rng)
        elseif nq == 1
            u00, u01, u10, u11 = _gate_1q_matrix(name, inst.params, T)
            _mps_apply_1q!(mps, qubits1[1], u00, u01, u10, u11)
        elseif nq == 2
            M = _gate_2q_matrix(name, inst.params, T)
            _mps_apply_2q!(mps, qubits1[1], qubits1[2], M)
        elseif nq == 3
            q1, q2, q3 = qubits1[1], qubits1[2], qubits1[3]
            if name == "ccx"
                _mps_apply_gate_seq!(mps, _ccx_1q2q_seq(q1, q2, q3))
            elseif name == "ccz"
                _mps_apply_gate_seq!(mps, (("h", [q3], Float64[]),))
                _mps_apply_gate_seq!(mps, _ccx_1q2q_seq(q1, q2, q3))
                _mps_apply_gate_seq!(mps, (("h", [q3], Float64[]),))
            elseif name == "cswap"
                _mps_apply_gate_seq!(mps, (("cx", [q3, q2], Float64[]),))
                _mps_apply_gate_seq!(mps, _ccx_1q2q_seq(q1, q2, q3))
                _mps_apply_gate_seq!(mps, (("cx", [q3, q2], Float64[]),))
            elseif name == "rccx"
                _mps_apply_gate_seq!(
                    mps,
                    (
                        ("u", [q3], [π / 2, π / 4, -π]),
                        ("cx", [q2, q3], Float64[]),
                        ("p", [q3], [-π / 4]),
                        ("cx", [q1, q3], Float64[]),
                        ("p", [q3], [π / 4]),
                        ("cx", [q2, q3], Float64[]),
                        ("u", [q3], [π / 2, 0.0, 3π / 4]),
                    ),
                )
            else
                throw(ArgumentError("Unsupported 3-qubit circuit instruction '$name' in MPS simulator."))
            end
        elseif nq == 4 && (name == "rcccx" || name == "c3x")
            q1, q2, q3, q4 = qubits1[1], qubits1[2], qubits1[3], qubits1[4]
            if name == "rcccx"
                _mps_apply_gate_seq!(
                    mps,
                    (
                        ("u", [q4], [π / 2, π / 4, -π]),
                        ("cx", [q3, q4], Float64[]),
                        ("u", [q4], [π / 2, 0.0, 3π / 4]),
                        ("cx", [q1, q4], Float64[]),
                        ("p", [q4], [π / 4]),
                        ("cx", [q2, q4], Float64[]),
                        ("p", [q4], [-π / 4]),
                        ("cx", [q1, q4], Float64[]),
                        ("p", [q4], [π / 4]),
                        ("cx", [q2, q4], Float64[]),
                        ("u", [q4], [π / 2, π / 4, 3π / 4]),
                        ("cx", [q3, q4], Float64[]),
                        ("u", [q4], [π / 2, 0.0, 3π / 4]),
                    ),
                )
            else
                # Exact 3-controlled X (c3x)
                _mps_apply_gate_seq!(mps, (("h", [q4], Float64[]), ("cp", [q3, q4], [π / 2])))
                _mps_apply_gate_seq!(mps, _ccx_1q2q_seq(q1, q2, q3))
                _mps_apply_gate_seq!(mps, (("cp", [q3, q4], [-π / 2]),))
                _mps_apply_gate_seq!(mps, _ccx_1q2q_seq(q1, q2, q3))
                _mps_apply_gate_seq!(
                    mps,
                    (
                        ("cp", [q2, q4], [π / 4]),
                        ("cx", [q1, q2], Float64[]),
                        ("cp", [q2, q4], [-π / 4]),
                        ("cx", [q1, q2], Float64[]),
                        ("cp", [q1, q4], [π / 4]),
                        ("h", [q4], Float64[]),
                    ),
                )
            end
        else
            throw(ArgumentError("Unsupported circuit instruction '$name' on $nq qubits in MPS simulator."))
        end
    end
    _mps_right_canonicalize!(mps)
    return mps
end

function _mps_sample_single_bitvector(
    mps::MatrixProductState{T},
    rng::AbstractRNG,
)::BitVector where {T<:AbstractFloat}
    n = length(mps.tensors)
    bits = falses(n)
    n == 0 && return bits
    vL = ones(Complex{T}, 1)
    for q in 1:n
        A = mps.tensors[q]
        dL, _, dR = size(A)
        v0 = zeros(Complex{T}, dR)
        v1 = zeros(Complex{T}, dR)
        @inbounds for r in 1:dR, l in 1:dL
            v0[r] += vL[l] * A[l, 1, r]
            v1[r] += vL[l] * A[l, 2, r]
        end
        p0 = sum(abs2, v0)
        p1 = sum(abs2, v1)
        tot = max(p0 + p1, eps(T))
        if rand(rng, T) * tot < p1
            bits[q] = true
            vL = v1 ./ sqrt(max(p1, eps(T)))
        else
            bits[q] = false
            vL = v0 ./ sqrt(max(p0, eps(T)))
        end
    end
    return bits
end

function _mps_pauli_term_expectation(
    mps::MatrixProductState{T},
    term::PauliTerm,
)::Float64 where {T<:AbstractFloat}
    n = length(mps.tensors)
    pauli_at = Dict{Int,Symbol}(q => p for (q, p) in zip(term.qubits, term.paulis))
    E = ones(Complex{T}, 1, 1)
    for q in 1:n
        A = mps.tensors[q]
        dL, _, dR = size(A)
        p = get(pauli_at, q, :I)
        next_E = zeros(Complex{T}, dR, dR)
        @inbounds for r1 in 1:dR, r2 in 1:dR
            acc = zero(Complex{T})
            for l1 in 1:dL, l2 in 1:dL
                e_val = E[l1, l2]
                iszero(e_val) && continue
                a0_conj = conj(A[l1, 1, r1])
                a1_conj = conj(A[l1, 2, r1])
                b0 = A[l2, 1, r2]
                b1 = A[l2, 2, r2]
                if p === :I
                    acc += e_val * (a0_conj * b0 + a1_conj * b1)
                elseif p === :Z
                    acc += e_val * (a0_conj * b0 - a1_conj * b1)
                elseif p === :X
                    acc += e_val * (a0_conj * b1 + a1_conj * b0)
                elseif p === :Y
                    acc += e_val * (Complex{T}(0, -1) * a0_conj * b1 + Complex{T}(0, 1) * a1_conj * b0)
                end
            end
            next_E[r1, r2] = acc
        end
        E = next_E
    end
    return term.coeff * Float64(real(E[1, 1]))
end

# ==============================================================================
# Shot Sampling & Expectation Evaluation
# ==============================================================================

function _should_use_mps(
    num_qubits::Int;
    max_bond_dimension::Union{Nothing,Integer} = nothing,
)::Bool
    return num_qubits > 18 || max_bond_dimension !== nothing
end

function _measurement_clbit_map(qc::Qiskit.QuantumCircuit)::Tuple{Int,Vector{Tuple{Int,Int}}}
    pairs = Tuple{Int,Int}[]
    offset = qc.offset
    for inst in qc.data
        inst.name == "measure" || continue
        for (q_raw, c_raw) in zip(inst.qubits, inst.clbits)
            push!(pairs, (q_raw - offset + 1, c_raw - offset + 1))
        end
    end
    if isempty(pairs)
        n = qc.num_qubits
        return (n, Tuple{Int,Int}[(q, q) for q in 1:n])
    end
    n_clbits = max(qc.num_clbits, maximum(last, pairs))
    return (n_clbits, pairs)
end

"""
    sample_bitvectors_from_circuit(qc::Qiskit.QuantumCircuit; shots::Integer = 1024, seed = nothing, precision = "double", max_bond_dimension = nothing, truncation_threshold = nothing) -> Vector{BitVector}

Sample `shots` LSB-first `BitVector` outcomes (`shot[c]` is classical bit `c`, 1-based) from `qc`.
"""
function sample_bitvectors_from_circuit(
    qc::Qiskit.QuantumCircuit;
    shots::Integer = 1024,
    seed::Union{Nothing,Integer} = nothing,
    precision::Union{Nothing,AbstractString} = "double",
    max_bond_dimension::Union{Nothing,Integer} = nothing,
    truncation_threshold::Union{Nothing,Real} = nothing,
)::Vector{BitVector}
    shots >= 1 || throw(ArgumentError("shots must be >= 1; got $shots."))
    n_clbits, meas_pairs = _measurement_clbit_map(qc)
    if qc.num_qubits == 0 || n_clbits == 0
        return BitVector[falses(n_clbits) for _ in 1:shots]
    end
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))

    if _should_use_mps(qc.num_qubits; max_bond_dimension = max_bond_dimension)
        mps = simulate_mps(
            qc;
            precision = precision,
            max_bond_dimension = max_bond_dimension,
            truncation_threshold = truncation_threshold,
        )
        results = Vector{BitVector}(undef, shots)
        for s in 1:shots
            q_bits = _mps_sample_single_bitvector(mps, rng)
            c_bits = falses(n_clbits)
            for (q, c) in meas_pairs
                c_bits[c] = q_bits[q]
            end
            results[s] = c_bits
        end
        return results
    end

    psi = simulate_statevector(qc; precision = precision, seed = seed)
    dim = length(psi)
    cdf = Vector{Float64}(undef, dim)
    acc = 0.0
    @inbounds for idx in 1:dim
        acc += Float64(abs2(psi[idx]))
        cdf[idx] = acc
    end
    if acc > 0.0
        cdf[end] = 1.0
    end

    results = Vector{BitVector}(undef, shots)
    for s in 1:shots
        u = rand(rng, Float64)
        state_idx0 = searchsortedfirst(cdf, u) - 1
        state_idx0 = clamp(state_idx0, 0, dim - 1)
        c_bits = falses(n_clbits)
        for (q, c) in meas_pairs
            c_bits[c] = ((state_idx0 >> (q - 1)) & 1) == 1
        end
        results[s] = c_bits
    end
    return results
end

function bitvectors_to_counts_dict(shots_vec::AbstractVector{BitVector})::Dict{String,Int}
    counts = Dict{String,Int}()
    for shot in shots_vec
        # Standard Qiskit count key order: highest classical bit index to 0
        key = String([shot[c] ? '1' : '0' for c in length(shot):-1:1])
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

const bitvectors_to_counts = bitvectors_to_counts_dict

"""
    sample_from_circuit(qc::Qiskit.QuantumCircuit; shots::Integer = 1024, seed = nothing, precision = "double", max_bond_dimension = nothing, truncation_threshold = nothing) -> Dict{String,Int}

Simulate `shots` measurements of `qc` and return a Qiskit-formatted count dictionary `Dict{String,Int}`.
"""
function sample_from_circuit(
    qc::Qiskit.QuantumCircuit;
    shots::Integer = 1024,
    seed::Union{Nothing,Integer} = nothing,
    precision::Union{Nothing,AbstractString} = "double",
    max_bond_dimension::Union{Nothing,Integer} = nothing,
    truncation_threshold::Union{Nothing,Real} = nothing,
)::Dict{String,Int}
    bitvecs = sample_bitvectors_from_circuit(
        qc;
        shots = shots,
        seed = seed,
        precision = precision,
        max_bond_dimension = max_bond_dimension,
        truncation_threshold = truncation_threshold,
    )
    return bitvectors_to_counts_dict(bitvecs)
end

function _statevector_pauli_term_expectation(
    psi::Vector{Complex{T}},
    term::PauliTerm,
)::Float64 where {T<:AbstractFloat}
    isempty(term.qubits) && return term.coeff
    dim = length(psi)
    x_mask = 0
    z_mask = 0
    y_count = 0
    for (q, p) in zip(term.qubits, term.paulis)
        bit = 1 << (q - 1)
        if p === :X
            x_mask |= bit
        elseif p === :Z
            z_mask |= bit
        elseif p === :Y
            x_mask |= bit
            z_mask |= bit
            y_count += 1
        end
    end

    # P |x⟩ = (i)^y_count * (-1)^{popcount(x & z_mask)} |x ⊻ x_mask⟩
    phase_y = (y_count & 3) == 0 ? Complex{T}(1, 0) :
              (y_count & 3) == 1 ? Complex{T}(0, 1) :
              (y_count & 3) == 2 ? Complex{T}(-1, 0) : Complex{T}(0, -1)

    acc = zero(Complex{T})
    @inbounds for x in 0:(dim - 1)
        y = xor(x, x_mask)
        sign_z = isodd(count_ones(x & z_mask)) ? -one(T) : one(T)
        acc += conj(psi[y + 1]) * (sign_z * phase_y * psi[x + 1])
    end
    return term.coeff * Float64(real(acc))
end

"""
    estimate_expectation(qc::Qiskit.QuantumCircuit, h::IsingHamiltonian; shots = nothing, seed = nothing, include_offset::Bool = false, precision = "double", max_bond_dimension = nothing, truncation_threshold = nothing) -> Float64

Evaluate `<ψ|H|ψ>` exactly (when `shots === nothing` or `shots <= 0`) or with finite-shot
sampling (`shots > 0`).
"""
function estimate_expectation(
    qc::Qiskit.QuantumCircuit,
    h::IsingHamiltonian;
    shots::Union{Nothing,Integer} = nothing,
    seed::Union{Nothing,Integer} = nothing,
    include_offset::Bool = false,
    precision::Union{Nothing,AbstractString} = "double",
    max_bond_dimension::Union{Nothing,Integer} = nothing,
    truncation_threshold::Union{Nothing,Real} = nothing,
)::Float64
    base_val = include_offset ? h.offset : 0.0
    (qc.num_qubits == 0 || isempty(h.terms)) && return base_val

    if _should_use_mps(qc.num_qubits; max_bond_dimension = max_bond_dimension)
        mps = simulate_mps(
            qc;
            precision = precision,
            max_bond_dimension = max_bond_dimension,
            truncation_threshold = truncation_threshold,
            seed = seed,
        )
        if shots !== nothing && shots > 0 && is_diagonal_ising(h)
            rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
            acc_shots = 0.0
            for _ in 1:shots
                bv = _mps_sample_single_bitvector(mps, rng)
                acc_shots += evaluate_ising_bitstring(h, bv; include_offset = false)
            end
            return base_val + acc_shots / shots
        end
        rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
        total = base_val
        for term in h.terms
            exact_term = _mps_pauli_term_expectation(mps, term)
            if shots === nothing || shots <= 0
                total += exact_term
            else
                mu = clamp(exact_term / term.coeff, -1.0, 1.0)
                p_plus = 0.5 * (1.0 + mu)
                n_plus = 0
                for _ in 1:shots
                    rand(rng, Float64) < p_plus && (n_plus += 1)
                end
                est_mu = (2.0 * n_plus - shots) / shots
                total += term.coeff * est_mu
            end
        end
        return total
    end

    psi = simulate_statevector(qc; precision = precision, seed = seed)
    dim = length(psi)

    if is_diagonal_ising(h)
        if shots === nothing || shots <= 0
            acc = 0.0
            @inbounds for x in 0:(dim - 1)
                prob = Float64(abs2(psi[x + 1]))
                prob == 0.0 && continue
                e_x = 0.0
                for term in h.terms
                    z_prod = 1.0
                    for q in term.qubits
                        z_prod *= (((x >> (q - 1)) & 1) == 1 ? -1.0 : 1.0)
                    end
                    e_x += term.coeff * z_prod
                end
                acc += prob * e_x
            end
            return base_val + acc
        else
            rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
            cdf = Vector{Float64}(undef, dim)
            c_acc = 0.0
            @inbounds for idx in 1:dim
                c_acc += Float64(abs2(psi[idx]))
                cdf[idx] = c_acc
            end
            c_acc > 0.0 && (cdf[end] = 1.0)
            shot_sum = 0.0
            for _ in 1:shots
                u = rand(rng, Float64)
                x = clamp(searchsortedfirst(cdf, u) - 1, 0, dim - 1)
                e_x = 0.0
                for term in h.terms
                    z_prod = 1.0
                    for q in term.qubits
                        z_prod *= (((x >> (q - 1)) & 1) == 1 ? -1.0 : 1.0)
                    end
                    e_x += term.coeff * z_prod
                end
                shot_sum += e_x
            end
            return base_val + shot_sum / shots
        end
    end

    # General non-diagonal Pauli Hamiltonian
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed))
    total = base_val
    for term in h.terms
        exact_term = _statevector_pauli_term_expectation(psi, term)
        if shots === nothing || shots <= 0
            total += exact_term
        else
            mu = clamp(exact_term / term.coeff, -1.0, 1.0)
            p_plus = 0.5 * (1.0 + mu)
            n_plus = 0
            for _ in 1:shots
                rand(rng, Float64) < p_plus && (n_plus += 1)
            end
            est_mu = (2.0 * n_plus - shots) / shots
            total += term.coeff * est_mu
        end
    end
    return total
end

function _parse_qiskit_pauli_string(pstr::AbstractString)::Tuple{Int,PauliTerm}
    m = length(pstr)
    qubits = Int[]
    paulis = Symbol[]
    chars = collect(pstr)
    for (pos, ch) in enumerate(chars)
        q = m - pos + 1
        if ch == 'I'
            continue
        elseif ch == 'X' || ch == 'Y' || ch == 'Z'
            push!(qubits, q)
            push!(paulis, Symbol(ch))
        else
            throw(ArgumentError("Invalid character '$ch' in Pauli string \"$pstr\"."))
        end
    end
    return (m, PauliTerm(1.0, qubits, paulis))
end

"""
    build_pce_basis_circuits(base_qc::Qiskit.QuantumCircuit, pauli_strings::AbstractVector{<:AbstractString}) -> Dict{Char,Qiskit.QuantumCircuit}

Construct up to 3 basis-rotated measured circuits (`'X'`, `'Y'`, `'Z'`) for active Pauli families in `pauli_strings`:
- `'X'` family: `h!` on all qubits followed by `measure_all!`
- `'Y'` family: `sdg!`, `h!` (or `rx!(π/2)`) on all qubits followed by `measure_all!`
- `'Z'` family: `measure_all!` directly
"""
function build_pce_basis_circuits(
    base_qc::Qiskit.QuantumCircuit,
    pauli_strings::AbstractVector{<:AbstractString},
)::Dict{Char,Qiskit.QuantumCircuit}
    clean = remove_final_measurements(base_qc)
    m = clean.num_qubits
    circuits = Dict{Char,Qiskit.QuantumCircuit}()
    for basis in ('X', 'Y', 'Z')
        any(s -> occursin(basis, s), pauli_strings) || continue
        qc_b = copy(clean)
        if basis == 'X'
            for q in 1:m
                Qiskit.h!(qc_b, q)
            end
        elseif basis == 'Y'
            for q in 1:m
                Qiskit.sdg!(qc_b, q)
                Qiskit.h!(qc_b, q)
            end
        end
        measure_all!(qc_b)
        circuits[basis] = qc_b
    end
    return circuits
end

function _resolve_basis_permutation(layout_permutation, basis::Char)
    if layout_permutation === nothing
        return nothing
    elseif layout_permutation isa AbstractDict
        return get(layout_permutation, basis, nothing)
    else
        return layout_permutation
    end
end

"""
    decode_pce_expectations_from_basis_counts(basis_counts::AbstractDict{Char,<:Any}, pauli_strings::AbstractVector{<:AbstractString}; layout_permutation = nothing) -> Vector{Float64}

Compute empirical Pauli correlator expectations `<Π_i> ∈ [-1, 1]` from measurement counts
or bitvectors in each active `'X'`, `'Y'`, `'Z'` basis.
"""
function decode_pce_expectations_from_basis_counts(
    basis_counts::AbstractDict{Char,<:Any},
    pauli_strings::AbstractVector{<:AbstractString};
    layout_permutation = nothing,
)::Vector{Float64}
    N = length(pauli_strings)
    N == 0 && return Float64[]
    evs = zeros(Float64, N)

    parsed_terms = Vector{Tuple{Char,Vector{Int}}}(undef, N)
    for (idx, pstr) in enumerate(pauli_strings)
        _, pt = _parse_qiskit_pauli_string(pstr)
        basis = isempty(pt.paulis) ? 'Z' : first(String(pt.paulis[1]))
        parsed_terms[idx] = (basis, pt.qubits)
    end

    for (idx, (basis, support_qubits)) in enumerate(parsed_terms)
        isempty(support_qubits) && (evs[idx] = 1.0; continue)
        raw_data = basis_counts[basis]
        perm_b = _resolve_basis_permutation(layout_permutation, basis)
        if raw_data isa AbstractVector{BitVector}
            total_shots = length(raw_data)
            total_shots == 0 && continue
            parity_sum = 0.0
            for shot in raw_data
                odd = false
                for q in support_qubits
                    bit_idx = perm_b === nothing ? q : Int(perm_b[q])
                    if shot[bit_idx]
                        odd = !odd
                    end
                end
                parity_sum += (odd ? -1.0 : 1.0)
            end
            evs[idx] = parity_sum / total_shots
        else
            total_shots = 0
            parity_sum = 0.0
            for (key, count_val) in pairs(raw_data)
                reads = Int(count_val)
                reads <= 0 && continue
                bits = sample_bits(String(key))
                odd = false
                for q in support_qubits
                    bit_idx = perm_b === nothing ? q : Int(perm_b[q])
                    if bits[bit_idx] == 1
                        odd = !odd
                    end
                end
                parity_sum += reads * (odd ? -1.0 : 1.0)
                total_shots += reads
            end
            evs[idx] = total_shots > 0 ? parity_sum / total_shots : 0.0
        end
    end
    return evs
end

"""
    estimate_pauli_expectations(qc::Qiskit.QuantumCircuit, pauli_strings::AbstractVector{<:AbstractString}; shots = nothing, seed = nothing, precision = "double", max_bond_dimension = nothing, truncation_threshold = nothing) -> Vector{Float64}

Evaluate `<ψ|Π_i|ψ>` for a vector of Qiskit-ordered Pauli strings `pauli_strings`.
"""
function estimate_pauli_expectations(
    qc::Qiskit.QuantumCircuit,
    pauli_strings::AbstractVector{<:AbstractString};
    shots::Union{Nothing,Integer} = nothing,
    seed::Union{Nothing,Integer} = nothing,
    precision::Union{Nothing,AbstractString} = "double",
    max_bond_dimension::Union{Nothing,Integer} = nothing,
    truncation_threshold::Union{Nothing,Real} = nothing,
)::Vector{Float64}
    N = length(pauli_strings)
    N == 0 && return Float64[]
    if shots !== nothing && shots > 0
        basis_circuits = build_pce_basis_circuits(qc, pauli_strings)
        basis_counts = Dict{Char,Vector{BitVector}}()
        for (b_idx, basis) in enumerate(sort!(collect(keys(basis_circuits))))
            b_seed = seed === nothing ? nothing : Int(seed) + 1009 * b_idx
            basis_counts[basis] = sample_bitvectors_from_circuit(
                basis_circuits[basis];
                shots = shots,
                seed = b_seed,
                precision = precision,
                max_bond_dimension = max_bond_dimension,
                truncation_threshold = truncation_threshold,
            )
        end
        return decode_pce_expectations_from_basis_counts(basis_counts, pauli_strings)
    end

    if _should_use_mps(qc.num_qubits; max_bond_dimension = max_bond_dimension)
        mps = simulate_mps(
            qc;
            precision = precision,
            max_bond_dimension = max_bond_dimension,
            truncation_threshold = truncation_threshold,
            seed = seed,
        )
        evs = Vector{Float64}(undef, N)
        for (idx, pstr) in enumerate(pauli_strings)
            _, pt = _parse_qiskit_pauli_string(pstr)
            evs[idx] = _mps_pauli_term_expectation(mps, pt)
        end
        return evs
    end

    psi = simulate_statevector(qc; precision = precision, seed = seed)
    evs = Vector{Float64}(undef, N)
    for (idx, pstr) in enumerate(pauli_strings)
        _, pt = _parse_qiskit_pauli_string(pstr)
        evs[idx] = _statevector_pauli_term_expectation(psi, pt)
    end
    return evs
end

# ==============================================================================
# Native Grover Adaptive Search (GAS) Statevector Phase-Oracle & Diffuser
# ==============================================================================

function _twos_complement_wrap(val::Int, m::Int)::Int
    modulus = 1 << m
    half = 1 << (m - 1)
    return mod(val + half, modulus) - half
end

"""
    gas_grover_search(n, L_int, Q_int; num_value_qubits, max_iter = 3, shots = 100, seed_simulator = nothing, seed_optimizer = nothing)

Execute Gilliam-Woerner-Gonciulea (2021) Grover Adaptive Search using an exact `m = num_value_qubits`
two's-complement statevector phase oracle and Grover diffuser on `2^n` key states.
"""
function gas_grover_search(
    n::Integer,
    L_int::AbstractVector{<:Integer},
    Q_int::AbstractMatrix{<:Integer};
    num_value_qubits::Integer,
    max_iter::Integer = 3,
    shots::Integer = 100,
    seed_simulator::Union{Nothing,Integer} = nothing,
    seed_optimizer::Union{Nothing,Integer} = nothing,
)
    num_keys = Int(n)
    m = Int(num_value_qubits)
    num_keys == 0 && return (
        best_state = Int[],
        threshold = nothing,
        evaluations = 0,
        state_counts = Pair{Vector{Int},Int}[Int[] => Int(shots)],
    )

    dim = 1 << num_keys
    obj_values = Vector{Int}(undef, dim)
    @inbounds for mask in 0:(dim - 1)
        val = 0
        for i in 1:num_keys
            xi = (mask >> (i - 1)) & 1
            if xi == 1
                val += Int(L_int[i])
                for j in 1:num_keys
                    xj = (mask >> (j - 1)) & 1
                    if xj == 1
                        val += Int(Q_int[i, j])
                    end
                end
            end
        end
        obj_values[mask + 1] = val
    end

    rng_opt = seed_optimizer === nothing ? Random.MersenneTwister(0x5eed1234) : Random.MersenneTwister(Int(seed_optimizer))
    rng_sim = seed_simulator === nothing ? Random.MersenneTwister(0x5eed5678) : Random.MersenneTwister(Int(seed_simulator))

    # Iteration 0: uniform sample to initialize threshold y
    init_idx0 = rand(rng_sim, 0:(dim - 1))
    best_idx0 = init_idx0
    y = obj_values[best_idx0 + 1]

    k_schedule = 1.0
    lambda = 8.0 / 7.0
    no_improvement = 0
    evaluations = 1
    last_probs = fill(1.0 / dim, dim)

    psi = Vector{Float64}(undef, dim)
    unif_amp = 1.0 / sqrt(Float64(dim))

    while no_improvement < max_iter
        rot_upper = max(1, ceil(Int, k_schedule))
        r = rand(rng_opt, 1:rot_upper)

        fill!(psi, unif_amp)
        for _ in 1:r
            # Two's-complement m-qubit phase oracle O_y
            @inbounds for idx in 1:dim
                diff_val = obj_values[idx] - y
                if _twos_complement_wrap(diff_val, m) < 0
                    psi[idx] = -psi[idx]
                end
            end
            # Grover diffuser D = 2|+><+| - I
            mean_amp = sum(psi) / dim
            @inbounds for idx in 1:dim
                psi[idx] = 2.0 * mean_amp - psi[idx]
            end
        end

        evaluations += 1
        @inbounds for idx in 1:dim
            last_probs[idx] = abs2(psi[idx])
        end

        # Draw `shots` samples from last_probs
        cdf = cumsum(last_probs)
        cdf[end] = 1.0
        shot_counts = Dict{Int,Int}()
        for _ in 1:shots
            u = rand(rng_sim, Float64)
            s_idx0 = clamp(searchsortedfirst(cdf, u) - 1, 0, dim - 1)
            shot_counts[s_idx0] = get(shot_counts, s_idx0, 0) + 1
        end

        # Pick highest-count (and lowest-energy tie-break) sampled outcome
        candidate_idx0 = best_idx0
        best_count = -1
        for (s_idx0, c) in shot_counts
            if c > best_count || (c == best_count && obj_values[s_idx0 + 1] < obj_values[candidate_idx0 + 1])
                best_count = c
                candidate_idx0 = s_idx0
            end
        end

        cand_val = obj_values[candidate_idx0 + 1]
        if cand_val < y
            best_idx0 = candidate_idx0
            y = cand_val
            k_schedule = 1.0
            no_improvement = 0
        else
            k_schedule = min(k_schedule * lambda, sqrt(Float64(dim)) + 1.0)
            no_improvement += 1
        end
    end

    best_state = Int[(best_idx0 >> (i - 1)) & 1 for i in 1:num_keys]

    return (
        best_state = best_state,
        threshold = Float64(y),
        evaluations = evaluations,
        state_counts = Pair{Vector{Int},Int}[best_state => Int(shots)],
    )
end
