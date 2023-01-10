module VQE

using Anneal
using PythonCall
using ..QiskitOpt: qiskit, qiskit_optimization, qiskit_optimization_algorithms, qiskit_algorithms, qiskit_optimization_runtime

Anneal.@anew Optimizer begin
    name    = "VQE @ IBMQ"
    sense   = :min
    domain  = :bool
    version = v"0.4.0"
    attributes = begin
        NumberOfReads["num_reads"]::Integer        = 1_000
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
        IBMBackend["ibm_backend"]::String          = "ibmq_qasm_simulator"
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # -*- Retrieve Attributes - *-
    seed        = MOI.get(sampler, VQE.RandomSeed())
    num_reads   = MOI.get(sampler, VQE.NumberOfReads())
    ibm_backend = MOI.get(sampler, VQE.IBMBackend())

    # -*- Retrieve Model -*- #
    Q, α, β = Anneal.qubo(sampler, Dict)

    linear    = PythonCall.pydict()
    quadratic = PythonCall.pydict()

    for ((i, j), q) in Q
        if i == j
            linear[string(i)] = q
        else
            quadratic[string(i), string(j)] = q
        end
    end

    # -*- Build Qiskit Model -*- #
    qp = qiskit_optimization.QuadraticProgram()

    for v in string.(Anneal.indices(sampler))
        qp.binary_var(v)
    end

    qp.minimize(linear = linear, quadratic = quadratic)

    # Results vector
    samples = Vector{Anneal.Sample{T,Int}}(undef, num_reads)

    # Timing Information 
    time_data = Dict{String,Any}()

    # Connect to IBMQ and get backend
    connect_vqe(ibm_backend,  qp.get_num_binary_vars()) do client
        vqe    = qiskit_optimization_algorithms.MinimumEigenOptimizer(client)
        results = vqe.solve(qp)

        Ψ = Vector{Int}[]
        ρ = Float64[]
        Λ = T[]

        for sample in results.samples
            # state:
            push!(Ψ, pyconvert.(Int, sample.x))
            # reads:
            push!(ρ, pyconvert(Float64, sample.probability))
            # value: 
            push!(Λ, α * (pyconvert(T, sample.fval) + β))
        end

        P = cumsum(ρ)

        for i = 1:num_reads
            p = rand()
            j = first(searchsorted(P, p))

            samples[i] = Sample{T}(Ψ[j], Λ[j])
        end

        time_data["effective"] =
            pyconvert(Float64, results.min_eigen_solver_result.optimizer_time)

        return nothing
    end

    metadata = Dict{String,Any}("time"   => time_data, "origin" => "IBMQ @ $(ibm_backend)")


    return Anneal.SampleSet{T}(samples, metadata)
end

function connect_vqe(callback::Function, ibm_backend::String, num_qubits, shots::Int = 1024)
    qiskit.IBMQ.load_account()

    provider = qiskit.IBMQ.get_provider()
    backend  = provider.get_backend(ibm_backend)
    SPSA = qiskit_algorithms.optimizers.SPSA(maxiter=50)

    ansatz = qiskit.circuit.library.EfficientSU2(
        num_qubits=num_qubits, 
        reps=1, 
        entanglement="linear"
    )

    client = qiskit_optimization_runtime.VQEClient(
        provider = provider,
        backend = backend,
        ansatz = ansatz,
        optimizer = SPSA,
        shots = shots
    )

    callback(client)

    return nothing
end

end # module VQE