module VQE

using Anneal
using ..QiskitOpt: connect, qiskit, qiskit_optimization

Anneal.@anew Optimizer begin
    name    = "VQE @ IBMQ"
    sense   = :min
    domain  = :bool
    version = v"0.4.0"
end

function Anneal.sample(sampler::Optimizer{T}) where {T}

    return Anneal.SampleSet{T}()
end

end # module VQE