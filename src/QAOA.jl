module QAOA

using Anneal
using PythonCall

Anneal.@anew Optimizer begin
    name = "QAOA @ IBMQ"
end

function Anneal.sample(sampler::Optimizer{T}) where {T}

    return Anneal.SampleSet{T}()
end
    
end # module QAOA