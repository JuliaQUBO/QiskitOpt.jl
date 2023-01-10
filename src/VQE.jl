module VQE

using Anneal
using PythonCall

Anneal.@anew Optimizer begin
    name = "VQE @ IBMQ"
end

function Anneal.sample(sampler::Optimizer{T}) where {T}

    return Anneal.SampleSet{T}()
end
    
end # module VQE