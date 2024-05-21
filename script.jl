using QiskitOpt, QUBO, JuMP

model = Model(QiskitOpt.VQE.Optimizer)

Q = [
   -1  2  2
    2 -1  2
    2  2 -1
]

@variable(model, x[1:3], Bin)
@objective(model, Min, x' * Q * x)
MOI.set(model, VQE.IBMBackend(), "ibm_nazca")
MOI.set(model, VQE.Channel(), "ibm_quantum")
MOI.set(model, VQE.Instance(), "ibm-q-asu/main/purdue-david-ber")


optimize!(model)


