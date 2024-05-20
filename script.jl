using QiskitOpt, QUBO, JuMP

model = Model(() -> ToQUBO.Optimizer(VQE.Optimizer))
@variable(model, 1 <= y[1:3] <= 5, Int)
@variable(model, 2 <= x[1:3] <= 5, Int)

@objective(model, Min, x' * x .- sum(y))

MOI.set(model, VQE.IBMBackend(), "ibm_osaka")

optimize!(model)