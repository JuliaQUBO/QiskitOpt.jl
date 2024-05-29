using QiskitOpt: QUBODrivers, VQE, QAOA

QUBODrivers.test(QAOA.Optimizer) do model
end
QUBODrivers.test(VQE.Optimizer) do model
end