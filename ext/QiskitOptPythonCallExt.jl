module QiskitOptPythonCallExt

using QiskitOpt
using PythonCall

function __init__()
    QiskitOpt._is_py_hook[] = x -> x isa PythonCall.Py
    QiskitOpt._py_import_hook[] = module_name -> PythonCall.pyimport(String(module_name))
    QiskitOpt._py_executable_hook[] = () -> begin
        try
            return PythonCall.pyconvert(String, PythonCall.pyimport("sys").executable)
        catch
            return nothing
        end
    end
    QiskitOpt._py_version_hook[] = () -> QiskitOpt._module_version(PythonCall)
    QiskitOpt._py_str_hook[] = x -> PythonCall.pyconvert(String, PythonCall.pystr(x))
    QiskitOpt._py_int_hook[] = x -> PythonCall.pyconvert(Int, x)
    QiskitOpt._py_float_hook[] = x -> PythonCall.pyconvert(Float64, x)
    QiskitOpt._py_bool_hook[] = x -> PythonCall.pyconvert(Bool, x)
    QiskitOpt._py_len_hook[] = x -> Int(PythonCall.pylen(x))
    QiskitOpt._py_hasattr_hook[] = (x, name) -> Bool(PythonCall.pyhasattr(x, String(name)))
    QiskitOpt._py_is_none_hook[] = x -> Bool(PythonCall.pyis(x, PythonCall.pybuiltins.None))
    QiskitOpt._py_int_arg_hook[] = x -> PythonCall.pyint(Int(x))
    QiskitOpt._py_list_arg_hook[] = x -> PythonCall.pylist(x)
    QiskitOpt._py_exception_type_hook[] = err -> begin
        if err isa PythonCall.PyException
            return QiskitOpt.QAOA._python_qualified_type(err.v)
        end
        return nothing
    end
    QiskitOpt._py_int_vec_hook[] = x -> PythonCall.pyconvert(Vector{Int}, x)
    QiskitOpt._py_float_vec_hook[] = x -> begin
        try
            return PythonCall.pyconvert(Vector{Float64}, x)
        catch
            return Float64[PythonCall.pyconvert(Float64, val) for val in x]
        end
    end
    return nothing
end

end # module QiskitOptPythonCallExt

