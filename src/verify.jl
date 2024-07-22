function verify_value(
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    V_h_model::Any,
    V_V_prime_model::Any;
    con_start_values::Union{Nothing, Vector{Float64}} = nothing,
    inv_start_values::Union{Nothing, Vector{Float64}} = nothing,
)::Tuple{ModelVerification.ResultInfo, ModelVerification.ResultInfo}
    search_method = BFS(max_iter=1000000, batch_size=1000)
    split_method = Bisect(1)
    solver = MIPVerify(pre_bound_method=Crown())
    X = Hyperrectangle(low=x_low, high=x_high)
    Y = Complement(HPolyhedron([1 0; 0 -1], [0, 0]))

    # verify constraint property
    con_problem = Problem(V_h_model, X, Y)
    con_t = @elapsed con_res = verify(search_method, split_method, solver, con_problem;
        collect_bound=true, start_values=con_start_values)
    @printf "Constraint property %s! Verification time: %.3fs\n" con_res.status con_t

    # verify invariance property
    inv_problem = Problem(V_V_prime_model, X, Y)
    inv_t = @elapsed inv_res = verify(search_method, split_method, solver, inv_problem;
        collect_bound=true, start_values=inv_start_values)
    @printf "Invariance property %s! Verification time: %.3fs\n" inv_res.status inv_t

    return con_res, inv_res
end
