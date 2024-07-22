function create_mlp(
    input_dim::Int64,
    output_dim::Int64,
    hidden_sizes::Vector{Int64},
)::Chain
    layers = []
    push!(layers, Dense(input_dim => hidden_sizes[1], relu))
    for i in 1:length(hidden_sizes) - 1
        push!(layers, Dense(hidden_sizes[i] => hidden_sizes[i + 1], relu))
    end
    push!(layers, Dense(hidden_sizes[end] => output_dim))
    return Chain(layers...)
end

function create_closed_loop_dynamics_model(
    f_model::Chain,
    pi_model::Any,
    data::Dict{String, Array{Float32}},
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    u_dim::Int64,
)::Chain
    x_dim = length(x_low)
    return Chain(
        Parallel(+,
            Dense(Matrix{Float32}(I(x_dim))),
            Chain(
                Parallel(+,
                    Chain(
                        Dense(diagm(1 ./ data["x_std"]), -data["x_mean"] ./ data["x_std"]),
                        Dense(vcat(Matrix{Float32}(I(x_dim)), zeros(Float32, u_dim, x_dim))),
                    ),
                    Chain(
                        pi_model,
                        Dense(diagm(1 ./ data["u_std"]), -data["u_mean"] ./ data["u_std"]),
                        Dense(vcat(zeros(Float32, x_dim, u_dim), Matrix{Float32}(I(u_dim)))),
                    ),
                ),
                f_model,
                Dense(diagm(data["dx_std"]), data["dx_mean"]),
            ),
        ),
        # max(x, x_low) = relu(x - x_low) + x_low
        Dense(Matrix{Float32}(I(x_dim)), -x_low, relu),
        Dense(Matrix{Float32}(I(x_dim)), x_low),
        # min(x, x_high) = -max(-x, -x_high) = -relu(-x + x_high) + x_high
        Dense(-Matrix{Float32}(I(x_dim)), x_high, relu),
        Dense(-Matrix{Float32}(I(x_dim)), x_high),
    )
end

function create_value_constraint_model(V_model::Any, h_model::Any)::Chain
    return Chain(Parallel(+,
        Chain(V_model, Dense(Float32[1; 0;;])),
        Chain(h_model, Dense(Float32[0; 1;;])),
    ))
end

function create_value_next_value_model(V_model::Any, f_pi_model::Any)::Chain
    return Chain(Parallel(+,
        Chain(V_model, Dense(Float32[1; 0;;])),
        Chain(f_pi_model, V_model, Dense(Float32[0; 1;;])),
    ))
end
