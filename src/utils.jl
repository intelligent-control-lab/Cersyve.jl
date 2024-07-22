function uniform(
    low::Vector{Float32},
    high::Vector{Float32},
    N::Int,
)::Matrix{Float32}
    return low .+ rand(Float32, (length(low), N)) .* (high - low)
end

function collect_data(
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    u_low::Vector{Float32},
    u_high::Vector{Float32},
    dynamics::Function,
    terminated::Function;
    step_num::Int64 = 1000000,
    max_ep_len::Int64 = 100,
    noise_scale::Float64 = 0.01,
    save_path::Union{String, Nothing} = nothing,
)
    data = Dict(
        "x" => Matrix{Float32}(undef, length(x_low), step_num),
        "u" => Matrix{Float32}(undef, length(u_low), step_num),
        "dx" => Matrix{Float32}(undef, length(x_low), step_num),
    )

    x = uniform(x_low, x_high, 1)
    ep_step = 0
    for i in ProgressBar(1:step_num)
        u = uniform(u_low, u_high, 1)
        x_prime = dynamics(x, u)
        ep_step += 1
    
        data["x"][:, i] = x[:, 1]
        data["u"][:, i] = u[:, 1]
        data["dx"][:, i] = x_prime[:, 1] - x[:, 1]
    
        if Bool(terminated(x)[1]) || (ep_step == max_ep_len)
            x = uniform(x_low, x_high, 1)
            ep_step = 0
        else
            x = x_prime
        end
    end

    data_prep = Dict{String, Array{Float32}}()
    rng = MersenneTwister(1)
    for (k, v) in data
        v_mean = mean(v; dims=2)[:, 1]
        v_std = std(v; dims=2)[:, 1]
        noise = Float32(noise_scale) * randn(rng, Float32, size(v))
        data_prep[k] = (v .- v_mean) ./ v_std + noise
        data_prep[k * "_mean"] = v_mean
        data_prep[k * "_std"] = v_std
    end

    if isnothing(save_path)
        save_path = joinpath(@__DIR__, "../data/data.jld2")
    end
    save(save_path, "data", data_prep)
end

function train_dynamics(
    data::Dict{String, Array{Float32}},
    f_model::Chain;
    closed_loop::Bool = false,
    lr::Float64 = 1e-3,
    batch_size::Int64 = 256,
    epoch_num::Int64 = 100,
    weight_decay::Float64 = 0.0,
    penalty::Union{String, Nothing} = nothing,  # "APA" / "SNR" / nothing
    noise_scale::Float64 = 0.1,
    space_size::Union{Vector{Float32}, Nothing} = nothing,
    apa_coef::Float64 = 0.01,
    snr_coef::Tuple{Float64, Float64} = (1e-3, 5e-4),
    log_dir::Union{String, Nothing} = nothing,
)
    if closed_loop
        xu = data["x"]
    else
        xu = [data["x"]; data["u"]]
    end
    dx = data["dx"]

    rng = MersenneTwister(1)
    loader = Flux.DataLoader((xu, dx), batchsize=batch_size, shuffle=true)
    optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), f_model)
    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "dynamics_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    for _ in ProgressBar(1:epoch_num)
        losses = []
        mses = []
        for (batch_xu, batch_dx) in loader
            function loss_fn(m)
                if isnothing(penalty)
                    return mean((m(batch_xu) - batch_dx) .^ 2)
                elseif penalty == "SNR"
                    noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(batch_xu))
                    dx_pred, snr_loss = forward_with_snr(m, [batch_xu batch_xu + noise]; alpha=snr_coef[1], beta=snr_coef[2])
                    return mean((dx_pred - batch_dx) .^ 2) + snr_loss
                elseif penalty == "APA"
                    noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(batch_xu))
                    dx_pred, apa_loss = forward_with_apa(m, [batch_xu batch_xu + noise]; alpha=apa_coef)
                    return mean((dx_pred - batch_dx) .^ 2) + apa_loss
                end
            end

            loss, grad = Flux.withgradient(loss_fn, f_model)
            Base.push!(losses, loss)
            Base.push!(mses, mean((f_model(batch_xu) - batch_dx) .^ 2))
            Flux.update!(optim, f_model, grad[1])
        end

        with_logger(logger) do
            @info "dynamics" loss=mean(losses)
            @info "dynamics" mean_squared_error=mean(mses) log_step_increment=0
        end
    end

    jldsave(joinpath(log_path, "f.jld2"); state=Flux.state(f_model))
end

function train_constraint(
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    h_model::Chain,
    constraint::Function;
    lr::Float64 = 3e-4,
    batch_size::Int64 = 256,
    n_iter::Int64 = 100000,
    log_dir::Union{String, Nothing} = nothing,
)
    optim = Flux.setup(Adam(lr), h_model)
    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "constraint_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    for _ in ProgressBar(1:n_iter)   
        x = uniform(x_low, x_high, batch_size)
        c = constraint(x)

        loss, grad = Flux.withgradient(m -> mean((m(x) - c) .^ 2), h_model)
        Flux.update!(optim, h_model, grad[1])

        with_logger(logger) do
            @info "constraint" loss=loss
            @info "constraint" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
        end
    end

    jldsave(joinpath(log_path, "h.jld2"); state=Flux.state(h_model))
end

function forward_with_snr(
    model::Chain,
    x::Matrix{Float32};
    alpha::Float64 = 1e-3,
    beta::Float64 = 5e-4,
)::Tuple{Matrix{Float32}, Float64}
    # Signal-to-Noise ratio
    batch_size = div(size(x, 2), 2)
    snr_loss = 0
    for layer in model.layers
        x = layer.weight * x .+ layer.bias
        if layer.σ == relu
            x_und = x[:, 1:batch_size]
            x_dis = x[:, batch_size + 1:end]
            var_loss = mean(sum((x_dis - x_und) .^ 2, dims=1) ./ 
                sum(Zygote.dropgrad(x_und) .^ 2, dims=1))
            sta_loss = mean(sum(abs.(x_dis - x_und) ./ (x_und .^ 2 .+ eps()), dims=1))
            snr_loss += alpha * var_loss + beta * sta_loss
        end
        x = layer.σ(x)
    end
    return x[:, 1:batch_size], snr_loss
end

function forward_with_apa(
    model::Chain,
    x::Matrix{Float32};
    alpha::Float64 = 0.01,
    eps::Float64 = 1e-4,
)::Tuple{Matrix{Float32}, Float64}
    # Activation pattern alignment
    batch_size = div(size(x, 2), 2)
    apa_loss = 0
    for layer in model.layers
        x = layer.weight * x .+ layer.bias
        if layer.σ == relu
            x_und = x[:, 1:batch_size]
            x_dis = x[:, batch_size + 1:end]
            apa_loss += alpha * mean(sum(max.(-x_und .* x_dis, 0) ./
                Zygote.dropgrad(max.(-x_und .* x_dis, eps)), dims=1))
        end
        x = layer.σ(x)
    end
    return x[:, 1:batch_size], apa_loss
end

function get_activation_pattern(model::Any, x::Matrix{Float32})::BitMatrix
    p = []

    function process_layer(layer, x)
        if isa(layer, Dense)
            z = layer.weight * x .+ layer.bias
            if layer.σ == relu
                push!(p, z .> 0)
            end
            return layer.σ.(z)
        elseif isa(layer, Chain)
            for (i, l) in enumerate(layer.layers)
                x = process_layer(l, x)
                if i < length(layer.layers) && layer.layers[i + 1] == relu
                    push!(p, x .> 0)
                end
            end
            return x
        elseif isa(layer, Parallel)
            outputs = []
            for l in layer.layers
                push!(outputs, process_layer(l, x))
            end
            # currently only support "+" operator
            return sum(outputs)
        else
            return layer(x)
        end
    end

    process_layer(model, x)
    return cat(p..., dims=1)
end

function count_activation_pattern(p::BitMatrix)::Tuple{Vector{Int}, Vector{Int}}
    unique_columns = Set{Tuple}()
    log_points = [10 ^ i for i in 1:Int64(round(log(10, size(p, 2))))]
    uni_pat_num = []
    for (i, col) in enumerate(eachcol(p))
        push!(unique_columns, tuple(col...))
        if i in log_points
            push!(uni_pat_num, length(unique_columns))
        end
    end
    return log_points, uni_pat_num
end
