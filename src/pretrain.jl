function pretrain_value(
    V_model::Chain,
    f_pi_model::Any,
    h_model::Any,
    x_low::Vector{Float32},
    x_high::Vector{Float32};
    gamma::Float64 = 0.9,
    lr::Float64 = 3e-4,
    batch_size::Int64 = 256,
    iter_num::Int64 = 100000,
    weight_decay::Float64 = 0.0,
    penalty::Union{String, Nothing} = nothing,  # "APA" / "SNR" / nothing
    noise_scale::Float64 = 0.1,
    space_size::Union{Vector{Float32}, Nothing} = nothing,
    apa_coef::Float64 = 0.01,
    snr_coef::Tuple{Float64, Float64} = (1e-3, 5e-4),
    log_dir::Union{String, Nothing} = nothing,
)
    rng = MersenneTwister(1)
    optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), V_model)
    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "pretrain_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    for _ in ProgressBar(1:iter_num)
        x = uniform(x_low, x_high, batch_size)
        x_prime = f_pi_model(x)
        c = h_model(x)
        v_targ = (1 - gamma) * c + gamma * max.(c, V_model(x_prime))

        function loss_fn(m)
            if isnothing(penalty)
                return mean((m(x) - v_targ) .^ 2)
            elseif penalty == "SNR"
                noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(x))
                v_pred, snr_loss = forward_with_snr(m, [x x + noise]; alpha=snr_coef[1], beta=snr_coef[2])
                return mean((v_pred - v_targ) .^ 2) + snr_loss
            elseif penalty == "APA"
                noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(x))
                v_pred, apa_loss = forward_with_apa(m, [x x + noise]; alpha=apa_coef)
                return mean((v_pred - v_targ) .^ 2) + apa_loss
            end
        end

        loss, grad = Flux.withgradient(loss_fn, V_model)
        Flux.update!(optim, V_model, grad[1])

        with_logger(logger) do
            @info "pretrain" loss=loss
            @info "pretrain" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
            @info "pretrain" predicted_feasible_rate=mean(V_model(x) .<= 0) log_step_increment=0
        end
    end

    jldsave(joinpath(log_path, "/V_pretrain.jld2"); state=Flux.state(V_model))
end
