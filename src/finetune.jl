function finetune_value(
    V_model::Any,
    f_pi_model::Any,
    h_model::Any,
    x_low::Vector{Float32},
    x_high::Vector{Float32};
    lr::Float64 = 1e-4,
    max_iter::Int64 = 100000,
    search_size::Int64 = 1000,
    bnd_ratio::Float64 = 0.1,
    bnd_ratio_avg::Float64 = 0.9,
    min_bnd_ratio::Float64 = 0.01,
    max_bnd_ratio::Float64 = 1.0,
    bnd_eps::Float64 = 0.1,
    search_method::String = "BGB",  # "BGB" / "PGD-B" / "PBS"
    pgd_step::Int64 = 10,
    pgd_eps::Float64 = 0.1,
    backtrack_step::Int64 = 20,
    length_discount::Float64 = 0.8,
    direct_discount::Float64 = 0.5,
    tol::Float64 = 1e-4,
    capacity::Int64 = 10000,
    sample_size::Int64 = 100,
    search_stop::Int64 = 1000,
    replay::Int64 = 1,
    max_skip::Int64 = 1000,
    reg_method::Union{String, Nothing} = "ESR",  # "ESR" / "RSR" / nothing
    esr_max_con::Union{Int64, Nothing} = nothing,
    eps_h::Float64 = 0.01,
    eps_v::Float64 = 0.01,
    reg_coef::Float64 = 0.1,
    log_dir::Union{String, Nothing} = nothing,
    eval_every::Int64 = 10,
    save_every::Int64 = 1000,
)
    skipped = 0
    verified = 0
    con_start_values = nothing
    inv_start_values = nothing

    buffer = Buffer(capacity, length(x_low))
    opt_state = Flux.setup(Adam(lr), V_model)
    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "finetune_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    V_h_model = create_value_constraint_model(V_model, h_model)
    V_V_prime_model = create_value_next_value_model(V_model, f_pi_model)

    for i in ProgressBar(1:max_iter)
        if (length(buffer.stored) < search_stop)
            x = uniform(x_low, x_high, round(Int64, search_size / bnd_ratio))
            v = V_model(x)[1, :]
            x_bnd = x[:, (v .> -bnd_eps) .& (v .<= tol)]
            bnd_ratio = bnd_ratio_avg * bnd_ratio + (1 - bnd_ratio_avg) * size(x_bnd, 2) / size(x, 2)
            bnd_ratio = clamp(bnd_ratio, min_bnd_ratio, max_bnd_ratio)

            if search_method == "BGB"
                x_pgd = boundary_guided_search(x_bnd, x_low, x_high, h_model, V_model, f_pi_model;
                    pgd_step=pgd_step, pgd_eps=pgd_eps, backtrack_step=backtrack_step,
                    length_discount=length_discount, bound_guide=true, direct_discount=direct_discount,
                    tol=tol)
            elseif search_method == "PGD-B"
                x_pgd = boundary_guided_search(x_bnd, x_low, x_high, h_model, V_model, f_pi_model;
                    pgd_step=pgd_step, pgd_eps=pgd_eps, backtrack_step=backtrack_step,
                    length_discount=length_discount, bound_guide=false, tol=tol)
            elseif search_method == "PBS"
                x_pgd = projected_boundary_search(x_bnd, x_low, x_high, h_model, V_model, f_pi_model)
            end
            con, inv = filter_counterexample(x_pgd, h_model, V_model, f_pi_model; tol=tol)
            ce = con .| inv
            push!(buffer, x_pgd[:, ce])

            with_logger(logger) do
                @info "finetune" searched_boundary_states=size(x_bnd, 2) log_step_increment=0
                @info "finetune" boundary_state_ratio=bnd_ratio log_step_increment=0
                @info "finetune" searched_constraint_counterexample=sum(con) log_step_increment=0
                @info "finetune" searched_invariance_counterexample=sum(inv) log_step_increment=0
            end
        end

        if length(buffer.stored) > 0
            skipped = 0

            n = min(sample_size, length(buffer.stored))
            x, c = pop!(buffer, n)
            con, inv = filter_counterexample(x, h_model, V_model, f_pi_model; tol=tol)
            x_con, x_inv = x[:, con], x[:, inv]
            c[con .| inv] .= 0
            c[.~con .& .~inv] .+= 1
            push_idx = c .< replay
            push!(buffer, x[:, push_idx], c[push_idx])

            n_con, n_inv = size(x_con, 2), size(x_inv, 2)

            if !isnothing(reg_method) && (n_con + n_inv > 0) && (
                isnothing(esr_max_con) || (n_con + n_inv < esr_max_con))
                if reg_method == "ESR"
                    # entering state regularization
                    x_reg = uniform(x_low, x_high, search_size)
                    h_reg = h_model(x_reg)[1, :]
                    v_reg = V_model(x_reg)[1, :]
                    v_reg_prime = V_model(f_pi_model(x_reg))[1, :]
                    entering = (h_reg .<= -eps_h) .& (v_reg .> 0) .& (
                        v_reg .<= eps_v) .& (v_reg_prime .<= -eps_v)
                    x_reg = x_reg[:, entering]
                    n_reg = size(x_reg, 2)
                elseif reg_method == "RSR"
                    # random state regularization
                    x_reg = uniform(x_low, x_high, search_size)
                    n_reg = size(x_reg, 2)
                end
            else
                n_reg = 0
            end

            function value_loss_fn(V_model)
                if n_con > 0
                    con_loss = sum(-V_model(x_con))
                else
                    con_loss = 0
                end

                if n_inv > 0
                    inv_loss = sum(-V_model(x_inv) + V_model(f_pi_model(x_inv)))
                else
                    inv_loss = 0
                end

                if n_reg > 0
                    reg_loss = mean(V_model(x_reg))
                else
                    reg_loss = 0
                end

                loss = (con_loss + inv_loss) / max(n_con + n_inv, 1) + reg_coef * reg_loss
                return loss
            end

            loss, grad = Flux.withgradient(value_loss_fn, V_model)
            Flux.update!(opt_state, V_model, grad[1])

            with_logger(logger) do
                @info "finetune" sample_size=n log_step_increment=0
                @info "finetune" value_loss=loss log_step_increment=0
                @info "finetune" sampled_constraint_counterexample=n_con log_step_increment=0
                @info "finetune" sampled_invariance_counterexample=n_inv log_step_increment=0
                if !isnothing(reg_method)
                    @info "finetune" regularization_state=n_reg log_step_increment=0
                end
            end
        else
            skipped += 1
        end

        if skipped == max_skip
            jldsave(joinpath(log_path, "/V_finetune.jld2"); state=Flux.state(V_model))
            println("----- Verification Starts -----")
            con_res, inv_res = verify_feasible_region(x_low, x_high, V_h_model, V_V_prime_model;
                con_start_values=con_start_values, inv_start_values=inv_start_values)
            println("----- Verification Ends -----")

            verified += 1

            if (con_res.status == :holds) & (inv_res.status == :holds)
                break
            else
                if con_res.status == :violated
                    ce = Float32.(con_res.info[:counter_example])
                    push!(buffer, reshape(ce, length(ce), 1))
                    println("Constraint counterexample: ", ce)
                    con_start_values = con_res.info[:verified_bounds][:values]
                end
                if inv_res.status == :violated
                    ce = Float32.(inv_res.info[:counter_example])
                    push!(buffer, reshape(ce, length(ce), 1))
                    println("Invariance counterexample: ", ce)
                    inv_start_values = inv_res.info[:verified_bounds][:values]
                end
                println("")
                skipped = 0
            end
        end

        with_logger(logger) do
            @info "finetune" total_counterexample=length(buffer.stored)
            @info "finetune" skipped_update=skipped log_step_increment=0
            @info "finetune" verified_times=verified log_step_increment=0
        end

        if i % eval_every == 0
            fea_rate = mean(V_model(uniform(x_low, x_high, search_size)) .<= 0)

            with_logger(logger) do
                @info "finetune" predicted_feasible_rate=fea_rate log_step_increment=0
            end
        end

        if i % save_every == 0
            jldsave(joinpath(log_path, "/V_finetune.jld2"); state=Flux.state(V_model))
        end
    end
end
