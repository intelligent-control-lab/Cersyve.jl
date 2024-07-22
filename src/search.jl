function boundary_guided_search(
    x::Matrix{Float32},
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    h_model::Any,
    V_model::Any,
    f_pi_model::Any;
    pgd_step::Int64 = 10,
    pgd_eps::Float64 = 0.1,
    pgd_beta::Float64 = 0.0,
    backtrack_step::Int64 = 20,
    length_discount::Float64 = 0.8,
    bound_guide::Bool = true,
    direct_discount::Float64 = 0.5,
    tol::Float64 = 1e-4,
)::Matrix{Float32}
    pgd = ones(Bool, size(x, 2))
    x_pgd = x
    m = zeros(Float32, size(x))

    for _ in 1:pgd_step
        h = h_model(x_pgd)[1, :]
        v = V_model(x_pgd)[1, :]
        v_prime = V_model(f_pi_model(x_pgd))[1, :]
        con = (v .<= tol) .& (h .> -tol)
        inv = (v .<= tol) .& (v_prime .> -tol)
        pgd_pgd = pgd[pgd]
        pgd_pgd[con .| inv] .= 0
        pgd[pgd] = pgd_pgd
        x_pgd = x[:, pgd]

        con_g = Flux.gradient(x -> sum(h_model(x)), x_pgd[:, 1:div(size(x_pgd, 2), 2)])[1]
        inv_g = Flux.gradient(x -> sum(V_model(f_pi_model(x))), x_pgd[:, size(con_g, 2) + 1:end])[1]
        g = hcat(con_g, inv_g) + Float32(pgd_beta) * m[:, pgd]
        g ./= sqrt.(sum(g .^ 2, dims=1))

        v_g = Flux.gradient(x -> sum(V_model(x)), x_pgd)[1]
        v_g ./= sqrt.(sum(v_g .^ 2, dims=1))

        a = sum(g .* v_g, dims=1)
        z = a .* g - v_g
        z ./= sqrt.(sum(z .^ 2, dims=1))

        dirc = zeros(Float32, size(x_pgd))
        coef = zeros(Float32, size(x_pgd, 2))
        tmp = ones(Bool, size(x_pgd, 2))

        for i in 1:backtrack_step + 1
            if bound_guide
                d_coef = Float32(direct_discount ^ (i - 1))
                d = d_coef * g[:, tmp] + (1 - d_coef) * z[:, tmp]
                d ./= sqrt.(sum(d .^ 2, dims=1))
            else
                d = g[:, tmp]
            end

            l_coef = Float32(length_discount ^ (i - 1))
            x_tmp = x_pgd[:, tmp] + Float32.(l_coef * pgd_eps) * d
            x_tmp = min.(max.(x_tmp, x_low), x_high)

            fea = V_model(x_tmp)[1, :] .<= tol

            dirc[:, tmp] = d

            coef_tmp = coef[tmp]
            coef_tmp[fea] .= l_coef
            coef[tmp] = coef_tmp

            tmp_tmp = tmp[tmp]
            tmp_tmp[fea] .= 0
            tmp[tmp] = tmp_tmp

            if maximum(tmp; init=0) == 0
                break
            end
        end
        dx = reshape(coef, 1, size(x_pgd, 2)) .* dirc
        x_pgd = x_pgd + Float32(pgd_eps) * dx
        x_pgd = min.(max.(x_pgd, x_low), x_high)
        x[:, pgd] = x_pgd

        m_pgd = m[:, pgd]
        m_pgd[:, .~tmp] = dx[:, .~tmp]
        m[:, pgd] = m_pgd
    end
    return x
end

function projected_boundary_search(
    x::Matrix{Float32},
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    h_model::Any,
    V_model::Any,
    f_pi_model::Any;
    lr::Float64 = 0.01,
    grad_step::Int64 = 10,
    proj_step::Int64 = 10,
    proj_tol::Float64 = 0.1,
    tol::Float64 = 1e-4,
)::Matrix{Float32}
    pgd = ones(Bool, size(x, 2))
    x_pgd = x

    for _ in 1:grad_step
        h = h_model(x_pgd)[1, :]
        v = V_model(x_pgd)[1, :]
        v_prime = V_model(f_pi_model(x_pgd))[1, :]
        con = (v .<= tol) .& (h .> -tol)
        inv = (v .<= tol) .& (v_prime .> -tol)
        pgd_pgd = pgd[pgd]
        pgd_pgd[con .| inv] .= 0
        pgd[pgd] = pgd_pgd
        x_pgd = x[:, pgd]

        con_g = Flux.gradient(x -> sum(h_model(x)), x_pgd[:, 1:div(size(x_pgd, 2), 2)])[1]
        inv_g = Flux.gradient(x -> sum(V_model(f_pi_model(x))), x_pgd[:, size(con_g, 2) + 1:end])[1]
        g = hcat(con_g, inv_g)

        # project gradient along boundary
        v_g = Flux.gradient(x -> sum(V_model(x)), x_pgd)[1]
        v_g ./= sqrt.(sum(v_g .^ 2, dims=1))
        proj_g = g - sum(g .* v_g, dims=1) .* v_g

        # update sample
        x_pgd = x_pgd + Float32(lr) * proj_g
        x_pgd = min.(max.(x_pgd, x_low), x_high)

        tmp = ones(Bool, size(x_pgd, 2))
        x_tmp = x_pgd[:, tmp]

        # project sample to boundary
        for _ in 1:proj_step
            projected = abs.(V_model(x_tmp)[1, :]) .<= proj_tol
            tmp_tmp = tmp[tmp]
            tmp_tmp[projected] .= 0
            tmp[tmp] = tmp_tmp

            if maximum(tmp; init=0) == 0
                break
            end

            x_tmp = x_pgd[:, tmp]
            d_tmp = Flux.gradient(x -> sum(abs.(V_model(x))), x_tmp)[1]
            x_tmp = x_tmp - Float32(lr) * d_tmp
            x_tmp = min.(max.(x_tmp, x_low), x_high)
            x_pgd[:, tmp] = x_tmp
        end
    end
    return x
end

function filter_counterexample(
    x::Matrix{Float32},
    h_model::Any,
    V_model::Any,
    f_pi_model::Any;
    tol::Float64 = 1e-4,
)::Tuple{BitVector, BitVector}
    h = h_model(x)[1, :]
    v = V_model(x)[1, :]
    v_prime = V_model(f_pi_model(x))[1, :]
    con = (v .<= tol) .& (h .> -tol)
    inv = (v .<= tol) .& (v_prime .> -tol) .& (.~con)
    return con, inv
end
