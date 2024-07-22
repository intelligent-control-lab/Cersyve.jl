module Unicycle

using Flux
using LinearAlgebra
using EllipsisNotation

# state: [v, xo, yo]
x_dim = 3
x_low = Float32[-1, -1, -1]
x_high = Float32[1, 1, 1]

# action: [a, w]
u_dim = 2
u_low = Float32[-1, -1]
u_high = Float32[1, 1]

dt = 0.1
ro = 0.4

function dynamics(x::Array{Float32}, u::Array{Float32})::Array{Float32}
    v = x[1, ..]
    xo = x[2, ..]
    yo = x[3, ..]

    a = u[1, ..]
    w = u[2, ..]

    xo_tmp = xo - v * dt
    dth = w * dt
    x_prime = Float32.(stack((
        v + a * dt,
        xo_tmp .* cos.(dth) + yo .* sin.(dth),
        yo .* cos.(dth) - xo_tmp .* sin.(dth),
    ); dims=1))
    return clamp.(x_prime, x_low, x_high)
end

function constraint(x::Array{Float32})::Array{Float32}
    return ro .- sqrt.(sum(x[2:3, ..] .^ 2; dims=1))
end

function terminated(x::Array{Float32})::Array{Float32}
    xo = x[2, ..]
    yo = x[3, ..]
    return (xo .== x_low[2]) .| (xo .== x_high[2]) .| (
        yo .== x_low[3]) .| (yo .== x_high[3])
end

pi_model = Dense(Float32[0 0 0; 0 0 -1], Float32[1, 0])

end
