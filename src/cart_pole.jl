module CartPole

using Flux
using EllipsisNotation
using LinearAlgebra
using MatrixEquations

# state: [y, th, y_dot, th_dot]
x_dim = 4
x_high = Float32[1, 24 * pi / 180, 1, 1]
x_low = -x_high

# action: [F]
u_dim = 1
u_high = Float32[1]
u_low = -u_high

y_con = 0.8
th_con = 12 * pi / 180

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
lengthpole = 0.5
polemass_length = masspole * lengthpole
force_mag = 10.0
dt = 0.05

function dynamics(x::Array{Float32}, u::Array{Float32})::Array{Float32}
    y, th, y_dot, th_dot = x[1, ..], x[2, ..], x[3, ..], x[4, ..]
    F = u[1, ..]
    temp = (force_mag * F + polemass_length * th_dot .^ 2 .* sin.(th)) / total_mass
    thacc = (gravity * sin.(th) - cos.(th) .* temp) ./ (
        lengthpole * (4.0 / 3.0 .- masspole * cos.(th) .^ 2 / total_mass)
    )
    yacc = temp - polemass_length * thacc .* cos.(th) / total_mass
    x_prime = Float32.(stack((
        y + dt * y_dot,
        th + dt * th_dot,
        y_dot + dt * yacc,
        th_dot + dt * thacc,
    ); dims=1))
    return clamp.(x_prime, x_low, x_high)
end

function terminated(x::Array{Float32})::Array{Float32}
    y, th = x[1, ..], x[2, ..]
    return (y .== x_low[1]) .| (y .== x_high[1]) .| (
        th .== x_low[2]) .| (th .== x_high[2])
end

h_y_model = Chain(
    Dense(Float32[1 0 0 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-y_con]),
    )
)
h_th_model = Chain(
    Dense(Float32[0 1 0 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-th_con]),
    )
)
h_model = Parallel(+,
    Chain(
        Parallel(+,
            h_y_model,
            Chain(
                h_th_model,
                Dense(-Matrix{Float32}(I(1))),
            ),
        ),
        Dense(Matrix{Float32}(I(1)), Float32[0], relu),
    ),
    h_th_model,
)

temp = lengthpole * (4.0 / 3.0 - masspole / total_mass)
A = Float32[
    1 0 dt 0;
    0 1 0 dt;
    0 0 1 -dt * polemass_length / total_mass;
    0 dt * gravity / temp 0 1;
]
B = Float32[
    0;
    0;
    dt * force_mag / total_mass;
    -dt * force_mag / (total_mass * temp);;
]
Q = diagm(Float32[1, 10, 0.1, 1])
R = diagm(Float32[0.1])
P, _, _ = ared(A, B, R, Q)
K = inv(R + B' * P * B) * (B' * P * A)

pi_model = Chain(
    Dense(-K),
    # max(u, u_low) = relu(x - u_low) + u_low
    Dense(Matrix{Float32}(I(u_dim)), -u_low, relu),
    Dense(Matrix{Float32}(I(u_dim)), u_low),
    # min(u, u_high) = -max(-x, -u_high) = -relu(-x + u_high) + u_high
    Dense(Matrix{Float32}(-I(u_dim)), u_high, relu),
    Dense(Matrix{Float32}(-I(u_dim)), u_high),
)

end
