module LaneKeep

using Flux
using LinearAlgebra
using MatrixEquations

# state: [y, phi, v, w]
x_dim = 4
x_high = Float32[2.0, pi / 3, 1.0, 1.0]
x_low = -x_high

u_dim = 1
u_high = Float32[pi / 6]
u_low = -u_high

y_con = 1.8
phi_con = pi / 4

k1 = -12000.0  # front wheel cornering stiffness [N/rad]
k2 = -8000.0   # rear wheel cornering stiffness [N/rad]
a  = 1.6        # distance from CG to front axle [m]
b  = 1.1        # distance from CG to rear axle [m]
m  = 1200.0    # mass [kg]
Iz = 1500.0    # polar moment of inertia at CG [kg*m^2]
u  = 10.0      # longitudinal speed [m/s]
dt = 0.1

A = Float32[
    1 u * dt dt 0;
    0 1 0 dt;
    0 0 1 + (k1 + k2) / (m * u) * dt ((a * k1 - b * k2) / (m * u) - u) * dt;
    0 0 (a * k1 - b * k2) / (Iz * u) * dt 1 + (k1 * a ^ 2 + k2 * b ^ 2) / (Iz * u) * dt;
]
B = Float32[
    0;
    0;
    -k1 / m * dt;
    -a * k1 / Iz * dt;;
]
Q = diagm(Float32[1, 10, 0.1, 1])
R = diagm(Float32[0.1])
P, _, _ = ared(A, B, R, Q)
K = inv(R + B' * P * B) * (B' * P * A)

h_y_model = Chain(
    Dense(Float32[1 0 0 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-y_con]),
    )
)
h_phi_model = Chain(
    Dense(Float32[0 1 0 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-phi_con]),
    )
)
h_model = Parallel(+,
    Chain(
        Parallel(+,
            h_y_model,
            Chain(
                h_phi_model,
                Dense(-Matrix{Float32}(I(1))),
            ),
        ),
        Dense(Matrix{Float32}(I(1)), Float32[0], relu),
    ),
    h_phi_model,
)

pi_model = Chain(
    Dense(-K),
    # max(u, u_low) = relu(x - u_low) + u_low
    Dense(Matrix{Float32}(I(u_dim)), -u_low, relu),
    Dense(Matrix{Float32}(I(u_dim)), u_low),
    # min(u, u_high) = -max(-x, -u_high) = -relu(-x + u_high) + u_high
    Dense(Matrix{Float32}(-I(u_dim)), u_high, relu),
    Dense(Matrix{Float32}(-I(u_dim)), u_high),
)

f_pi_model = Chain(
    # x' = Ax + Bu
    Parallel(+,
        Dense(A),
        Chain(pi_model, Dense(B)),
    ),
    # max(x, x_low) = relu(x - x_low) + x_low
    Dense(Matrix{Float32}(I(x_dim)), -x_low, relu),
    Dense(Matrix{Float32}(I(x_dim)), x_low),
    # min(x, x_high) = -max(-x, -x_high) = -relu(-x + x_high) + x_high
    Dense(-Matrix{Float32}(I(x_dim)), x_high, relu),
    Dense(-Matrix{Float32}(I(x_dim)), x_high),
)

end
