using Cersyve
using Test

task = Unicycle

function policy(x::Matrix{Float32})::Matrix{Float32}
    return stack((ones(size(x, 2)), -x[end, :]); dims=1)
end

x = Cersyve.uniform(task.x_low, task.x_high, 10)
@test isapprox(task.pi_model(x), policy(x))
