module RobotDog

using EllipsisNotation

# state: [v, xg, yg, xo, yo]
x_dim = 5
x_low = Float32[0, -2, -2, -2, -2]
x_high = Float32[2, 2, 2, 2, 2]

dynamics_path = joinpath(@__DIR__, "../model/robot_dog/f.onnx")

function constraint(x::Array{Float32})::Array{Float32}
    xo, yo = x[4, ..], x[5, ..]
    h = 0.4 .- sqrt.(xo .^ 2 + yo .^ 2)
    return reshape(h, 1, :)
end

end
