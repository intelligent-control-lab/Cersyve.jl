using SafeTestsets

@safetestset "Double Integrator" begin include("double_integrator_test.jl") end
@safetestset "Pendulum" begin include("pendulum_test.jl") end
@safetestset "Unicycle" begin include("unicycle_test.jl") end
@safetestset "Lane Keep" begin include("lane_keep_test.jl") end
@safetestset "Quadrotor" begin include("quadrotor_test.jl") end
@safetestset "Cart Pole" begin include("cart_pole_test.jl") end
@safetestset "Robot Arm" begin include("robot_arm_test.jl") end

@safetestset "model" begin include("model_test.jl") end
