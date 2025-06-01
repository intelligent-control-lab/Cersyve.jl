from math import pi


# tasks
DoubleIntegrator = 'double_integrator'
Pendulum = 'pendulum'
Unicycle = 'unicycle'
LaneKeep = 'lane_keep'
Quadrotor = 'quadrotor'
CartPole = 'cart_pole'
PointMass = 'point_mass'
RobotArm = 'robot_arm'
RobotDog = 'robot_dog'


# input bounds
input_bound = {
    DoubleIntegrator: [[-1, -1], [1, 1]],
    Pendulum: [[-pi / 4, -4], [pi / 4, 4]],
    Unicycle: [[-1, -1, -1], [1, 1, 1]],
    LaneKeep: [[-2, -pi / 3, -1, -1], [2, pi / 3, 1, 1]],
    Quadrotor: [[-1, -pi / 6, -1, -1], [1, pi / 6, 1, 1]],
    CartPole: [[-1, -24 * pi / 180, -1, -1], [1, 24 * pi / 180, 1, 1]],
    PointMass: [[-1, -1, -1, -1], [1, 1, 1, 1]],
    RobotArm: [[pi / 3, -pi / 6, -pi / 6, -1, -1, -1], [pi / 2, pi / 6, pi / 6, 1, 1, 1]],
    RobotDog: [[0, -2, -2, -2, -2], [2, 2, 2, 2, 2]],
}


def write_prop(task: str, f):
    f.write(f'; Cersyve-9 property {task}\n\n')

    # declare constants
    lb, ub = input_bound[task]
    for i in range(len(lb)):
        f.write(f'(declare-const X_{i} Real)\n')
    f.write('\n')
    for i in range(2):
        f.write(f'(declare-const Y_{i} Real)\n')
    f.write('\n')

    # input constraints
    for i in range(len(lb)):
        f.write(f'(assert (<= X_{i} {ub[i]}))\n')
        f.write(f'(assert (>= X_{i} {lb[i]}))\n')
    f.write('\n')

    # output constraints
    f.write('(assert (<= Y_0 0))\n')
    f.write('(assert (>= Y_1 0))\n')


def main():
    # generate vnnlib
    for task in input_bound.keys():
        with open(f'vnnlib/prop_{task}.vnnlib', 'w') as f:
            write_prop(task, f)

    # generate csv
    with open('instances.csv', 'w') as f:
        for task in input_bound.keys():
            for version in ['pretrain', 'finetune']:
                for cond in ['con', 'inv']:
                    onnx = f'onnx/{task}_{version}_{cond}.onnx'
                    prop = f'vnnlib/prop_{task}.vnnlib'
                    f.write(f'{onnx},{prop},{100}\n')


if __name__ == '__main__':
    main()
