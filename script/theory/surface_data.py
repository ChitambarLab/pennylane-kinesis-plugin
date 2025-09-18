import numpy as np
import numpy.linalg as la
import plotly.graph_objects as go
from pennylane_kinesis import KinesisSimulatorDevice
from pennylane_kinesis import (
    ElliptecRotatorDevice,
    ElliptecMotors,
    make_dir,
    datetime_now_string,
)
import json
import qnetvo
from pennylane_kinesis.operators import HWPRotStage, QWPRotStage
import pennylane as qml


class Operation:
    def __init__(self, name, parameters=[]):
        self.name = name
        self.parameters = parameters

    def __str__(self):
        return f"{self.name}({self.parameters})"

    def __repr__(self):
        return self.__str__()


aHWP_op = Operation("HWPRotStage")
aQWP_op = Operation("QWPRotStage")
bHWP_op = Operation("HWPRotStage")
bQWP_op = Operation("QWPRotStage")

json_file = "./data/two-state_discrimination/optimization_2024-11-15T21-26-24Z.json"
json_2 = "./data/two-state_discrimination/optimization_2024-11-16T00-40-18Z.json"

numerics = True
# device = KinesisSimulatorDevice(wires=4,shots=10000,background_counts=400,seed=1234)
shots = 10000
background_counts = 400
qwp_period = np.pi
hwp_period = np.pi / 2
test_mode = False
diff_method = "parameter-shift"
eval_num = 0


two_wp = False
filepath = "./data/surface_collection/"

fixed_meas_settings = [0, 0]
data_path = make_dir(filepath)


def arb_unitary(settings, wires):
    HWPRotStage(settings[0], wires=[wires[0]])
    QWPRotStage(settings[1], wires=[wires[0]])


def arb_rot(settings, wires):
    qml.Rot(settings[0], settings[1], settings[2], wires=[wires[0]])


# Function to use for MeasureNode if we are not moving the measurement waveplates.
def meas_fn(settings, wires):
    HWPRotStage(settings[0], wires=wires[0:1])
    QWPRotStage(settings[1], wires=wires[0:1])


def noise_fn(settings, wires):
    kraus_ops = np.array(
        np.array(
            [
                [
                    [0.0 + 0.0749752072j, -0.1032453648 + 0.2310816375j],
                    [0.1029833616 - 0.2309243091j, 0.0546661044 - 0.0500401343j],
                ],
                [
                    [-0.3030531991 + 0.0j, -0.0216678152 + 0.0362852244j],
                    [-0.0323707284 + 0.0524531731j, -0.1515895127 - 0.2620140203j],
                ],
                [
                    [-0.0730095495 + 0.0j, 0.2646422792 - 0.293498816j],
                    [0.2543979443 - 0.2870121043j, -0.0081668458 - 0.0622479936j],
                ],
                [
                    [-0.4051025135 + 0.0j, 0.086442595 + 0.0585767426j],
                    [-0.1170769182 - 0.0643395223j, 0.1932879045 + 0.3592735014j],
                ],
            ]
        ),
        dtype=complex,
    )
    qml.QubitChannel(kraus_ops, wires=wires[0])


if numerics:
    data_path += "numerics/"
    dev_name = "kinesis.simulator"
    additional_dev_kwargs = {
        # "shot_rate": shot_rate,
        "shots": shots,
        "background_counts": background_counts,
        "qwp_domain_period": qwp_period,
        "hwp_domain_period": hwp_period,
    }
else:
    motors = ElliptecMotors(test_mode=test_mode)
    data_path += "hardware/"
    dev_name = "elliptec.rotator"
    additional_dev_kwargs = {
        "motors": motors,
        "test_mode": test_mode,
        "filepath": filepath,
    }

make_dir(data_path)

meas_node = None
if two_wp:
    data_path += "two_wp/"
    meas_node = qnetvo.MeasureNode(
        num_in=1,
        num_out=2,
        wires=[0],
        ansatz_fn=lambda *args: meas_fn(fixed_meas_settings, args[1]),
        num_settings=0,
    )

else:
    data_path += "four_wp/"
    meas_node = qnetvo.MeasureNode(num_in=1, wires=[0], ansatz_fn=arb_unitary, num_settings=2)

make_dir(data_path)

datetime_str = datetime_now_string()

# network ansatz class manages circuit construction and settings
state_disc_ansatz = qnetvo.NetworkAnsatz(
    [qnetvo.PrepareNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
    # [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
    [meas_node],
    dev_kwargs={
        "name": dev_name,
        # "shot_rate": shot_rate,
        "shots": shots,
        **additional_dev_kwargs,
    },
)

device = qml.device(**state_disc_ansatz.dev_kwargs)
print(device.shot_rate)
shot_rate = device.shot_rate.tolist()
background_rate = device.take_background()

# construct a linear inequality cost function that
# corresponds to the minimal qubit dimensionality witness.
qnode_kwargs = {"diff_method": diff_method}
if diff_method == "finite-diff":
    qnode_kwargs["h"] = 1e-3
    qnode_kwargs["strategy"] = "center"
    qnode_kwargs["approx_order"] = 2
state_disc_cost = qnetvo.linear_probs_cost_fn(state_disc_ansatz, np.eye(2), qnode_kwargs=qnode_kwargs)

# loss = lambda settings: state_disc_cost(*settings)

eval_num = 0


def loss(settings):
    global eval_num
    eval_num += 1
    return state_disc_cost(*settings)


# print(np.array(state_disc_ansatz.rand_network_settings()))
# def loss(x):
#     # put your loss function here
#     a_hwp, a_qwp, b_hwp, b_qwp = x
#     aHWP_op.parameters = [a_hwp]
#     aQWP_op.parameters = [a_qwp]
#     bHWP_op.parameters = [b_hwp]
#     bQWP_op.parameters = [b_qwp]
#     device.apply([aHWP_op, aQWP_op, bHWP_op, bQWP_op])
#     h_counts = device.shots - np.sum(device.generate_samples())
#     return h_counts/device.shots

# def loss2(x):
#     return 1-loss(x)


# eval_num = 0
# def loss_6(x):
#     global eval_num
#     eval_num += 1
#     device.reset()
#     a_0hwp, a_0qwp, a_1hwp, a_1qwp, b_hwp, b_qwp = x
#     aHWP_op.parameters = [a_0hwp]
#     aQWP_op.parameters = [a_0qwp]
#     bHWP_op.parameters = [b_hwp]
#     bQWP_op.parameters = [b_qwp]
#     device.apply([aHWP_op, aQWP_op, bHWP_op, bQWP_op])
#     v_counts = np.sum(device.generate_samples())
#     h_counts = device.shots - v_counts
#     device.reset()

#     # print(v_counts, h_counts)
#     row_0 = np.array([h_counts, v_counts])/device.shots

#     aHWP_op.parameters = [a_1hwp]
#     aQWP_op.parameters = [a_1qwp]
#     device.apply([aHWP_op, aQWP_op, bHWP_op, bQWP_op])
#     v_counts = np.sum(device.generate_samples())
#     h_counts = device.shots - v_counts
#     # print(v_counts, h_counts)

#     row_1 = np.array([h_counts, v_counts])/device.shots

#     behavior_matrix = np.vstack([row_0, row_1])

#     game = np.eye(2)
#     score = 0
#     for i in range(game.shape[0]):
#         score += np.sum(game[:, i] * behavior_matrix[i])
#     return -score


def plot_loss_landscape(loss, directions, steps, res=25, scale=None):
    assert directions.shape[0] == 2
    assert directions.shape[1] == steps.shape[1]
    N = directions.shape[1]

    mean = np.mean(steps, axis=0)
    Q = la.qr(directions.T)[0]

    if scale is None:
        distances = la.norm(Q.T @ (steps - mean).T, axis=0)
        scale = np.max(distances) + 0.1

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ll = np.zeros_like(xx)

    # Evaluate loss on grid
    for i, _x in enumerate(x):
        for j, _y in enumerate(y):
            xy = np.array([_x, _y])
            # print(mean+Q@xy*scale)
            ll[i, j] = loss(mean + Q @ xy * scale)

    params_x = []
    params_y = []
    params_z = []

    for step in steps:
        # Project each step onto principal directions
        proj = Q.T @ ((step - mean).flatten())
        params_x.append(proj[0].item() / scale)
        params_y.append(proj[1].item() / scale)
        params_z.append(loss(step).item())

    data = [
        go.Surface(z=ll, x=xx, y=yy, opacity=0.95),
        go.Scatter3d(x=params_x, y=params_y, z=params_z, mode="markers+lines", marker={"size": 1}),
    ]

    print("plotting")
    fig = go.Figure(data=data)
    fig.update_layout(width=800, height=800)
    fig.write_html(data_path + "plot_11-18.html", auto_open=False)
    print(np.unravel_index(ll.argmin(), ll.shape))
    argmin_idx = np.unravel_index(ll.argmin(), ll.shape)
    argmin = [np.array(x[argmin_idx[0]]), np.array(y[argmin_idx[1]])]
    # argmin = [np.array(x[8]), np.array(y[3])]
    print(ll.min())
    global_min = mean + Q @ argmin * scale
    print("Loss at global min:", loss(global_min))
    print("Global minimum:", global_min)
    a0hwp_angle = global_min[0] % (np.pi / 2)
    a0qwp_angle = global_min[1] % (np.pi)
    print("prep 0 angles", a0hwp_angle, a0qwp_angle)
    a1hwp_angle = global_min[2] % (np.pi / 2)
    a1qwp_angle = global_min[3] % (np.pi)
    print("prep 1 angles", a1hwp_angle, a1qwp_angle)
    bhwp_angle = global_min[2] % (np.pi / 2)
    bqwp_angle = global_min[3] % (np.pi)
    print("meas angles", bhwp_angle, bqwp_angle)
    print("optimization min:", params_z[-1])
    print("Optimization angles:", mean + Q @ [params_x[-1], params_y[-1]] * scale)
    print(
        "loss at optimization min:",
        loss(mean + Q @ [params_x[-1], params_y[-1]] * scale),
    )
    # print(params_z[-1])
    # print(mean+Q@[params_x[-1],params_y[-1]]*scale, loss(mean+Q@[params_x[-1],params_y[-1]]*scale))
    # final_params = mean+Q@[params_x[-1],params_y[-1]]*scale
    # print(loss(final_params))
    # print(loss2(final_params))
    # print(-2*(loss(final_params)+loss2(final_params)))
    # print(params_z[-1])


# get rid of this junk
alpha = np.linspace(0, 1, 1001)[:-1]

finish = np.zeros(10)
start = np.ones(10)

with open(json_file, "r") as f:
    data = json.load(f)

settings = np.array(data["settings_history"])


last_settings = settings[-1]

# print(settings.shape)
# settings = np.hstack([settings[:,0:2],settings[:,4:6]])
# print(settings)
# print(settings.shape)

# param_history should be the history of your parameter data during training of size (N, D)
# where N is the size of your training history and D is the dimensionality of your optimization problem
# row i should contain parameter data at training iteration i
# param_history = start.reshape(1, -1) * (1-alpha[:, None]) + finish.reshape(1, -1) * alpha[:, None]
# print(param_history.shape)

param_history = settings
# # # PCA on training history
X = param_history - np.mean(param_history, axis=0)
U, S, Vt = la.svd(X, full_matrices=False)
plot_loss_landscape(loss, Vt[:2], param_history, res=50)
print(eval_num)
