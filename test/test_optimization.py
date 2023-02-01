import pytest
import pennylane as qml
from pennylane import numpy as qnp
import numpy as np
import os
import re

import pennylane_kinesis as pk

from test_file_utilities import tmp_dir_cleanup


def test_simple_optimization(tmp_dir_cleanup):
    tmp_dir = pk.tmp_dir("./test/")
    filename = "test_simple_optimization"

    dev = qml.device("kinesis.simulator", shots=10000, wires=[0], seed=2)

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(settings):
        qml.RY(-settings[0], wires=dev.wires[0])
        qml.PauliZ(dev.wires[0])
        qml.RY(settings[0], wires=dev.wires[0])

        qml.RY(-settings[1], wires=dev.wires[0])
        qml.S(dev.wires[0])
        qml.RY(settings[1], wires=dev.wires[0])

        return qml.expval(qml.Projector([1], wires=dev.wires[0]))

    init_settings = qnp.array([qnp.pi / 4, qnp.pi / 2], requires_grad=True)
    opt_meta_kwargs = {"test_key": "test_val"}

    opt_dict = pk.optimize(
        cost,
        init_settings,
        num_steps=12,
        step_size=0.6,
        filepath=tmp_dir,
        filename=filename,
        init_opt_dict=None,
        **opt_meta_kwargs,
    )

    saved_filename = tmp_dir + filename + "_" + opt_dict["datetime"] + ".json"
    assert os.path.isfile(saved_filename)
    assert opt_dict == pk.read_optimization_json(saved_filename)

    assert len(opt_dict["cost_vals"]) == 13
    assert qnp.allclose(
        opt_dict["cost_vals"],
        [
            0.4987,
            0.501,
            0.3659,
            0.2698,
            0.1878,
            0.133,
            0.0807,
            0.0535,
            0.0286,
            0.0162,
            0.0112,
            0.006,
            0.0031,
        ],
        atol=1e-3,
    )
    assert len(opt_dict["settings_history"]) == 13

    match_settings_history = [
        [0.7853, 1.5707],
        [0.7889, 1.2702],
        [0.7368, 1.0211],
        [0.5935, 0.8604],
        [0.4685, 0.7057],
        [0.3623, 0.5690],
        [0.2789, 0.4516],
        [0.2207, 0.3539],
        [0.1698, 0.2736],
        [0.1297, 0.2073],
        [0.1035, 0.1569],
        [0.0720, 0.1231],
        [0.0575, 0.1023],
    ]
    assert qnp.all(
        [
            qnp.allclose(opt_dict["settings_history"][i], match_settings_history[i], atol=1e-3)
            for i in range(13)
        ]
    )

    assert opt_dict["num_steps"] == 12
    assert opt_dict["step_size"] == 0.6
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", opt_dict["datetime"])
    assert opt_dict["test_key"] == "test_val"


def test_optimization_error(tmp_dir_cleanup):
    tmp_dir = pk.tmp_dir("./test/")
    filename = "test_optimization_error"

    dev = qml.device("kinesis.simulator", shots=1000, wires=[0])

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(settings):
        if cost.call_count:
            raise Exception("This is an expected test error.")
        else:
            cost.call_count += 1
            return qml.expval(qml.Identity(dev.wires[0]))

    cost.call_count = 0

    init_settings = qnp.array([0.1, 0.2], requires_grad=True)

    qnp.random.seed(22)
    opt_dict = pk.optimize(
        cost,
        init_settings,
        num_steps=12,
        step_size=0.6,
        filepath=tmp_dir,
        filename=filename,
        init_opt_dict=None,
    )

    saved_filename = tmp_dir + "tmp/" + filename + "_" + opt_dict["datetime"] + ".json"
    assert os.path.isfile(saved_filename)
    assert opt_dict == pk.read_optimization_json(saved_filename)

    assert len(opt_dict["cost_vals"]) == 1
    assert opt_dict["cost_vals"] == [1.0]
    assert len(opt_dict["settings_history"]) == 1

    assert opt_dict["settings_history"][0] == [0.1, 0.2]

    assert opt_dict["num_steps"] == 12
    assert opt_dict["step_size"] == 0.6
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", opt_dict["datetime"])
