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
        pk.HWPRotStage(settings[0], wires=dev.wires[0])
        pk.QWPRotStage(settings[1], wires=dev.wires[0])

        return qml.expval(qml.Projector([1], wires=dev.wires[0]))

    init_settings = qnp.array([qnp.pi / 4, qnp.pi / 2], requires_grad=True)
    opt_meta_kwargs = {"test_key": "test_val"}

    opt_dict = pk.optimize(
        cost,
        init_settings,
        num_steps=10,
        step_size=0.2,
        filepath=tmp_dir,
        filename=filename,
        init_opt_dict=None,
        **opt_meta_kwargs,
    )

    print(opt_dict["cost_vals"])
    print(opt_dict["settings_history"])

    saved_filename = tmp_dir + filename + "_" + opt_dict["datetime"] + ".json"
    assert os.path.isfile(saved_filename)
    assert opt_dict == pk.read_optimization_json(saved_filename)

    assert len(opt_dict["cost_vals"]) == 11
    assert qnp.allclose(
        opt_dict["cost_vals"],
        [0.9882, 0.9914, 0.9901, 0.9787, 0.9968, 0.9642, 0.7643, 0.1291, 0.0739, 0.0404, 0.024],
        atol=1e-3,
    )
    assert len(opt_dict["settings_history"]) == 11

    match_settings_history = [
        [0.7853981633974483, 1.5707963267948966],
        [0.783927869024933, 1.5714626009648762],
        [0.7792217672943804, 1.574007525205368],
        [0.7646346605131938, 1.580361007169212],
        [0.7216651273251361, 1.60841369836267],
        [0.5927014520014546, 1.683000002118578],
        [0.26498664969405106, 1.8693658599829543],
        [0.1176365277609629, 1.8445056161167779],
        [0.14340089284235247, 1.7313759104892932],
        [0.047299816364969216, 1.717640002118578],
        [0.08620258765069447, 1.642377575711707],
    ]
    assert qnp.all(
        [
            qnp.allclose(opt_dict["settings_history"][i], match_settings_history[i], atol=1e-3)
            for i in range(11)
        ]
    )

    assert opt_dict["num_steps"] == 10
    assert opt_dict["step_size"] == 0.2
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
