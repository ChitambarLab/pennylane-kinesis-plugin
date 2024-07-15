import pytest
import pennylane as qml
from pennylane import numpy as qnp
import numpy as np
import os
import re

import pennylane_kinesis as pk

from test_file_utilities import tmp_dir_cleanup
import logging


@pytest.fixture
def cost():
    dev = qml.device("kinesis.simulator", shots=10000, wires=[0], seed=2)

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(settings):
        pk.HWPRotStage(settings[0], wires=dev.wires[0])
        pk.QWPRotStage(settings[1], wires=dev.wires[0])

        return qml.expval(qml.Projector([1], wires=dev.wires[0]))

    return cost


class TestOptimization:
    @pytest.mark.parametrize(
        "optimizer, match_cost, match_settings",
        [
            (
                "adam",
                [
                    0.9949,
                    0.9199,
                    0.7902,
                    0.6252,
                    0.3604,
                    0.0916,
                    0.1769,
                    0.3394,
                    0.3136,
                    0.1818,
                    0.0731,
                ],
                [
                    [0.7853981633974483, 1.5707963267948966],
                    [0.9853781653972483, 1.7706442018180257],
                    [1.1340576049052722, 1.9296860968223613],
                    [1.305006434288613, 2.057280010509621],
                    [1.4867028239939348, 1.9770966655900302],
                    [1.6743478780555308, 1.860118621578505],
                    [1.849064667215512, 1.7176904634316918],
                    [1.9166516493951529, 1.6437326256861469],
                    [1.9041270360972058, 1.6470577551123509],
                    [1.8450567428133389, 1.6968222171496175],
                    [1.760134578089353, 1.768231909231275],
                ],
            ),
            (
                "gradient-descent",
                [
                    0.9949,
                    0.9949,
                    0.9945,
                    0.9947,
                    0.9943,
                    0.9913,
                    0.9697,
                    0.7827,
                    0.1535,
                    0.0902,
                    0.0523,
                ],
                [
                    [0.7853981633974483, 1.5707963267948966],
                    [0.7855981633974483, 1.5708226009648762],
                    [0.7862381633974482, 1.5707815959142424],
                    [0.7876781633974482, 1.5700108852361236],
                    [0.7919581633974483, 1.567542600964876],
                    [0.8051181633974482, 1.5592197582524014],
                    [0.8453581633974482, 1.5346939003880251],
                    [0.9668381633974482, 1.4604294638798863],
                    [1.2805581633974483, 1.277998336896164],
                    [1.4599981633974484, 1.2802011796086388],
                    [1.4142781633974484, 1.4103494638798861],
                ],
            ),
        ],
    )
    def test_simple_optimization(
        self, tmp_dir_cleanup, cost, optimizer, match_cost, match_settings
    ):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_simple_optimization"

        init_settings = qnp.array([qnp.pi / 4, qnp.pi / 2], requires_grad=True)
        opt_meta_kwargs = {"test_key": "test_val"}
        print("OPTIMIZER IS ", optimizer)
        opt_dict = pk.optimize(
            cost,
            init_settings,
            num_steps=10,
            step_size=0.2,
            filepath=tmp_dir,
            filename=filename,
            init_opt_dict=None,
            optimizer=optimizer,
            verbose=False,
            **opt_meta_kwargs,
        )

        saved_filename = tmp_dir + filename + ".json"
        assert os.path.isfile(saved_filename)
        assert opt_dict == pk.read_json(saved_filename)

        print(opt_dict["settings_history"])
        assert len(opt_dict["cost_vals"]) == 11

        print(opt_dict["cost_vals"])
        assert qnp.allclose(
            opt_dict["cost_vals"],
            match_cost,
            atol=1e-2,
        )
        assert len(opt_dict["settings_history"]) == 11

        print(opt_dict["settings_history"])
        assert qnp.all(
            [
                qnp.allclose(opt_dict["settings_history"][i], match_settings[i], atol=1e-3)
                for i in range(11)
            ]
        )

        assert opt_dict["num_steps"] == 10
        assert opt_dict["step_size"] == 0.2
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", opt_dict["datetime"])
        assert opt_dict["test_key"] == "test_val"
        assert opt_dict["optimizer"] == optimizer

    def test_optimization_error(self, tmp_dir_cleanup):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_optimization_error"

        dev = qml.device("kinesis.simulator", shots=1000, wires=[0])

        @qml.qnode(dev, diff_method="parameter-shift")
        def cost(settings):
            raise Exception("This is an expected test error.")

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

        saved_filename = tmp_dir + "tmp/" + filename + ".json"
        assert os.path.isfile(saved_filename)
        assert opt_dict == pk.read_json(saved_filename)

        assert len(opt_dict["cost_vals"]) == 0
        assert opt_dict["cost_vals"] == []
        assert len(opt_dict["settings_history"]) == 1

        assert opt_dict["settings_history"][0] == [0.1, 0.2]

        assert opt_dict["num_steps"] == 12
        assert opt_dict["step_size"] == 0.6
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", opt_dict["datetime"])

    @pytest.mark.parametrize("proj_state", [1, 0])
    def test_test_cost_argument(self, tmp_dir_cleanup, cost, proj_state):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_test_cost_argument"

        # the eval cost uses fewer shots to distinguish itself from the grad cost
        dev2 = qml.device("kinesis.simulator", shots=100, wires=[0], seed=2, background_counts=0)

        # test the eval cost as projecting on to the 0/1 state
        @qml.qnode(dev2, diff_method="parameter-shift")
        def eval_cost(settings):
            pk.HWPRotStage(settings[0], wires=dev2.wires[0])
            pk.QWPRotStage(settings[1], wires=dev2.wires[0])

            return qml.expval(qml.Projector([proj_state], wires=dev2.wires[0]))

        # init settings shifted slightly to facilitate better optimization results
        init_settings = qnp.array([qnp.pi / 4 - 0.0001, qnp.pi / 2 - 0.0001], requires_grad=True)

        opt_dict = pk.optimize(
            cost,
            init_settings,
            num_steps=10,
            step_size=0.2,
            filepath=tmp_dir,
            filename=filename,
            eval_cost=eval_cost,
            verbose=False,
        )

        match_cost_vals = qnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.96, 0.64, 0.12, 0.07])
        if not proj_state:
            match_cost_vals = 1 - match_cost_vals

        assert np.allclose(opt_dict["cost_vals"], match_cost_vals)
        assert np.allclose(
            opt_dict["settings_history"],
            [
                [0.7852981633974483, 1.5706963267948966],
                [0.7854181633974483, 1.5709431897099067],
                [0.7851781633974483, 1.5713276262180456],
                [0.7846981633974482, 1.5720231897099066],
                [0.7829381633974483, 1.5734407633030354],
                [0.7774581633974482, 1.5777421846592727],
                [0.7597781633974482, 1.5892476262180457],
                [0.7050981633974481, 1.6242690475742831],
                [0.5416581633974481, 1.725166621167412],
                [0.17909816339744794, 1.923070052624917],
                [0.17533816339744795, 1.7963110576755508],
            ],
        )


class TestScanTwoParamFn:
    def test_scan_two_param_fn_simple(self, tmp_dir_cleanup):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_scan_two_param_fn_simple"

        fn = lambda settings: settings[0] * settings[1]

        x_range = list(range(3))
        y_range = list(range(4))

        data_dict = pk.scan_two_param_fn(fn, x_range, y_range, filename=filename, filepath=tmp_dir)

        saved_filename = tmp_dir + filename + ".json"

        assert os.path.isfile(saved_filename)
        assert data_dict == pk.read_json(saved_filename)

        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", data_dict["datetime"])

        assert data_dict["x_grid"] == [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        assert data_dict["y_grid"] == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        assert data_dict["z_grid"] == [[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6]]
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]

    def test_scan_two_param_fn_error(self, tmp_dir_cleanup):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_scan_two_param_fn_error"

        def fn(settings):
            x, y = settings
            if x == 1 and y == 2:
                raise Exception("This is an expected test error.")
            else:
                return x * y

        x_range = list(range(3))
        y_range = list(range(4))

        data_dict = pk.scan_two_param_fn(fn, x_range, y_range, filename=filename, filepath=tmp_dir)

        saved_filename = tmp_dir + "tmp/" + filename + ".json"

        assert os.path.isfile(saved_filename)
        assert data_dict == pk.read_json(saved_filename)

        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", data_dict["datetime"])

        assert data_dict["x_grid"] == [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        assert data_dict["y_grid"] == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        assert data_dict["z_grid"] == [[0, 0, 0], [0, 1, 2], [0, 0, 0], [0, 0, 0]]
        assert data_dict["x_curr"] == 1
        assert data_dict["y_curr"] == 2
        assert data_dict["grid_shape"] == [4, 3]

    def test_scan_two_param_fn_warm_start(self, tmp_dir_cleanup):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_scan_two_param_fn_warm_start"

        fn = lambda settings: settings[0] * settings[1]

        x_range = list(range(3))
        y_range = list(range(4))

        init_dict = {
            "x_grid": [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            "y_grid": [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            "z_grid": [[0, 0, 0], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            "x_curr": 1,
            "y_curr": 2,
            "grid_shape": [4, 3],
            "datetime": pk.datetime_now_string(),
        }

        data_dict = pk.scan_two_param_fn(
            fn, x_range, y_range, filename=filename, filepath=tmp_dir, init_data_dict=init_dict
        )

        saved_filename = tmp_dir + filename + ".json"

        assert os.path.isfile(saved_filename)
        assert data_dict == pk.read_json(saved_filename)

        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", data_dict["datetime"])

        assert data_dict["x_grid"] == [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        assert data_dict["y_grid"] == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        assert data_dict["z_grid"] == [[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6]]
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]

    def test_scan_two_param_fn_realistic_example(self, tmp_dir_cleanup, cost):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_scan_two_param_fn_realistic_example"

        x_range = np.arange(0, 1, 0.4)
        y_range = np.arange(0, 1, 0.3)

        data_dict = pk.scan_two_param_fn(
            cost, x_range, y_range, filename=filename, filepath=tmp_dir
        )

        saved_filename = tmp_dir + filename + ".json"

        assert os.path.isfile(saved_filename)
        assert data_dict == pk.read_json(saved_filename)

        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", data_dict["datetime"])
        assert np.allclose(
            data_dict["x_grid"],
            [[0.0, 0.4, 0.8], [0.0, 0.4, 0.8], [0.0, 0.4, 0.8], [0.0, 0.4, 0.8]],
        )
        assert np.allclose(
            data_dict["y_grid"], [[0, 0, 0], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]]
        )
        assert np.allclose(
            data_dict["z_grid"],
            [
                [0.0048, 0.5148, 0.9949],
                [0.1627, 0.279, 0.8499],
                [0.4365, 0.3347, 0.5745],
                [0.4752, 0.6105, 0.5198],
            ],
            atol=1e-3,
        )
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]
