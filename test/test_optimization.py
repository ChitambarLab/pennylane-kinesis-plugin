import pytest
import pennylane as qml
from pennylane import numpy as qnp
import numpy as np
import os
import re

import pennylane_kinesis as pk

from test_file_utilities import tmp_dir_cleanup


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
    def test_simple_optimization(self, tmp_dir_cleanup, cost):
        tmp_dir = pk.tmp_dir("./test/")
        filename = "test_simple_optimization"

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

        saved_filename = tmp_dir + filename + ".json"
        assert os.path.isfile(saved_filename)
        assert opt_dict == pk.read_json(saved_filename)

        assert len(opt_dict["cost_vals"]) == 11
        assert qnp.allclose(
            opt_dict["cost_vals"],
            [0.9882, 1.0, 0.9991, 0.9806, 0.9827, 0.8373, 0.2494, 0.1316, 0.0657, 0.0436, 0.0307],
            atol=1e-3,
        )
        assert len(opt_dict["settings_history"]) == 11

        match_settings_history = [
            [0.7853981633974483, 1.5707963267948966],
            [0.7846498791262008, 1.5656985807623407],
            [0.787007869024933, 1.5644428448310528],
            [0.7990493618079196, 1.5581442661872902],
            [0.8305673517066519, 1.5361498087587406],
            [0.9290580623847705, 1.4766202459847546],
            [1.2016180623847705, 1.31313740327228],
            [1.4841475955728283, 1.2267822560860224],
            [1.3822712708965097, 1.39392301727048],
            [1.5297022254408048, 1.3851409357424629],
            [1.451780215339537, 1.4936719912994356],
        ]
        print(opt_dict["settings_history"])
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
                [0.0, 0.5102, 0.9869],
                [0.1564, 0.2727, 0.8481],
                [0.4246, 0.3265, 0.5765],
                [0.475, 0.6053, 0.5131],
            ],
            atol=1e-3,
        )
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]
