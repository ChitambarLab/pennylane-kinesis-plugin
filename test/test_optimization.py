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
    @pytest.mark.parametrize(
        "use_adam, match_cost, match_settings",
        [
            (
                True,
                [
                    0.9982,
                    0.6635,
                    0.3129,
                    0.2395,
                    0.3014,
                    0.3144,
                    0.2668,
                    0.1452,
                    0.0358,
                    0.0335,
                    0.0992,
                ],
                [
                    [0.7853981633974483, 1.5707963267948966],
                    [0.5854037187986871, 1.770793821341018],
                    [0.4366324074100085, 1.9203652166497454],
                    [0.26780170935407366, 2.07182869899535],
                    [0.12545675012125146, 2.0997464481972883],
                    [0.03265352738003979, 2.0530974719011876],
                    [-0.005384497495539438, 1.9655938324263846],
                    [0.004345840316356155, 1.8514407766898235],
                    [0.043808777371450384, 1.7207149587236892],
                    [0.08694478602871161, 1.5892646248052025],
                    [0.10354706666355949, 1.483587108471009],
                ],
            ),
            (
                False,
                [0.9982, 0.9899, 0.9912, 1.0, 1.0, 0.9277, 0.5185, 0.1515, 0.081, 0.0502, 0.0362],
                [
                    [0.7853981633974483, 1.5707963267948966],
                    [0.7846781633974482, 1.5723928239106422],
                    [0.7821981633974482, 1.5734756666231169],
                    [0.7759981633974482, 1.5793984797496632],
                    [0.7548781633974482, 1.592551961713507],
                    [0.6867581633974482, 1.6338829876845515],
                    [0.48875816339744815, 1.7522518607008295],
                    [0.12215816339744806, 1.9429951197189075],
                    [0.19371816339744807, 1.776774530973877],
                    [0.06503816339744803, 1.765957962431382],
                    [0.11471816339744803, 1.6672645814802156],
                ],
            ),
        ],
    )
    def test_simple_optimization(self, tmp_dir_cleanup, cost, use_adam, match_cost, match_settings):
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
            use_adam=use_adam,
            verbose=False,
            **opt_meta_kwargs,
        )

        saved_filename = tmp_dir + filename + ".json"
        assert os.path.isfile(saved_filename)
        assert opt_dict == pk.read_json(saved_filename)

        print(opt_dict["cost_vals"])
        assert len(opt_dict["cost_vals"]) == 11
        assert qnp.allclose(
            opt_dict["cost_vals"],
            match_cost,
            atol=1e-3,
        )
        assert len(opt_dict["settings_history"]) == 11
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
        assert opt_dict["use_adam"] == use_adam

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
                [0.01, 0.5202, 0.9969],
                [0.1663, 0.2826, 0.8581],
                [0.4345, 0.3364, 0.5865],
                [0.4849, 0.6153, 0.5231],
            ],
            atol=1e-3,
        )
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]
