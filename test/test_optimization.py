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
                    1.0,
                    0.9244,
                    0.6742,
                    0.2465,
                    0.0359,
                    0.1969,
                    0.2982,
                    0.2634,
                    0.1358,
                    0.0171,
                    0.0181,
                ],
                [
                    [0.7853981633974483, 1.5707963267948966],
                    [0.5856002190585581, 1.3710782029769721],
                    [0.43708985865593525, 1.4784722461904805],
                    [0.2768870560476778, 1.6056859608738832],
                    [0.10188258917158663, 1.7610291486005583],
                    [-0.04843905015343425, 1.8568052809023832],
                    [-0.12432238818350479, 1.8571773274830923],
                    [-0.13687668929546332, 1.8034728913561724],
                    [-0.10471606359077776, 1.7152472166054609],
                    [-0.045493441313964836, 1.6065004856342746],
                    [0.019754616953376977, 1.4967612578256366],
                ],
            ),
            (
                False,
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999, 0.9990, 0.9900, 0.9058, 0.4007],
                [
                    [0.7853981633974483, 1.5707963267948966],
                    [0.7853783868721838, 1.5707821561670452],
                    [0.7853392364351349, 1.5708046372247755],
                    [0.7852404157903629, 1.5708370227507995],
                    [0.7849630502247159, 1.5710012757129224],
                    [0.7840924958558755, 1.5715309569194567],
                    [0.7813950175701141, 1.5731528943549011],
                    [0.7730778481011844, 1.5782125051292664],
                    [0.7474539466302413, 1.5939550143008228],
                    [0.6687834896641971, 1.6423276219236902],
                    [0.44210574338727193, 1.7790608572176059],
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

        print(opt_dict["settings_history"])
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
                [0.0, 0.5146, 0.9992],
                [0.1594, 0.2770, 0.8537],
                [0.4343, 0.3331, 0.5754],
                [0.4742, 0.6114, 0.5193],
            ],
            atol=1e-3,
        )
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]
