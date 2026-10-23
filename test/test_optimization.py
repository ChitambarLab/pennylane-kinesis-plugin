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
    @pytest.mark.parametrize("optimizer", ["adam", "gradient-descent"])
    def test_simple_optimization(self, tmp_dir_cleanup, cost, optimizer):
        # tolerance for meaningful cost improvement over the course of optimization
        DIFF_TOL = 0.3

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

        print(opt_dict["cost_vals"])
        assert len(opt_dict["cost_vals"]) == 11
        # cost_vals are shot-based estimates of qml.expval(qml.Projector(...)), a probability
        assert all(-1e-6 <= v <= 1 + 1e-6 for v in opt_dict["cost_vals"])

        # optimization should make significant progress from the initial cost
        assert (opt_dict["cost_vals"][0] - min(opt_dict["cost_vals"])) > DIFF_TOL

        print(opt_dict["settings_history"])
        assert len(opt_dict["settings_history"]) == 11
        # settings should have actually moved away from the initial point
        assert not qnp.allclose(opt_dict["settings_history"][-1], opt_dict["settings_history"][0])

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
        dev2 = qml.device(
            "kinesis.simulator", shots=100, wires=[0], seed=2, background_counts=[0, 0]
        )

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

        # tolerance for meaningful cost movement over the course of optimization
        DIFF_TOL = 0.3

        assert len(opt_dict["cost_vals"]) == 11
        # eval_cost's cost_vals are shot-based probability estimates
        assert all(-1e-6 <= v <= 1 + 1e-6 for v in opt_dict["cost_vals"])

        # `cost` is optimized to minimize the projection onto |1>, so eval_cost's projection
        # onto |proj_state> should trend toward 0 if proj_state == 1, and toward 1 otherwise.
        if proj_state:
            assert (opt_dict["cost_vals"][0] - opt_dict["cost_vals"][-1]) > DIFF_TOL
        else:
            assert (opt_dict["cost_vals"][-1] - opt_dict["cost_vals"][0]) > DIFF_TOL

        assert len(opt_dict["settings_history"]) == 11
        assert not qnp.allclose(opt_dict["settings_history"][-1], opt_dict["settings_history"][0])


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
        # z_grid holds shot-based probability estimates of qml.expval(qml.Projector(...))
        z_grid = np.array(data_dict["z_grid"])
        assert z_grid.shape == (4, 3)
        assert np.all((-1e-6 <= z_grid) & (z_grid <= 1 + 1e-6))
        # sanity check that the scan produced varying, non-degenerate cost values
        assert np.std(z_grid) > 0.1
        assert data_dict["x_curr"] == 2
        assert data_dict["y_curr"] == 3
        assert data_dict["grid_shape"] == [4, 3]
