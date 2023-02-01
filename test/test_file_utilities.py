import pytest
import re
import os
import shutil

import pennylane_kinesis as pk


def test_datetime_now_string():
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", pk.datetime_now_string())


@pytest.fixture()
def tmp_dir_cleanup():
    tmp_dir = "./test/tmp/"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    assert not os.path.exists(tmp_dir)

    yield

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    assert not os.path.exists(tmp_dir)


def test_tmp_dir(tmp_dir_cleanup):
    assert not os.path.exists("./test/tmp/")

    tmp_dir = pk.tmp_dir("./test/")

    assert os.path.exists("./test/tmp/")
    assert tmp_dir == "./test/tmp/"


@pytest.mark.parametrize(
    "opt_dict",
    [
        {
            "cost_vals": [0.0, 1.0, 1.2],
            "settings_history": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
            "step_size": 0.1,
            "datetime": "2023-01-31T12-12-12Z",
            "num_steps": 2,
            "optimizer": "GradientDescentOptimizer",
        },
        {"test_key": "val", "other_key": 123456789},
    ],
)
def test_read_write_optimization_json(tmp_dir_cleanup, opt_dict):
    tmp_dir = pk.tmp_dir("./test/")
    filename = "test_read_write_optimization_json"

    pk.write_optimization_json(opt_dict, tmp_dir + filename)

    assert os.path.isfile(tmp_dir + filename + ".json")

    read_opt_dict = pk.read_optimization_json(tmp_dir + filename + ".json")

    assert read_opt_dict == opt_dict
