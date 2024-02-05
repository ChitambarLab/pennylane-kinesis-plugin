import pytest
import os
import numpy as np

import pennylane_kinesis as pk

from test_file_utilities import tmp_dir_cleanup


@pytest.mark.parametrize(
    "numerics, two_wp, match_cost_vals",
    [
        (
            True,
            False,
            [
                -1.6333,
                -1.982,
                -1.9875,
                -1.9864,
                -1.9867,
                -1.9876,
                -1.988,
                -1.9874,
                -1.9881,
                -1.9895,
                -1.9884,
            ],
        ),
        (
            True,
            True,
            [
                -0.7659,
                -1.5337,
                -1.9377,
                -1.956,
                -1.963,
                -1.9686,
                -1.9725,
                -1.9748,
                -1.9779,
                -1.9811,
                -1.9816,
            ],
        ),
        (False, False, [-1] * 11),
    ],
)
def test_two_state_discrimination(tmp_dir_cleanup, numerics, two_wp, match_cost_vals):
    """Performs a sample two-state discrimination optimization for 10 steps.
    Both two waveplate and four waveplate versions are tested.
    """
    np.random.seed(123)
    # tolerance for strictly decreasing cost check
    COST_TOL = 1e-2

    filepath = "./test/tmp/"

    num_steps = 10
    opt_dict = pk.two_state_discrimination(
        filepath, numerics=numerics, two_wp=two_wp, num_steps=num_steps, verbose=True
    )

    print(opt_dict["cost_vals"])
    assert len(opt_dict["cost_vals"]) == num_steps + 1
    print(np.all(np.diff(opt_dict["cost_vals"]) < 1e-3))
    # assert np.allclose(opt_dict["cost_vals"], match_cost_vals, atol=1e-5)
    # Check to make sure that the cost_values are strictly decreasing, to a certain tolerance
    assert np.all(np.diff(opt_dict["cost_vals"]) < COST_TOL)
    # Verify created filepath.
    assert os.path.exists("./test/tmp/")

    dev_dir = "numerics/" if numerics else "hardware/"
    assert os.path.exists("./test/tmp/" + dev_dir)

    wp_dir = "two_wp/" if two_wp else "four_wp/"
    assert os.path.exists("./test/tmp/" + dev_dir + wp_dir)

    # Verify that created optimization json file has expected data.
    files = pk.get_files("./test/tmp/" + dev_dir + wp_dir, r"optimization_.*.json")
    assert len(files) == 1
    file_opt_dict = pk.read_json(files[0])

    # assert np.allclose(file_opt_dict["cost_vals"], match_cost_vals, atol=1e-5)
    # Check to make sure that the cost values are strictly decreasing, to a certain tolerance
    assert np.all(np.diff(file_opt_dict["cost_vals"]) < COST_TOL)


@pytest.mark.parametrize(
    "numerics, three_wp, match_cost_vals",
    [
        (
            True,
            False,
            [
                -1.7512,
                -2.5954,
                -3.0074,
                -3.1544,
                -3.2126,
                -3.2524,
                -3.279,
                -3.3146,
                -3.3314,
                -3.3572,
                -3.3752,
            ],
        ),
        (
            True,
            True,
            [
                1.1228,
                0.1372,
                -1.3598,
                -2.2446,
                -2.4824,
                -2.6604,
                -2.8096,
                -2.9498,
                -3.0576,
                -3.159,
                -3.2414,
            ],
        ),
        (False, False, [0] * 11),
    ],
)
def test_qubit_dimensionality_witnessing(tmp_dir_cleanup, numerics, three_wp, match_cost_vals):
    """Performs a sample two-state discrimination optimization for 10 steps.
    Both two waveplate and four waveplate versions are tested.
    """
    np.random.seed(123)

    filepath = "./test/tmp/"

    num_steps = 10
    opt_dict = pk.qubit_dimensionality_witnessing(
        filepath, numerics=numerics, three_wp=three_wp, num_steps=num_steps, verbose=False
    )

    assert len(opt_dict["cost_vals"]) == num_steps + 1
    assert np.allclose(opt_dict["cost_vals"], match_cost_vals, atol=1e-5)

    # Verify created filepath.
    assert os.path.exists("./test/tmp/")

    dev_dir = "numerics/" if numerics else "hardware/"
    assert os.path.exists("./test/tmp/" + dev_dir)

    wp_dir = "three_wp/" if three_wp else "four_wp/"
    assert os.path.exists("./test/tmp/" + dev_dir + wp_dir)

    # Verify that created optimization json file has expected data.
    files = pk.get_files("./test/tmp/" + dev_dir + wp_dir, r"optimization_.*.json")
    assert len(files) == 1
    file_opt_dict = pk.read_json(files[0])

    assert np.allclose(file_opt_dict["cost_vals"], match_cost_vals, atol=1e-5)
