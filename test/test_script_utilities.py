import pytest
import os
import numpy as np

import pennylane_kinesis as pk

from test_file_utilities import tmp_dir_cleanup


@pytest.mark.parametrize(
    "two_wp, match_cost_vals",
    [
        (
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
    ],
)
def test_two_state_discrimination(tmp_dir_cleanup, two_wp, match_cost_vals):
    """Performs a sample two-state discrimination optimization for 10 steps.
    Both two waveplate and four waveplate versions are tested.
    """
    np.random.seed(123)

    filepath = "./test/tmp/"

    num_steps = 10
    opt_dict = pk.two_state_discrimination(
        filepath, numerics=True, two_wp=two_wp, num_steps=num_steps, verbose=False
    )

    assert len(opt_dict["cost_vals"]) == num_steps + 1
    assert np.allclose(opt_dict["cost_vals"], match_cost_vals, atol=1e-5)

    # Verify created filepath.
    assert os.path.exists("./test/tmp/")
    assert os.path.exists("./test/tmp/numerics/")

    wp_dir = "two_wp/" if two_wp else "four_wp/"
    assert os.path.exists("./test/tmp/numerics/" + wp_dir)

    # Verify that created optimization json file has expected data.
    files = pk.get_files("./test/tmp/numerics/" + wp_dir, r"optimization_.*.json")
    assert len(files) == 1
    file_opt_dict = pk.read_json(files[0])

    assert np.allclose(file_opt_dict["cost_vals"], match_cost_vals, atol=1e-5)
