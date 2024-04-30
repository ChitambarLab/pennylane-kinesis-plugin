import pennylane_kinesis as pk
from datetime import datetime

"""This script generates a plot showing the optimization precision as
dependent upon the optimization step size. Data for this plot is obtained numerically
using the `./script/fiber_inversion_numeric_step_size_dependence.py` script.
"""

filepath = "./data/state_discrimination/numerics_step_size_dependence/finite-diff/adagrad/"
# step_size_list = [1.0, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001]
# step_size_list = [0.2, 0.15, 0.1, 0.05]
step_size_list = [0.01, 0.5, 0.2, 0.1, 0.05]

plot_filename = "step_size_dependence_finite_diff_numerics_adagrad"

date_after = datetime.strptime("2024-04-25T00-17-20Z", "%Y-%m-%dT%H-%M-%SZ")
pk.plot_step_size_dependence(
    filepath,
    step_size_list,
    plot_filename,
    date_after=date_after,
    numerics=True,
    two_wp=False,
    add_background=True,
    background=300,
    shot_rate=10000,
    title="Step Size Dependence \n(adagrad, finite-diff, numerics)"
)
