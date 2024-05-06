import pennylane_kinesis as pk
from datetime import datetime

"""This script generates a plot showing the optimization precision as
dependent upon the optimization step size. Data for this plot is obtained numerically
using the `./script/fiber_inversion_numeric_step_size_dependence.py` script.
"""

filepath = "./data/state_discrimination/hardware_shot_dependence/parameter-shift/gradient-descent/2024-04-30T23-30-05Z/"
# step_size_list = [1.0, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001]
# step_size_list = [0.2, 0.15, 0.1, 0.05]
# shots_list = [10, 100, 1000, 10000, 100000]
shots_list = [10, 100, 1000, 10000, 100000]
plot_filename = "shots_dependence_parameter_shift_grad_descent_hardware"

date_after = datetime.strptime("2024-04-25T00-17-20Z", "%Y-%m-%dT%H-%M-%SZ")
pk.plot_shot_dependence(
    filepath,
    shots_list=shots_list,
    plot_filename=plot_filename,
    numerics=False,
    two_wp=False,
    add_background=True,
    background=300,
    shot_rate=10000,
    title="Number of Shots Vs.\nOptimization Performance \n(gradient-descent, \n parameter-shift, hardware)"
)
