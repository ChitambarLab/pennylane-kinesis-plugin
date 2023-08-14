import pennylane_kinesis as pk

"""This script generates a plot showing the optimization precision as
dependent upon the optimization step size. Data for this plot is obtained numerically
using the `./script/fiber_inversion_hardware_step_size_dependence.py` script.
"""

filepath = "./data/state_discrimination_hardware_step_size_dependence/two_wp/"
step_size_list = [
    0.09,
    0.1,
    0.11,
    0.12,
    0.13,
    0.135,
    0.15,
    0.17,
    0.2,
]
plot_filename = "step_size_dependence_numerics"

pk.plot_step_size_dependence_convergence(
    filepath, step_size_list, plot_filename, background=4, shot_rate=50000
)

pk.plot_step_size_dependence(
    filepath, step_size_list, plot_filename, background=4, shot_rate=50000
)
