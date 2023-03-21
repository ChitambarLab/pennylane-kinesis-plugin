import pennylane_kinesis as pk

"""This script generates a plot showing the optimization precision as
dependent upon the optimization step size. Data for this plot is obtained numerically
using the `./script/fiber_inversion_hardware_step_size_dependence.py` script.
"""

filepath = "./data/fiber_inversion_hardware_step_size_dependence/"
step_size_list = [0.2, 0.15, 0.1, 0.05]
plot_filename = "step_size_dependence_numerics"

pk.plot_step_size_dependence(filepath, step_size_list, plot_filename, background=300, shot_rate=45000)
