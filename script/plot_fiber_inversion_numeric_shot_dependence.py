import pennylane_kinesis as pk

"""This script generates a plot showing the optimization precision as
dependent upon the number of shots. Data for this plot is obtained numerically
using the `./script/fiber_inversion_numeric_shot_dependence.py` script.
"""

filepath = "./data/fiber_inversion_numeric_shot_dependence/"
shots_list = [10000, 3500, 1000, 350, 100, 35, 10]
plot_filename = "shot_dependence_numerics"

pk.plot_shot_dependence(filepath, shots_list, plot_filename)
