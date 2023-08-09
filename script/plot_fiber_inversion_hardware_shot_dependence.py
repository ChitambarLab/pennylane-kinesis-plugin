import pennylane_kinesis as pk

"""This script simply generates a plot from the data obtained using
the `./script/fiber_inversion_hardware_shot_dependence.py` script.
"""

filepath = "./data/state_discrimination_hardware_shot_dependence/"
shots_list = [100000, 10000, 1000, 100, 10]
# shots_list = [10000, 3500, 1000, 350, 100, 35, 10]
plot_filename = "shot_dependence_hardware"

pk.plot_shot_dependence(
    filepath, shots_list, plot_filename, background=4, shot_rate=100000
)
