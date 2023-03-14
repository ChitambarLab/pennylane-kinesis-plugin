import pennylane_kinesis as pk

"""This script simply generates a plot comparing hardware with simulated
shot dependence data.
"""

hardware_filepath = "./data/fiber_inversion_hardware_shot_dependence/"
numerics_filepath = "./data/fiber_inversion_numeric_shot_dependence/"
shots_list = [10000, 1000]
# shots_list = [10000, 3500, 1000, 350, 100, 35, 10]
plot_filename = "./data/plot_hardware_numerics_shot_dependence_comparison/plot"

pk.plot_shot_dependence_comparison(hardware_filepath, numerics_filepath, shots_list, plot_filename)
