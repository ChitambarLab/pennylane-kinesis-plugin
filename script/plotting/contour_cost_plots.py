import pennylane_kinesis as pk

"""This script creates a plot from the data obtained using the
`./script/fiber_inversion_hardware.py` script.
"""

filepath = "./data/fiber_inversion_hardware/"

scan_dict = pk.read_json(filepath + "cost_landscape_2023-02-04T03-24-26Z.json")

opt_dict = pk.read_json(filepath + "fixed_init_optimize_2023-02-04T03-54-20Z.json")

datetime_str = pk.datetime_now_string()

pk.plot_trajectory_on_cost_contour(
    filepath + "cost_contour_plot_" + datetime_str, scan_dict, opt_dict, background_shots=100
)
