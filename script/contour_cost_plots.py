import pennylane_kinesis as pk

filepath = "./data/fiber_inversion_hardware/"


scan_dict = pk.read_json(
    filepath + "cost_landscape_2023-02-03T17-51-50Z.json",
)

opt_dict = pk.read_json(
    filepath + "fixed_init_optimize_2023-02-03T18-21-58Z.json"
)

pk.plot_trajectory_on_cost_contour(scan_dict, opt_dict)