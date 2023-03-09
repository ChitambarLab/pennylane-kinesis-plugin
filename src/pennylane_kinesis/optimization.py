import pennylane as qml
from pennylane import numpy as qnp
import time

from .file_utilities import datetime_now_string, tmp_dir, write_json


def optimize(
    cost,
    init_settings,
    num_steps=10,
    step_size=0.1,
    filepath="./",
    filename="",
    init_opt_dict=None,
    domain_periods=None,
    use_adam=False,
    verbose=True,
    **meta_kwargs,
):
    """Optimizes the specified ``cost`` function using gradient descent starting from the
    specified ``init_settings``.

    In a gradient descent step the settings :math:`\\vec{\\theta}` are updated as

    .. math::

        \\vec{\\theta}' = \\vec{\\theta} - \\eta \\nabla_{\\vec{\\theta}} \\textrm{Cost}(\\vec{\\theta})

    where :math:`\\eta` is the stepsize.

    Upon completion, the optimization data is written to a JSON file that takes the same form
    as the return optimization dictionary.

    If an error occurs during optimization, then accumulated data is written to JSON file within
    a ``tmp/`` directory of the specified ``filepath``. This optimization data can be used to start
    a new optimization using the ``init_opt_dict`` argument.

    :param cost: The cost function to minimize.
    :type cost: function

    :param init_settings: The settings to initialize the gradient descent optimization with.
    :type init_settings: list[float]

    :param num_steps: The number of gradient descent iterations. Default: 10.
    :type num_steps: int

    :param step_size: The step to take in the direction of steepest descent. Default: 0.1.
    :type step_size: float

    :param filepath: The directory to write optimization data to. Note that this should
                     string should end with a ``"/"``. Default "./"
    :type filepath: string

    :param filename: The name of the file to write. Default "".
    :type filename: string

    :param init_opt_dict: An optimization dictionary to continue optimizing. Default: ``None``.
    :type init_opt_dict: dictionary

    :param domain_periods: The period of each parameter in the settings list.
    :type domain_periods: list[float]

    :param use_adam: If ``True``, uses the ADAM optimizer instead of the standard gradient descent optimizer. Default ``False``
    :type use_adam: Bool

    :param verbose: If ``True``, prints out optimization progress. Default ``True``.
    :type verbose: Bool

    :param opt_meta_kwargs: Additional keyword arguments to write to optimization dictionary and saved JSON.
    :type opt_meta_kwargs: dictionary

    :returns: Contains all optimization data.
    :rtype: dictionary
    """

    opt_dict = (
        init_opt_dict
        if init_opt_dict
        else {
            "cost_vals": [],
            "settings_history": [init_settings.tolist()],
            "step_size": step_size,
            "num_steps": num_steps,
            "datetime": datetime_now_string(),
            "domain_periods": domain_periods,
            "opt_step_time": [],
            "use_adam": use_adam,
            **meta_kwargs,
        }
    )

    opt = (
        qml.AdamOptimizer(stepsize=step_size)
        if use_adam
        else qml.GradientDescentOptimizer(stepsize=step_size)
    )
    current_step = len(opt_dict["cost_vals"])
    settings = qnp.array(opt_dict["settings_history"][-1], requires_grad=True)
    try:
        curr_time = time.time()
        for i in range(current_step, num_steps):
            if verbose:
                print("\n\nOptimization Step : ", i, "\n\n")
            settings, cost_val = opt.step_and_cost(cost, settings)
            if domain_periods:
                settings = qnp.array(
                    [settings[i] % domain_periods[i] for i in range(len(settings))],
                    requires_grad=True,
                )

            opt_dict["settings_history"] += [settings.tolist()]
            opt_dict["cost_vals"] += [float(cost_val)]

            if verbose:
                print("current cost : ")
                print(opt_dict["cost_vals"])

            new_time = time.time()
            opt_dict["opt_step_time"] += [float(new_time - curr_time)]
            curr_time = new_time

        opt_dict["cost_vals"] += [float(cost(settings))]

    except BaseException as err:
        msg_template = (
            "An exception of type {0} occurred during optimization step {1}. Arguments:\n{2!r}"
        )
        message = msg_template.format(type(err).__name__, str(i), err.args)
        print(message)

        tmp_path = tmp_dir(filepath)
        write_json(opt_dict, tmp_path + filename)
    else:
        write_json(opt_dict, filepath + filename)

    return opt_dict


def scan_two_param_fn(
    fn, x_range, y_range, filename="", filepath="./", init_data_dict=None, **meta_kwargs
):
    """Scans function ``fn`` over two paramater ranges.

    All data is written to a JSON file upon completion. If an error occurs during the scan, then
    the resulting data is stored in a ``"tmp/"`` directory on ``filepath``.

    :param fn: A function called as ``fn([x,y])`` where ``x`` and ``y`` are the two parameters.
    :type fn: function

    :param x_range: The values for ``x``.
    :type x_range: list

    :param y_range: The value of ``y``.
    :type y_range: list

    :param filename: The name of the JSON file to write. Default ``""``.
    :type filename: string

    :param filepath: The path to the JSON file. Note that this
                     string should end with a ``"/"``. Default ``"./"``.
    :type filepath: string

    :param init_data_dict: Used to warm start a scan.
    :type init_data_dict: dictionary

    :param meta_kwargs: extraneous keyword arguments that are written to the JSON file.

    :returns: dictionary containing the keys
    :rtype: dictionary

    * ``x_grid``: x data for plotting on a contour plot.
    * ``y_grid``: y data for plotting on a contour plot.
    * ``z_grid``: z data for plotting on a contour plot.
    * ``x_curr``: current x id when data was saved.
    * ``y_curr``: current y id when data was saved.
    * ``grid_shape``: the shape of the grid matrices.
    * ``datetime``: the datetime at which datacollection began.
    """

    x_grid, y_grid = qnp.meshgrid(x_range, y_range)

    data_dict = (
        init_data_dict
        if init_data_dict
        else {
            "x_grid": x_grid.tolist(),
            "y_grid": y_grid.tolist(),
            "grid_shape": [*x_grid.shape],
            "x_curr": 0,
            "y_curr": 0,
            "datetime": datetime_now_string(),
            **meta_kwargs,
        }
    )

    z_grid = qnp.array(data_dict["z_grid"]) if init_data_dict else qnp.zeros(x_grid.shape)

    try:
        for y in range(data_dict["y_curr"], z_grid.shape[0]):
            data_dict["y_curr"] = y
            for x in range(0, z_grid.shape[1]):
                data_dict["x_curr"] = x
                z_grid[y, x] = fn([x_grid[y, x], y_grid[y, x]])

    except BaseException as err:
        msg_template = "An exception of type {0} occurred during two-param scan step (x,y) = {1}. Arguments:\n{2!r}"
        message = msg_template.format(
            type(err).__name__,
            "(" + str(data_dict["x_curr"]) + "," + str(data_dict["y_curr"]) + ")",
            err.args,
        )
        print(message)

        data_dict["z_grid"] = z_grid.tolist()

        tmp_path = tmp_dir(filepath)
        write_json(data_dict, tmp_path + filename)

    else:
        data_dict["z_grid"] = z_grid.tolist()
        write_json(data_dict, filepath + filename)

    return data_dict
