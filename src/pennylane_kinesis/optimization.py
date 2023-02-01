import pennylane as qml
from pennylane import numpy as qnp

from .file_utilities import datetime_now_string, tmp_dir, write_json


def optimize(
    cost,
    init_settings,
    num_steps=10,
    step_size=0.1,
    filepath="./",
    filename="",
    init_opt_dict=None,
    **opt_meta_kwargs
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

    :param opt_meta_kwargs: Additional keyword arguments to write to optimization dictionary and saved JSON.
    :type opt_meta_kwargs: dictionary

    :returns: Contains all optimization data.
    :rtype: dictionary
    """

    opt_dict = (
        init_opt_dict
        if init_opt_dict
        else {
            "cost_vals": [float(cost(init_settings))],
            "settings_history": [init_settings.tolist()],
            "step_size": step_size,
            "num_steps": num_steps,
            "datetime": datetime_now_string(),
            **opt_meta_kwargs,
        }
    )

    unique_filename = filename + "_" + opt_dict["datetime"]

    opt = qml.GradientDescentOptimizer(stepsize=step_size)
    current_step = len(opt_dict["cost_vals"]) - 1
    settings = qnp.array(opt_dict["settings_history"][-1], requires_grad=True)
    try:
        for i in range(current_step, num_steps):
            settings, cost_val = opt.step_and_cost(cost, settings)
            opt_dict["settings_history"] += [settings.tolist()]
            opt_dict["cost_vals"] += [float(cost_val)]

    except BaseException as err:
        msg_template = (
            "An exception of type {0} occurred during optimization step {1}. Arguments:\n{2!r}"
        )
        message = msg_template.format(type(err).__name__, str(i), err.args)
        print(message)

        tmp_path = tmp_dir(filepath)
        write_json(opt_dict, tmp_path + unique_filename)
    else:
        write_json(opt_dict, filepath + unique_filename)

    return opt_dict


def scan_two_param_cost(cost, x_range, y_range):
    cost_grid = qnp.zeros((len(x_range), len(y_range)))

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            cost_val = cost([x_range[i], y_range[j]])

            cost_grid[i, j] = cost_val

    # TODO:
