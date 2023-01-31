from datetime import datetime
import json
import os


def datetime_now_string():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def tmp_dir(filepath):
    tmp_path = filepath + "tmp/"
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    return tmp_path


def write_optimization_json(opt_dict, filename):
    """Writes the optimization dictionary to a JSON file.

    :param opt_dict: The dictionary returned by a network optimization.
    :type opt_dict: dict

    :param filename: The name of the JSON file to be written. Note that ``.json`` extension is automatically added.
    :type filename: string

    :returns: ``None``
    """

    with open(filename + ".json", "w") as file:
        file.write(json.dumps(opt_dict, indent=2))


def read_optimization_json(filename):
    """Reads data from an optimization JSON created via ``write_optimization_json``.

    :param filename: The path to the JSON file. Note this string must contain the ``.json`` extension.
    :type filename: string

    :returns: The optimization dictionary read from the file.
    :rtype: dict
    """

    with open(filename) as file:
        opt_dict = json.load(file)

    return opt_dict
