from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import sys


def winapi_path(dos_path, encoding=None):
    """Stolen from stack overflow, this function converts a dos path to a windows api path.
    This is to support filenames longer than 260 characters (which windows doesn't allow with DOS paths).
    """
    if sys.platform == "win32":
        if not isinstance(dos_path, str) and encoding is not None:
            dos_path = dos_path.decode(encoding)
        path = os.path.abspath(dos_path)
        if path.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path[2:]
        return "\\\\?\\" + path
    else:
        return dos_path


def datetime_now_string():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def get_date_from_filename(filename):
    stripped_filename = filename.split("_")[-1]
    stripped_filename = stripped_filename.split(".")[0]
    return datetime.strptime(stripped_filename, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)


def tmp_dir(filepath):
    return make_dir(filepath + "tmp/")


def make_dir(filepath):
    win_path = winapi_path(filepath)
    Path.mkdir(Path(win_path), parents=True, exist_ok=True)
    return filepath


def get_files(path, regex):
    """Retrieves all data files that match the ``regex`` in the
    directory specified by ``path``.
    """
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if (
            f.endswith(".json")
            and os.path.isfile(os.path.join(path, f))
            and bool(re.match(regex, f))
        )
    ]


def write_json(json_dict, filename):
    """Writes the dictionary to JSON file with name ``filename``.

    :param json_dict: The dictionary to write as a JSON file.
    :type json_dict: dict

    :param filename: The name of the JSON file. Note that ``.json`` extension is automatically added.
    :type filename: string

    :returns: ``None``
    """
    with open(winapi_path(filename) + ".json", "w") as file:
        file.write(json.dumps(json_dict, indent=2))


def read_json(filename):
    """Reads data from a JSON file.

    :param filename: The path to the JSON file. Note this string must contain the ``.json`` extension.
    :type filename: string

    :returns: The dictionary read from the JSON file.
    :rtype: dict
    """

    with open(filename) as file:
        opt_dict = json.load(file)

    return opt_dict
