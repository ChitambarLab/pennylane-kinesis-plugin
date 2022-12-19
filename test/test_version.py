import re

import pennylane_kinesis


def test_version():

    version_str = pennylane_kinesis.__version__

    assert isinstance(version_str, str)
    assert re.fullmatch(r"0\.1\.\d+", version_str) != None
