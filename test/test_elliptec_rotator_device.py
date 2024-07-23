import pytest
import pennylane_kinesis as pk
import os
from pathlib import Path

# Can't really test much other than this without using the hardware...


def setup_module():
    global motors
    global device
    motors = pk.ElliptecMotors(test_mode=True)
    device = pk.ElliptecRotatorDevice(wires=[0], motors=motors, test_mode=True)


class TestElliptecMotors:
    def test_config_serial_nos(self):
        global motors
        assert motors.prep_hwp_sn == "11400288"
        assert motors.prep_qwp_sn == "11400240"
        assert motors.meas_hwp_sn == "11400238"
        assert motors.meas_qwp_sn == "11400239"

    def test_config_angles(self):
        global motors
        assert float(motors.wp_details[0]["angle_offset"]) == 204.128
        assert float(motors.wp_details[1]["angle_offset"]) == 191.734
        assert float(motors.wp_details[2]["angle_offset"]) == 97.551
        assert float(motors.wp_details[3]["angle_offset"]) == -23.635


class TestElliptecRotatorDevice:
    def test_device_timing_file_path(self):
        global device
        assert os.path.exists("./test/tmp/")
        assert device.timing_file == Path("./test/tmp/timing_data_" + device.datetime + ".pkl")
