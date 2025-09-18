import numpy as np
from pennylane import QubitDevice
from pylablib.devices import Thorlabs
import time
from enum import Enum
import os
import csv
from .file_utilities import datetime_now_string
import pickle

from .version import __version__

_TEST_MODE = False
_SWABIAN_TEST_MODE = False


class TimeTaggerType(Enum):
    NONE = 0
    UQD = 1
    SWABIAN = 2


from pylablib.devices import Thorlabs


class KinesisMotors:
    """Connect and control kinesis motors for the Cage Rotator Device."""

    def __init__(self, test_mode=_TEST_MODE):
        self.read_config()
        if test_mode:
            self.prep_hwp = self.prep_hwp_sn
            self.prep_qwp = self.prep_qwp_sn
            self.meas_hwp = self.meas_hwp_sn
            self.meas_qwp = self.meas_qwp_sn
            return
        self.prep_hwp = Thorlabs.KinesisMotor(self.prep_hwp_sn, scale="stage")
        self.prep_qwp = Thorlabs.KinesisMotor(self.prep_qwp_sn, scale="stage")
        self.meas_hwp = Thorlabs.KinesisMotor(self.meas_hwp_sn, scale="stage")
        self.meas_qwp = Thorlabs.KinesisMotor(self.meas_qwp_sn, scale="stage")
        self.wps = [self.prep_hwp, self.prep_qwp, self.meas_hwp, self.meas_qwp]

    def read_config(self):
        """Read the configuration file and set the motor serial numbers."""
        config_file = "./src/pennylane_kinesis/2023_07_10_michael_wp_config.txt"

        with open(config_file) as f:
            config_lines = f.readlines()

        wp_details = [["" for j in range(4)] for i in range(5)]

        for i in range(5):
            entries = config_lines[i].split("\t")
            wp_details[i][0] = entries[0]
            wp_details[i][1] = entries[1]
            wp_details[i][2] = entries[2]
            wp_details[i][3] = entries[3][:-1]  # remove newline at end of line

        self.prep_hwp_sn = wp_details[1][1]
        self.prep_qwp_sn = wp_details[2][1]
        self.meas_hwp_sn = wp_details[3][1]
        self.meas_qwp_sn = wp_details[4][1]

    def close(self):
        for wp in self.wps:
            wp.close()


class KinesisCageRotatorDevice(QubitDevice):
    """A PennyLane plugin device for integrating with Kinesis Cage Rotational Stages.

    This plugin device assumes a single qubit experiment with time-tagger detector device.


    :param wires: A single qubit wire, should be set as ``[0]``.
    :type wires: list[int]

    :param motors: A dataclass containing the connections to all motors.
    :type motors: KinesisMotors

    :param shots: The number shots desired for the experiment.
    :type shots: int

    :param shot_rate: The mean number of photons incident upon the
                      detector per second. This quantity may want to include
                      one or more standard deviations of error where ``std_dev = np.sqrt(shots)``.
    :type shot_rate: int

    :param wimp_mode: If ``True``, device velocities are set to one-half speed.
    :type wimp_mode: bool
    """

    name = "Kinesis Cage Rotator Device"
    short_name = "kinesis.cage.rotator"
    pennylane_requires = "0.34"
    version = __version__
    author = "Brian Doolittle, Michael Vayninger, Timur Javid, and Colin Lualdi"

    operations = {"HWPRotStage", "QWPRotStage"}
    observables = {"Projector", "Identity", "PauliZ"}
    init_state = np.array([1, 0])  # placeholder

    def __init__(
        self,
        wires,
        motors: KinesisMotors,
        shots=1000,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        analytic=None,
        shot_rate=6000,
        wimp_mode=False,
        ttag_type=TimeTaggerType.SWABIAN,
        ttag_channel_transmit=5,
        ttag_channel_reflect=4,
        test_mode=_TEST_MODE,
        filepath=None,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self.test_mode = test_mode
        # Doesn't make sense to run any timetagger if we're in test mode so we ignore the ttag specification
        if self.test_mode == True:
            self.ttag_type = TimeTaggerType.NONE
        else:
            self.ttag_type = ttag_type

        self.wimp_mode = wimp_mode
        self.ttag_channel_transmit = ttag_channel_transmit
        self.ttag_channel_reflect = ttag_channel_reflect
        self.shot_rate = shot_rate  # mean num photons incident upon detector per second
        self.shots = shots
        self.integration_time = shots / (self.shot_rate)

        self.active_times = [[] for _ in range(4)]

        datetime = datetime_now_string()
        self.timing_file = filepath + datetime + "timing_data.pkl"
        # Initialize device list
        self.device_list()

        # Connect and initlaize devices
        print()
        print("Status Updates:")
        print("----------------------------------------------")

        print("Reading configuration file...")
        print()

        config_file = "./src/pennylane_kinesis/2023_07_10_michael_wp_config.txt"

        with open(config_file) as f:
            config_lines = f.readlines()

        wp_details = [["" for j in range(4)] for i in range(5)]

        for i in range(5):
            print("i = ", i)
            entries = config_lines[i].split("\t")
            print(entries)
            wp_details[i][0] = entries[0]
            wp_details[i][1] = entries[1]
            wp_details[i][2] = entries[2]
            wp_details[i][3] = entries[3][:-1]  # remove newline at end of line

        print("Configuration File:")
        print("----------------------------------------------")
        for line in wp_details:
            print(*line)
        print()

        self.a_hwp_sn = wp_details[1][1]
        self.a_qwp_sn = wp_details[2][1]
        self.b_hwp_sn = wp_details[3][1]
        self.b_qwp_sn = wp_details[4][1]

        self.a_hwp_off = float(wp_details[1][2])
        self.a_qwp_off = float(wp_details[2][2])
        self.b_hwp_off = float(wp_details[3][2])
        self.b_qwp_off = float(wp_details[4][2])

        # a waveplates are 1550
        self.dev_a_hwp = motors.prep_hwp
        self.dev_a_qwp = motors.prep_qwp
        self.a_hwp_config = self.device_setup(self.dev_a_hwp)
        self.a_qwp_config = self.device_setup(self.dev_a_qwp)

        # b waveplates are 810
        self.dev_b_hwp = motors.meas_hwp
        self.dev_b_qwp = motors.meas_qwp
        self.b_hwp_config = self.device_setup(self.dev_b_hwp)
        self.b_qwp_config = self.device_setup(self.dev_b_qwp)

        # Home all a devices
        print("Homing all devices...")
        self.time_rotation(self.device_home, self.active_times, 0, self.dev_a_hwp)
        self.time_rotation(self.device_home, self.active_times, 1, self.dev_a_qwp)
        print()

        # Home all b devices
        print("Homing all devices...")
        self.time_rotation(self.device_home, self.active_times, 2, self.dev_b_hwp)
        self.time_rotation(self.device_home, self.active_times, 3, self.dev_b_qwp)
        print()

        self.buffer = self.detector_setup()

    def time_rotation(self, func, active_times, wp_num, *args):
        start = time.time()
        func(*args)
        end = time.time()
        active_times[wp_num].append((start, end))
        with open(self.timing_file, "wb") as f:
            pickle.dump(active_times, f)
        print(f"Time taken for operation {func.__name__} on WP {wp_num}: {end - start:.2f} seconds")

    def device_list(self):
        """Builds device list and prints detected devices."""
        if self.test_mode:
            print("No devices detected.")
            return []
        else:
            devices = Thorlabs.list_kinesis_devices()

            if len(devices) == 0:
                print("No devices detected.")
            else:
                print("%d devices detected!" % len(devices))
                print("----------------------------------------------")
                i = 1
                for device in devices:
                    print("Device " + str(i) + ": S/N " + device[0] + " Type " + device[1])
                    i += 1

            return devices

    def device_setup(self, device: Thorlabs.KinesisMotor):
        """Set device parameters and get configuration information.

        :param device: Device connection.
        :type device: KenisisMotor

        :returns: Configuration information.
        :rtype: dict
        """
        # Set velocity
        # If waveplates are unreliable, can set velocity at near half to keep
        # current load down so not over burden the USB hub (wimp mode)
        # Default value is 10 degrees/s, half is 5 degrees/s
        if self.wimp_mode:
            deviceVel = 5
        else:
            deviceVel = 25

        if self.test_mode:
            return {"device_vel": deviceVel}
        else:
            velocity_parameters = device.get_velocity_parameters()
            updated_velocity_parameters = (
                velocity_parameters.min_velocity,
                velocity_parameters.acceleration,
                deviceVel,
            )
            homing_parameters = device.get_homing_parameters()
            updated_homing_parameters = (
                homing_parameters.home_direction,
                homing_parameters.limit_switch,
                deviceVel,
                homing_parameters.offset_distance,
            )

            device.setup_velocity(*updated_velocity_parameters)
            device.setup_homing(*updated_homing_parameters)

            deviceConfig = device.get_full_info()

            deviceInfo = device.get_device_info()

            print(deviceInfo.model_no + " (S/N " + str(deviceInfo.serial_no) + ") is ready to use.")
            print()

            return deviceConfig

    def detector_setup(self):
        """Initializes the time-tagger buffer."""
        if self.ttag_type == TimeTaggerType.NONE:
            print("Detector in test mode. No time tagger used.")
            buffer = {}
            return buffer
        elif self.ttag_type == TimeTaggerType.UQD:
            try:
                # importing the local time-tagger library
                from . import ttag
            except BaseException as ex:
                msg_template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = msg_template.format(type(ex).__name__, ex.args)
                print(message)

                print("\nAn error occured while importing time tagger")
                print("Defaulting to TimeTaggerType.NONE due to import error.")
                self.ttag_type = TimeTaggerType.NONE
                self.detector_setup()
                return {}

            buffer = ttag.TTBuffer(0)  # Default buffer number is 0
            print("Properties of buffer:")
            print("Channels: ", buffer.channels)
            print("Resolution: ", buffer.resolution)
            print("Datapoints: ", buffer.datapoints)
            return buffer
        elif self.ttag_type == TimeTaggerType.SWABIAN:
            try:
                import TimeTagger as TT

            except ImportError as e:
                print("The Swabian Time Tagger library is not installed.")
                print("Please install it using the instructions on the github README:")
                print("https://github.com/ChitambarLab/pennylane-kinesis-plugin")
                print("\nDefaulting to TimeTaggerType.NONE due to import error.")
                self.ttag_type = TimeTaggerType.NONE
                self.detector_setup()
                return {}

            self.ttag = TT.createTimeTagger()
            self.ttag_stream = TT.TimeTagStream(
                self.ttag,
                1.5 * self.shots,
                channels=[self.ttag_channel_reflect, self.ttag_channel_transmit],
            )
            self.counter = TT.Countrate(self.ttag, channels=[self.ttag_channel_reflect, self.ttag_channel_transmit])
            print(
                "Started Swabian timetagger for channels: ",
                self.ttag_channel_reflect,
                " and ",
                self.ttag_channel_transmit,
            )
            if _SWABIAN_TEST_MODE:
                self.ttag.setTestSignal(self.ttag_channel_reflect, True)
                self.ttag.setTestSignal(self.ttag_channel_transmit, True)
                print("SWABIAN_TEST_MODE is set to True, test signals are enabled.")
                print(
                    "Test signal on channels",
                    self.ttag_channel_reflect,
                    " and ",
                    self.ttag_channel_transmit,
                    "enabled.",
                )
            self.counter.startFor(int(3 * 1e12))  # Take data for 3 second
            self.counter.waitUntilFinished()
            self.shot_rate = self.counter.getData()  # Get average shot rate from the channel
            print("Shot rates:")
            print(f"T rate: {self.shot_rate[0]:.2f} counts/s. R rate: {self.shot_rate[1]:.2f} counts/s")
            # self.shots is user input, calculate integration time based on shot rate from the detector
            self.integration_time = self.shots / (np.sum(self.shot_rate))
            print(f"Set integration time to: {self.integration_time:.5f} seconds.")
            return {}

    def device_home(self, device: Thorlabs.KinesisMotor):
        """Sends the specified device to the home position.

        :param device: A cage rotaator device.
        :type device: CageRotator object

        :returns: None
        """
        if not self.test_mode:
            deviceInfo = device.get_device_info()
            deviceSN = str(deviceInfo.serial_no)

            # if False: #set it to home every time, just as a precaution
            if device.is_homed():
                print("Device S/N " + deviceSN + " is already homed")
                return
            else:
                # Sends the motor to its home position.
                # Argument is the timeout. Function will wait until the move completes
                # or the timeout elapses, whichever comes first.
                device.home()
                device.wait_for_home()
                if device.is_homed():
                    print("Homed device S/N " + deviceSN)
                else:
                    print("Failed to home device S/N " + deviceSN)
                    device.close()

        return True

    def device_move_to_angle(self, device: Thorlabs.KinesisMotor, angle):
        """Rotates a device to the specified angle.

        :param device: A cage rotator device.
        :type device: CageRotator object

        :param angle: The desired angular position (in degrees)
        :type angle: float

        :returns: None
        """
        if self.test_mode:
            print("Moved device S/N " + device + " to position " + str(angle) + "\u00b0")
        else:
            device_info = device.get_device_info()
            device_sn = str(device_info.serial_no)
            device.move_to(angle)  # Moves to position in degrees
            device.wait_for_stop()
            print("Moved device S/N " + device_sn + " to position " + str(angle) + "\u00b0")

    def device_move_by_angle(self, device: Thorlabs.KinesisMotor, angle):
        """Rotates a device by the specified angle.

        :param device: A cage rotator device.
        :type device: CageRotator object

        :param angle: The desired angular position (in degrees)
        :type angle: float

        :returns: None
        """
        if self.test_mode:
            print("Moved device S/N " + device + " by " + str(angle) + "\u00b0")
        else:
            device_info = device.get_device_info()
            device_sn = str(device_info.serial_no)
            device.move_by(angle)  # Moves by position in degrees
            device.wait_for_stop()
            print("Moved device S/N " + device_sn + " by " + str(angle) + "\u00b0")
            print("Current position:", device.get_position())

    def disconnect(self):
        """Disconnects all Kinesis devices."""
        if not self.test_mode:
            # Code must gracefully disconnect from device at its conclusion.
            # Otherwise, the Connect() method will throw an error when the code is rerun

            self.dev_a_hwp.close()  # Disconnects the device.
            self.dev_a_qwp.close()

            self.dev_b_hwp.close()  # Disconnects the device.
            self.dev_b_qwp.close()

        print("Devices disconnected.")

    def reset(self):
        """PennyLane internal device function.

        We leave this function as a passthrough because the kinesis device does not
        store the state of the rotational polarizers.
        """
        super().reset()
        self._state = self.init_state

    @classmethod
    def capabilities(self):
        """PennyLane internal device function."""
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=False,
            supports_analytic_computation=False,
            supports_broadcasting=False,
            returns_state=False,
        )
        return capabilities

    def apply(self, operations, rotations=[]):
        """Instucts the Kinesis devices to perform the specified operations.

        This function extends a built-in PennyLane device method.

        :param operations: A list of operations supported by the kinesis cage rotator device.
        :type operations: A PennyLane Operataion
        """
        all_ops = operations + rotations if rotations else operations

        # TODO: more generic management of device operations
        dev_list = [self.dev_a_hwp, self.dev_a_qwp, self.dev_b_hwp, self.dev_b_qwp]
        for i, op in enumerate(all_ops):
            angle = op.parameters[0] * 360 / (2 * np.pi)

            if op.name == "HWPRotStage":
                self.time_rotation(self.device_move_to_angle, self.active_times, i, dev_list[i], angle % 90)
            elif op.name == "QWPRotStage":
                self.time_rotation(self.device_move_to_angle, self.active_times, i, dev_list[i], angle % 180)
            else:
                print("Operation : ", op.name, ", is not supported")

    def generate_samples(self):
        """Constructs a sequence of sampled data from the device.
        This data can be used to construct probability distributions and expectations.
        """

        if self.ttag_type == TimeTaggerType.NONE:
            singles_array = [0, 0, 0, 0, int(self.shots / 2), int(self.shots / 2)]
            h_counts = singles_array[self.ttag_channel_transmit]
            v_counts = singles_array[self.ttag_channel_reflect]
        else:
            channels = []
            iteration = 0
            while True:
                if iteration == 0 and self.ttag_type == TimeTaggerType.SWABIAN:
                    _ = self.ttag_stream.getData()  # Clear the buffer (it will return stored data otherwise)
                time.sleep(self.integration_time)
                iteration += 1
                # Get data from UQD time tagger
                if self.ttag_type == TimeTaggerType.UQD:
                    channels, _ = self.buffer(self.integration_time * iteration)
                    channel_filter = np.logical_or(
                        channels == self.ttag_channel_transmit,
                        channels == self.ttag_channel_reflect,
                    )
                    channels = channels[channel_filter]
                # Get data from Swabian time tagger
                if self.ttag_type == TimeTaggerType.SWABIAN:
                    ttag_buffer = self.ttag_stream.getData()
                    channels = np.append(channels, ttag_buffer.getChannels())
                if len(channels) >= self.shots:
                    break
            channels = channels[: self.shots]  # truncate to desired number of shots
            h_counts = len(channels[channels == self.ttag_channel_transmit])
            v_counts = len(channels[channels == self.ttag_channel_reflect])

        print()
        print("Measured Singles:")
        print(f"H counts: {h_counts}\t V counts: {v_counts}")

        counts = h_counts + v_counts
        print("total counts : ", counts)

        return np.concatenate(
            (
                np.ones((int(v_counts), 1), dtype=int),
                np.zeros((int(h_counts), 1), dtype=int),
            )
        )
