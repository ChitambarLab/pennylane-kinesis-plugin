from pennylane import QubitDevice
import pennylane as qml
import numpy as np
from pylablib.devices import Thorlabs

import time

from .version import __version__

_TEST_MODE = False

try:
    # importing the local time-tagger library
    from . import ttag

except BaseException as ex:
    msg_template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = msg_template.format(type(ex).__name__, ex.args)
    print(message)

    print("\nAn error occured while importing time tagger")

    _TEST_MODE = True
    print("setting `_TEST_MODE=True`")


class KinesisCageRotatorDevice(QubitDevice):
    """A PennyLane plugin device for integrating with Kinesis Cage Rotational Stages.

    This plugin device assumes a single qubit experiment with time-tagger detector device.


    :param wires: A single qubit wire, should be set as ``[0]``.
    :type wires: list[int]

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
    pennylane_requires = "0.29.1"
    version = __version__
    author = "Brian Doolittle, Michael Vayninger, and Colin Lualdi"

    operations = {"HWPRotStage", "QWPRotStage"}
    observables = {"Projector", "Identity", "PauliZ"}
    init_state = np.array([1, 0])  # placeholder

    def __init__(
        self,
        wires,
        shots=1000,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        analytic=None,
        shot_rate=6000,
        wimp_mode=False,
        ttag_channel_transmit=5,
        ttag_channel_reflect=4,
        # integration_time=1,
        test_mode=_TEST_MODE,
    ):
        super().__init__(
            wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic
        )
        self.test_mode = test_mode
        self.wimp_mode = wimp_mode
        self.ttag_channel_transmit = ttag_channel_transmit
        self.ttag_channel_reflect = ttag_channel_reflect

        self.shot_rate = shot_rate  # mean num photons incident upon detector per second
        self.integration_time = shots / (self.shot_rate)
        # self.integration_time = integration_time

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
        self.dev_a_hwp, self.a_hwp_config = self.device_setup(self.a_hwp_sn)
        self.dev_a_qwp, self.a_qwp_config = self.device_setup(self.a_qwp_sn)

        # b waveplates are 810
        self.dev_b_hwp, self.b_hwp_config = self.device_setup(self.b_hwp_sn)
        self.dev_b_qwp, self.b_qwp_config = self.device_setup(self.b_qwp_sn)

        # Home all a devices
        print("Homing all devices...")
        self.device_home(self.dev_a_hwp)
        self.device_home(self.dev_a_qwp)
        print()

        # Home all b devices
        print("Homing all devices...")
        self.device_home(self.dev_b_hwp)
        self.device_home(self.dev_b_qwp)
        print()

        self.buffer = self.detector_setup()

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
                    print(
                        "Device " + str(i) + ": S/N " + device[0] + " Type " + device[1]
                    )
                    i += 1

            return devices

    def device_setup(self, SN):
        """Connect and initialize device with S/N ``SN``.

        :param SN: Device serial number.
        :type SN: string

        :returns: A CageRotator Object and configuration information.
        :rtype: tuple
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
            return {"sn": SN}, {"device_vel": deviceVel}
        else:
            # Create CageRotator and connect to it
            try:
                device = Thorlabs.KinesisMotor(SN, scale="stage")
                print("Connected to device S/N " + SN)
            except:
                print("Failed to connect device S/N " + SN)

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

            print(
                deviceInfo.model_no
                + " (S/N "
                + str(deviceInfo.serial_no)
                + ") is ready to use."
            )
            print()

            return device, deviceConfig

    def detector_setup(self):
        """Initializes the time-tagger buffer."""
        if self.test_mode:
            print("detector in test mode")
            buffer = {}
            return buffer
        else:
            buffer = ttag.TTBuffer(0)  # Default buffer number is 0
            print("Properties of buffer:")
            print("Channels: ", buffer.channels)
            print("Resolution: ", buffer.resolution)
            print("Datapoints: ", buffer.datapoints)

            return buffer

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

        :param device: A cage rotaator device.
        :type device: CageRotator object

        :param angle: The desired angular position (in degrees)
        :type angle: float

        :returns: None
        """
        if self.test_mode:
            print(
                "Moved device S/N "
                + device["sn"]
                + " to position "
                + str(angle)
                + "\u00B0"
            )
        else:
            device_info = device.get_device_info()
            device_sn = str(device_info.serial_no)
            device.move_to(angle)  # Moves to position in degrees
            device.wait_for_stop()
            print(
                "Moved device S/N "
                + device_sn
                + " to position "
                + str(angle)
                + "\u00B0"
            )

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
                self.device_move_to_angle(dev_list[i], angle % 90)
            elif op.name == "QWPRotStage":
                self.device_move_to_angle(dev_list[i], angle % 180)
            else:
                print("Operation : ", op.name, ", is not supported")

    def generate_samples(self):
        """Constructs a sequence of sampled data from the device.
        This data can be used to construct probability distributions and expectations.

        Since our measurement only detects one polarization, we estimate the total number
        of counts to construct the relative probability distribution.
        """

        timeout = 5 * self.integration_time

        if self.test_mode:
            singles_array = [0, 0, 0, 0, int(self.shots / 2), int(self.shots / 2)]
            h_counts = singles_array[self.ttag_channel_transmit]
            v_counts = singles_array[self.ttag_channel_reflect]
        else:
            channels = []
            start_time = time.perf_counter()
            iteration = 0
            while (
                len(channels) < self.shots
                and time.perf_counter() - start_time < timeout
            ):
                time.sleep(self.integration_time)
                iteration += 1
                channels, _ = self.buffer(self.integration_time * iteration)
                channel_filter = np.logical_or(
                    channels == self.ttag_channel_transmit,
                    channels == self.ttag_channel_reflect,
                )
                channels = channels[channel_filter]
            if len(channels) < self.shots:
                # if buffer did not fill up, warn user and compensate
                print("Warning: buffer did not fill up in time")
                h_counts = len(channels[channels == self.ttag_channel_transmit])
                v_counts = len(channels[channels == self.ttag_channel_reflect])
                missing_counts = self.shots - len(channels)
                h_compensation = missing_counts * (h_counts // (h_counts + v_counts))
                v_compensation = missing_counts - h_compensation
                h_counts += h_compensation
                v_counts += v_compensation
            else:
                channels = channels[: self.shots]
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
