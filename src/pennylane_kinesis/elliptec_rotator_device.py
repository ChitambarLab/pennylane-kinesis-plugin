import elliptec
import numpy as np
from pennylane import QubitDevice
from pylablib.devices import Thorlabs
import time
from enum import Enum
from .file_utilities import datetime_now_string, winapi_path
import pickle
import re
from pathlib import Path
from multiprocessing import Pool

from .version import __version__

_TEST_MODE = False
_SWABIAN_TEST_MODE = False


class TimeTaggerType(Enum):
    NONE = 0
    UQD = 1
    SWABIAN = 2


class ElliptecMotors:
    """Connect and control kinesis motors for the Cage Rotator Device."""

    def __init__(
        self,
        com_port=["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"],
        test_mode=_TEST_MODE,
    ):
        self.wp_details = self.read_config()
        self.test_mode = test_mode
        self.parallel = False
        if test_mode:
            self.prep_hwp = self.prep_hwp_sn
            self.prep_qwp = self.prep_qwp_sn
            self.meas_hwp = self.meas_hwp_sn
            self.meas_qwp = self.meas_qwp_sn
            return

        if isinstance(com_port, list):
            self.controller = [
                elliptec.Controller(port, timeout=5, write_timeout=2, debug=False)
                for port in com_port
            ]
            self.parallel = True
        else:
            self.controller = elliptec.Controller(
                com_port, timeout=5, write_timeout=2, debug=False
            )

        # devices = elliptec.scan_for_devices(self.controller)
        addresses = [int(device["address"]) for device in self.wp_details]
        serial_nos = [device["serial_number"] for device in self.wp_details]
        print(serial_nos)
        self.wps = [None, None, None, None]

        if self.parallel:
            print("Scanning devices...")
            scanned_devices = []
            for control in self.controller:
                scanned = elliptec.scan_for_devices(
                    control, start_address=0, stop_address=0
                )
                # Each motor has it's own controller, so it should ideally only return
                # one device when scanned
                scanned_devices.append(scanned[0])
        else:
            scanned_devices = elliptec.scan_for_devices(
                self.controller, start_address=1, stop_address=4
            )
        if len(scanned_devices) < 4:
            print("All 4 waveplates not detected! Exiting...")
            exit()
        for i, device in enumerate(scanned_devices):
            device_serial = device["info"]["Serial No."]
            device_address = int(device["info"]["Address"])
            serial_address_mapping = int(addresses[serial_nos.index(device_serial)])

            if device_serial not in serial_nos:
                print("Unexpected device serial number! Exiting...")
                exit()
            elif not self.parallel and device_address != serial_address_mapping:
                print(
                    "Device ",
                    device_serial,
                    " has unexpected address ",
                    device_address,
                    ". Expected ",
                    serial_address_mapping,
                    ".",
                )
                exit()

            if self.parallel:
                rotator = elliptec.Rotator(self.controller[i], address="0", debug=False)
            else:
                rotator = elliptec.Rotator(
                    self.controller, address=device["info"]["Address"], debug=False
                )
            self.wps[serial_nos.index(device_serial)] = rotator

        # for address in addresses:
        #     print(address)
        #     self.wps[int(address)-1] = elliptec.Rotator(self.controller, address=str(address))

        print(self.wps)
        self.prep_hwp = self.wps[0]
        self.prep_qwp = self.wps[1]
        self.meas_hwp = self.wps[2]
        self.meas_qwp = self.wps[3]

    def read_config(self):
        """Read the configuration file and set the motor serial numbers."""
        config_file = "./src/pennylane_kinesis/2025_09_12_timur_elliptec_config.txt"

        with open(config_file) as f:
            config_lines = f.readlines()

        wp_details = []

        for i in range(1, 5):
            config_line = re.sub(r"[\n\t\s]*", "", config_lines[i])

            entries = config_line.strip().split(",")
            rot_mount = {}
            rot_mount["device"] = entries[0]
            rot_mount["address"] = entries[1]
            rot_mount["serial_number"] = entries[2]
            rot_mount["angle_offset"] = entries[3]
            rot_mount["name"] = entries[4]

            # If we have extra entries in the config file, it's going to be the port
            # which is used for parallel operation
            if len(entries) > 5:
                rot_mount["port"] = entries[5]
            wp_details.append(rot_mount)

        self.prep_hwp_sn = wp_details[0]["serial_number"]
        self.prep_qwp_sn = wp_details[1]["serial_number"]
        self.meas_hwp_sn = wp_details[2]["serial_number"]
        self.meas_qwp_sn = wp_details[3]["serial_number"]

        return wp_details

    def close(self):
        if self.test_mode:
            return
        for wp in self.wps:
            wp.controller.s.close()


class ElliptecRotatorDevice(QubitDevice):
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

    name = "Elliptec Rotator Device"
    short_name = "elliptec.rotator"
    pennylane_requires = "0.34"
    version = __version__
    author = "Brian Doolittle, Michael Vayninger, Timur Javid, and Colin Lualdi"

    operations = {"HWPRotStage", "QWPRotStage"}
    observables = {"Projector", "Identity", "PauliZ"}
    init_state = np.array([1, 0])  # placeholder

    def __init__(
        self,
        wires,
        motors: ElliptecMotors,
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
        filepath="./test/tmp/",
        debug=False,
    ):
        self.motors = motors
        self.debug = debug
        super().__init__(
            wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic
        )
        self.test_mode = test_mode
        # Doesn't make sense to run any timetagger if we're in test mode so we ignore the ttag specification
        if self.test_mode == True:
            self.ttag_type = TimeTaggerType.NONE
        else:
            self.ttag_type = ttag_type

        self.num_buffer = 0
        self.wimp_mode = wimp_mode
        self.ttag_channel_transmit = ttag_channel_transmit
        self.ttag_channel_reflect = ttag_channel_reflect
        self.shot_rate = shot_rate  # mean num photons incident upon detector per second
        self.shots = shots
        self.integration_time = shots / (self.shot_rate)
        self.ttag_stream = None

        self.active_times = [[] for _ in range(4)]

        datetime = datetime_now_string()
        self.datetime = datetime
        self.timing_file = Path(
            winapi_path(filepath + "timing_data_" + datetime + ".pkl")
        )
        self.timing_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect and initlaize devices
        config_file = "./src/pennylane_kinesis/2025_09_12_timur_elliptec_config.txt"

        with open(config_file) as f:
            config_lines = f.readlines()

        self.wp_details = []

        for i in range(1, 5):
            config_line = re.sub(r"[\n\t\s]*", "", config_lines[i])

            entries = config_line.strip().split(",")
            rot_mount = {}
            rot_mount["device"] = entries[0]
            rot_mount["address"] = entries[1]
            rot_mount["serial_number"] = entries[2]
            rot_mount["angle_offset"] = float(entries[3])
            rot_mount["name"] = entries[4]
            rot_mount["background_angle"] = float(entries[5])
            self.wp_details.append(rot_mount)

        self.a_hwp_sn = self.wp_details[0]["serial_number"]
        self.a_qwp_sn = self.wp_details[1]["serial_number"]
        self.b_hwp_sn = self.wp_details[2]["serial_number"]
        self.b_qwp_sn = self.wp_details[3]["serial_number"]

        self.a_hwp_offset = float(self.wp_details[0]["angle_offset"])
        self.a_qwp_offset = float(self.wp_details[1]["angle_offset"])
        self.b_hwp_offset = float(self.wp_details[2]["angle_offset"])
        self.b_qwp_offset = float(self.wp_details[3]["angle_offset"])

        self.a_hwp_background_angle = float(self.wp_details[0]["background_angle"])
        self.a_qwp_background_angle = float(self.wp_details[1]["background_angle"])
        self.b_hwp_background_angle = float(self.wp_details[2]["background_angle"])
        self.b_qwp_background_angle = float(self.wp_details[3]["background_angle"])

        # a waveplates are 1550
        self.dev_a_hwp = motors.prep_hwp
        self.dev_a_qwp = motors.prep_qwp

        print(self.dev_a_hwp)
        self.a_hwp_config = self.device_setup(self.dev_a_hwp)
        self.a_qwp_config = self.device_setup(self.dev_a_qwp)

        # b waveplates are 810
        self.dev_b_hwp = motors.meas_hwp
        self.dev_b_qwp = motors.meas_qwp
        self.b_hwp_config = self.device_setup(self.dev_b_hwp)
        self.b_qwp_config = self.device_setup(self.dev_b_qwp)

        if self.motors.parallel:
            for controller in self.motors.controller:
                controller.s.flush()
        else:
            self.motors.controller.s.flush()

        # Home all a devices
        print("Homing all devices...")

        if self.motors.parallel:
            print(self.motors.wps)
            self.time_parallel_rotation(
                self.device_home, self.active_times, self.motors.wps
            )
            print("Setting angle to zero")
            self.time_parallel_rotation(
                self.device_move_to_angle,
                self.active_times,
                self.motors.wps,
                [
                    self.a_hwp_offset,
                    self.a_qwp_offset,
                    self.b_hwp_offset,
                    self.b_qwp_offset,
                ],
            )
        else:
            self.time_rotation(self.device_home, self.active_times, 0, self.dev_a_hwp)
            self.time_rotation(self.device_home, self.active_times, 1, self.dev_a_qwp)

            # Home all b devices
            self.time_rotation(self.device_home, self.active_times, 2, self.dev_b_hwp)
            self.time_rotation(self.device_home, self.active_times, 3, self.dev_b_qwp)
            print("Setting angle to zero")
            self.device_move_to_angle(self.dev_a_hwp, self.a_hwp_offset)
            self.device_move_to_angle(self.dev_a_qwp, self.a_qwp_offset)
            self.device_move_to_angle(self.dev_b_hwp, self.b_hwp_offset)
            self.device_move_to_angle(self.dev_b_qwp, self.b_qwp_offset)

        self.device_move_to_angle(self.dev_b_hwp, self.b_hwp_offset + 22.5)

        self.buffer = self.detector_setup()

    def time_parallel_rotation(self, func, active_times, devices, *args):
        start_time = time.time()

        print("args", args)
        print("devices", devices)
        if not self.test_mode:
            if args != ():
                func_args = list(zip(devices, args[0]))
                with Pool(4) as p:
                    p.starmap(func, func_args)
            else:
                with Pool(4) as p:
                    p.map(func, devices)

        end_time = time.time()
        if self.debug:
            print(f"Time taken to move waveplates: {end_time - start_time:.2f} seconds")

    def time_rotation(self, func, active_times, wp_num, *args):
        h_counts_end, v_counts_end = 0, 0
        h_counts_start, v_counts_start = 0, 0
        if self.ttag_stream:
            start_counts = self.generate_samples()
            h_counts_start, v_counts_start = (
                len(start_counts) - np.sum(start_counts),
                np.sum(start_counts),
            )
            if self.debug:
                print(h_counts_start, v_counts_start)
        set_angle = 0
        if len(args) > 1:
            set_angle = args[-1]
        start_pos = 0
        if not self.test_mode:
            match wp_num:
                case 0:
                    start_pos = self.dev_a_hwp.get_angle()
                case 1:
                    start_pos = self.dev_a_qwp.get_angle()
                case 2:
                    start_pos = self.dev_b_hwp.get_angle()
                case 3:
                    start_pos = self.dev_b_qwp.get_angle()
        else:
            start_pos = 0
        start = time.time()
        func(*args)
        end = time.time()
        end_pos = 0
        if not self.test_mode:
            match wp_num:
                case 0:
                    end_pos = self.dev_a_hwp.get_angle()
                case 1:
                    end_pos = self.dev_a_qwp.get_angle()
                case 2:
                    end_pos = self.dev_b_hwp.get_angle()
                case 3:
                    end_pos = self.dev_b_qwp.get_angle()
        else:
            end_pos = 0

        if self.ttag_stream:
            end_counts = self.generate_samples()
            h_counts_end, v_counts_end = (
                len(end_counts) - np.sum(end_counts),
                np.sum(end_counts),
            )

        active_times[wp_num].append(
            (
                start,
                end,
                set_angle,
                start_pos,
                end_pos,
                h_counts_start,
                v_counts_start,
                h_counts_end,
                v_counts_end,
            )
        )
        self.num_buffer += 1
        if self.num_buffer >= 500:
            with open(self.timing_file, "ab") as f:
                pickle.dump(active_times, f)
            active_times = [[] for _ in range(4)]
            self.num_buffer = 0

        if self.debug:
            print(
                f"Time taken for operation {func.__name__} on WP {wp_num}: {end - start:.2f} seconds"
            )
            print(f"Moved WP {wp_num} from {start_pos} to {end_pos} degrees")
            print(f"Set angle to {set_angle} degrees" if set_angle else "")

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

    def device_setup(self, device: elliptec.Rotator):
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
            deviceVel = 32  # 50% of max velocity
        else:
            deviceVel = 50  # 70% of max velocity

        if self.test_mode:
            return {"device_vel": deviceVel}
        else:
            status = device.send_instruction(b"sv", str(deviceVel))  # Set velocity
        if self.debug:
            print("Device status: ", status)
            print(
                f"{device.info['Motor Type']} (S/N {device.info['Serial No.']}) is ready to use."
            )
            print()

            return device.info

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
                print(e)
                self.ttag_type = TimeTaggerType.NONE
                self.detector_setup()
                return {}

            self.ttag = TT.createTimeTagger()
            self.ttag_stream = TT.TimeTagStream(
                self.ttag,
                1.5 * self.shots,
                channels=[self.ttag_channel_reflect, self.ttag_channel_transmit],
            )
            self.counter = TT.Countrate(
                self.ttag,
                channels=[self.ttag_channel_reflect, self.ttag_channel_transmit],
            )
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
            # counter.getData() returns per-second average
            self.shot_rate = (
                self.counter.getData()
            )  # Get average shot rate from the channel
            print("Shot rates:")
            print(
                f"T rate: {self.shot_rate[0]:.2f} counts/s. R rate: {self.shot_rate[1]:.2f} counts/s"
            )
            # self.shots is user input, calculate integration time based on shot rate from the detector
            self.integration_time = self.shots / (np.sum(self.shot_rate))
            print(f"Set integration time to: {self.integration_time:.5f} seconds.")
            return {}

    def device_home(self, device: elliptec.Rotator):
        """Sends the specified device to the home position.

        :param device: A cage rotator device.
        :type device: CageRotator object

        :returns: None
        """
        if self.debug:
            print(
                "Homing device S/N "
                + device.info["Serial No."]
                + ", address "
                + device.info["Address"]
            )
        if not self.test_mode:
            device.home()

        if self.debug:
            print("Device S/N " + device.info["Serial No."] + " is homed.")

        return True

    def take_background(self):
        """Takes background by moving the device to the specified angle that blocks
        the beam.

        :param device: A cage rotator device.
        :type device: CageRotator object

        :returns: None
        """
        if not self.test_mode:
            print("Taking background measurement, moving")
            for i, device in enumerate(
                [self.dev_a_hwp, self.dev_a_qwp, self.dev_b_hwp, self.dev_b_qwp]
            ):
                print(
                    "device S/N " + device.info["Serial No."] + " to background angle",
                    self.wp_details[i]["background_angle"],
                )
                self.device_move_to_angle(
                    device, self.wp_details[i]["background_angle"]
                )

            print("Taking background measurement for 3 seconds...")
            self.counter.startFor(int(3 * 1e12))  # Take data for 3 second
            self.counter.waitUntilFinished()
            # counter.getData() returns per-second average
            self.background_shot_rate = (
                self.counter.getData()
            )  # Get average shot rate from the channel
            print("Background Shot rates:")
            print(
                f"T rate: {self.background_shot_rate[0]:.2f} counts/s. R rate: {self.background_shot_rate[1]:.2f} counts/s"
            )

        return self.background_shot_rate.tolist()

    def get_shot_rate(self):
        if not self.test_mode:
            print("Taking background measurement, moving")
            for i, device in enumerate(
                [self.dev_a_hwp, self.dev_a_qwp, self.dev_b_hwp, self.dev_b_qwp]
            ):
                print(
                    "device S/N " + device.info["Serial No."] + " to zero angle",
                    self.wp_details[i]["angle_offset"],
                )
                device.set_angle(self.wp_details[i]["angle_offset"])

            print("Setting bHWP to 22.5 to balance detector counts")
            self.device_move_to_angle(self.dev_b_hwp, self.b_hwp_offset + 22.5)

            print("Taking measurement for 3 seconds...")
            self.counter.startFor(int(3 * 1e12))  # Take data for 3 second
            self.counter.waitUntilFinished()
            # counter.getData() returns per-second average
            self.shot_rate = (
                self.counter.getData()
            )  # Get average shot rate from the channel
            print("Shot rates:")
            print(
                f"T rate: {self.shot_rate[0]:.2f} counts/s. R rate: {self.shot_rate[1]:.2f} counts/s"
            )

        return self.shot_rate.tolist()

    def device_move_to_angle(self, device: elliptec.Rotator, angle):
        """Rotates a device to the specified angle.

        :param device: A cage rotator device.
        :type device: CageRotator object

        :param angle: The desired angular position (in degrees)
        :type angle: float

        :returns: None
        """
        if self.test_mode:
            print("Moved device S/N " + device + " to position " + str(angle))
        else:
            angle_ret = None
            while angle_ret is None:
                angle_ret = device.set_angle(
                    angle
                )  # This will block until the device reaches the desired angle
                if angle_ret is None:
                    print(
                        "Angle return is None when setting",
                        device.info["Serial No."],
                        "to ",
                        angle,
                    )
                    print("Trying again...")
                    time.sleep(0.01)
            if self.debug:
                print(
                    "Moved device S/N "
                    + device.info["Serial No."]
                    + " to position "
                    + str(angle)
                )

    def device_move_by_angle(self, device: elliptec.Rotator, angle):
        """Rotates a device by the specified angle.

        :param device: A cage rotator device.
        :type device: CageRotator object

        :param angle: The desired angular position (in degrees)
        :type angle: float

        :returns: None
        """
        if self.test_mode:
            print("Moved device S/N " + device + " by " + str(angle))
        else:
            device_info = device.get_device_info()
            device_sn = str(device_info.serial_no)
            curr_angle = device.get_angle()
            if (curr_angle + angle) > 360:
                angle = angle - 360
            device.set_angle(
                device.get_position() + angle
            )  # Moves by position in degrees
            if self.debug:
                print(
                    "Moved device S/N "
                    + device.info["Serial No."]
                    + " to position "
                    + device.get_angle()
                )
            device.close_connection()

    def disconnect(self):
        """Disconnects all Kinesis devices."""
        if not self.test_mode:
            # Code must gracefully disconnect from device at its conclusion.
            # Otherwise, the Connect() method will throw an error when the code is rerun

            # Close the connection to the controller

            self.dev_a_hwp.controller.close_connection()
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

    def apply_parallel(self, operations, rotations=[]):
        all_ops = operations + rotations if rotations else operations
        dev_list = [self.dev_a_hwp, self.dev_a_qwp, self.dev_b_hwp, self.dev_b_qwp]
        angles = []
        for i, op in enumerate(all_ops):
            angle = op.parameters[0] * 360 / (2 * np.pi)
            if op.name == "HWPRotStage":
                angles.append((angle % 90) + float(self.wp_details[i]["angle_offset"]))
            elif op.name == "QWPRotStage":
                angles.append((angle % 180) + float(self.wp_details[i]["angle_offset"]))
            else:
                print("Operation : ", op.name, ", is not supported")

        self.time_parallel_rotation(
            self.device_move_to_angle, self.active_times, dev_list, angles
        )

    def apply(self, operations, rotations=[]):
        """Instucts the Kinesis devices to perform the specified operations.

        This function extends a built-in PennyLane device method.

        :param operations: A list of operations supported by the kinesis cage rotator device.
        :type operations: A PennyLane Operataion
        """

        if isinstance(self.motors.controller, list):
            return self.apply_parallel

        all_ops = operations + rotations if rotations else operations

        # # TODO: more generic management of device operations
        dev_list = [self.dev_a_hwp, self.dev_a_qwp, self.dev_b_hwp, self.dev_b_qwp]
        for i, op in enumerate(all_ops):
            angle = op.parameters[0] * 360 / (2 * np.pi)
            if op.name == "HWPRotStage":
                self.time_rotation(
                    self.device_move_to_angle,
                    self.active_times,
                    i,
                    dev_list[i],
                    ((angle % 90) + float(self.wp_details[i]["angle_offset"])),
                )
            elif op.name == "QWPRotStage":
                self.time_rotation(
                    self.device_move_to_angle,
                    self.active_times,
                    i,
                    dev_list[i],
                    ((angle % 180) + float(self.wp_details[i]["angle_offset"])),
                )
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
                    _ = (
                        self.ttag_stream.getData()
                    )  # Clear the buffer (it will return stored data otherwise)
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
            # channels = channels[: self.shots]  # truncate to desired number of shots
            h_counts = len(channels[channels == self.ttag_channel_transmit])
            v_counts = len(channels[channels == self.ttag_channel_reflect])
        counts = h_counts + v_counts
        if self.debug:
            print()
            print("Measured Singles:")
            print(f"H counts: {h_counts}\t V counts: {v_counts}")

            print("total counts : ", counts)

        return np.concatenate(
            (
                np.ones((int(v_counts), 1), dtype=int),
                np.zeros((int(h_counts), 1), dtype=int),
            )
        )
