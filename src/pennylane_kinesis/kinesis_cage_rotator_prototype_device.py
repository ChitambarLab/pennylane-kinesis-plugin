from pennylane import QubitDevice
import pennylane as qml
import numpy as np

import time

from .version import __version__

_TEST_MODE = False

try:
    import clr
    import System
    import sys

    # The Kinesis software needs to be downloaded in order to use this code.
    # IMPORTANT: Upon initial install of Kinesis, make sure to connect and operate
    # all of the rotation stages via the Kinesis GUI prior to running this code.
    # Otherwise code will throw errors relating to
    # "System.NullReferenceException: Object reference not set to an instance of an object."
    sys.path.append("C:\\Program Files\\Thorlabs\\Kinesis\\")

    # Note: This code is written for use with the K10CR1 rotation mount.
    # The full CageRotator API may be found in Kinesis Help, under
    # "CageRotator Member List"
    # More details may be found under "Integrated Stepper Motors
    # (K10CR1, LTS150, LTS300, L490MZ, MLJ050, MLJ150) "

    #######################################################
    # Load the required .NET assemblies via PythonNET
    #######################################################
    clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
    clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsUI")
    clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
    clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")

    from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
    from Thorlabs.MotionControl.IntegratedStepperMotorsUI import *
    from Thorlabs.MotionControl.GenericMotorCLI import *
    from Thorlabs.MotionControl.DeviceManagerCLI import *

    # importing the local time-tagger library
    from . import ttag

except BaseException as ex:
    msg_template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = msg_template.format(type(ex).__name__, ex.args)
    print(message)

    print("\nAn error occured while importing pythonnet, time tagger, or kinesis software")

    _TEST_MODE = True
    print("setting `_TEST_MODE=True`")


class KinesisCageRotatorPrototypeDevice(QubitDevice):
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

    name = "Kinesis Cage Rotator Prototype Device"
    short_name = "kinesis.cage.rotator.prototype"
    pennylane_requires = "0.29.1"
    version = __version__
    author = "Brian Doolittle and Colin Lualdi"

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
        ttag_channel=13,
        test_mode=_TEST_MODE,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self.test_mode = test_mode
        self.wimp_mode = wimp_mode
        self.ttag_channel = ttag_channel

        self.shot_rate = shot_rate  # mean num photons incident upon detector per second
        self.integration_time = shots / (self.shot_rate)

        # Initialize device list
        self.device_list()

        # Connect and initlaize devices
        print()
        print("Status Updates:")
        print("----------------------------------------------")

        print("Reading configuration file...")
        print()

        config_file = "./src/pennylane_kinesis/2022_02_21_colin_wp_config.txt"

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
            # Build an internal list of all connected devices.
            DeviceManagerCLI.BuildDeviceList()
            # Get the device list size.
            DeviceManagerCLI.GetDeviceListSize()
            # Get a list of all connected devices, by serial number.
            devices = DeviceManagerCLI.GetDeviceList()

            if len(devices) == 0:
                print("No devices detected.")
            else:
                print("%d devices detected!" % len(devices))
                print("----------------------------------------------")
                i = 1
                for device in devices:
                    print("Device " + str(i) + ": S/N " + device)
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
            deviceVel = 10

        if self.test_mode:
            return {"sn": SN}, {"device_vel": deviceVel}
        else:
            # Create CageRotator and connect to it
            device = CageRotator.CreateCageRotator(SN)
            device.Connect(SN)  # Makes a connection to device with S/N SN
            if device.IsConnected:
                print("Connected to device S/N " + SN)
            else:
                print("Failed to connect device S/N " + SN)

            # If device is not initalized, wait for the given timeout period
            if not device.IsSettingsInitialized():
                try:
                    # Wait for the device settings to be initialized.
                    # Argument is timeout period (ms).
                    device.WaitForSettingsInitialized(5000)
                except:
                    print("Did not initalize: Device S/N " + SN)
                    device.Disconnect(False)

            if device.IsSettingsInitialized():
                print("Initalized device S/N " + SN)
            else:
                print("Failed to initalize device S/N " + SN)
                device.Disconnect(False)

            # Start the device polling:
            # The polling loop requests regular status requests to the motor to ensure the
            # program keeps track of the device.

            # The polling loop will request a status update every ms milliseconds.
            # NOTE the ms time interval starts on receipt of the previous status, so
            # status messages will NOT stack up
            device.StartPolling(250)

            # Needs a delay so that the current enabled state can be obtained
            time.sleep(0.5)  # wait for 500 ms

            device.EnableDevice()  # Enable the device, otherwise any move is ignored

            # Needs a delay so that the device can be enabled
            time.sleep(0.5)  # wait for 500 ms

            # Call LoadMotorConfiguration on the device to initialize the
            # DeviceUnitConverter object required for real world unit parameters.
            # Loads configuration information into channel
            deviceConfig = device.LoadMotorConfiguration(SN)
            # aHWPSettings = aHWP.MotorDeviceSettings # Not sure what this does yet

            # TODO: Set rotation mode
            # device.SetRotationModes(0x01,0x00) # unlimited direction and quickest

            device.SetVelocityParams(System.Decimal(deviceVel), System.Decimal(deviceVel))
            device.SetHomingVelocity(System.Decimal(deviceVel))

            deviceInfo = device.GetDeviceInfo()
            print(deviceInfo.Name + " (S/N " + str(deviceInfo.SerialNumber) + ") is ready to use.")
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

    def device_home(self, device):
        """Sends the specified device to the home position.

        :param device: A cage rotaator device.
        :type device: CageRotator object

        :returns: None
        """
        if not self.test_mode:
            deviceInfo = device.GetDeviceInfo()
            deviceSN = str(deviceInfo.SerialNumber)

            # if False: #set it to home every time, just as a precaution
            if device.Status.IsHomed:
                print("Device S/N " + deviceSN + " is already homed")
                return
            else:
                # Sends the motor to its home position.
                # Argument is the timeout. Function will wait until the move completes
                # or the timeout elapses, whichever comes first.
                device.Home(60000)
                if device.Status.IsHomed:
                    print("Homed device S/N " + deviceSN)
                else:
                    print("Failed to home device S/N " + deviceSN)
                    device.Disconnect(False)

        return True

    def device_move_to_angle(self, device, angle):
        """Rotates a device to the specified angle.

        :param device: A cage rotaator device.
        :type device: CageRotator object

        :param angle: The desired angular position (in degrees)
        :type angle: float

        :returns: None
        """
        if self.test_mode:
            print("Moved device S/N " + device["sn"] + " to position " + str(angle) + "\u00B0")
        else:
            device_info = device.GetDeviceInfo()
            device_sn = str(device_info.SerialNumber)
            device.MoveTo(System.Decimal(angle), 60000)  # Moves to position in degrees
            print("Moved device S/N " + device_sn + " to position " + str(angle) + "\u00B0")

    def disconnect(self):
        """Disconnects all Kinesis devices."""
        if not self.test_mode:
            # Code must gracefully disconnect from device at its conclusion.
            # Otherwise, the Connect() method will throw an error when the code is rerun

            # TODO: manage dev a and dev b
            self.dev_a_hwp.Disconnect(
                False
            )  # Disconnects the device. If argument is True, shuts down device.
            self.dev_a_qwp.Disconnect(False)

            self.dev_b_hwp.Disconnect(
                False
            )  # Disconnects the device. If argument is True, shuts down device.
            self.dev_b_qwp.Disconnect(False)

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
        for op in all_ops:
            angle = op.parameters[0] * 360 / (2 * np.pi)

            # TODO: manage dev a and dev b
            # if op.name == "HWPRotStage":
            #     self.device_move_to_angle(self.dev_a_hwp, angle % 90)
            # elif op.name == "QWPRotStage":
            #     self.device_move_to_angle(self.dev_a_qwp, angle % 180)
            # else:
            #     print("Operation : ", op.name, ", is not supported")

            if op.name == "HWPRotStage":
                self.device_move_to_angle(self.dev_b_hwp, angle % 90)
            elif op.name == "QWPRotStage":
                self.device_move_to_angle(self.dev_b_qwp, angle % 180)
            else:
                print("Operation : ", op.name, ", is not supported")

    def generate_samples(self):
        """Constructs a sequence of sampled data from the device.
        This data can be used to construct probability distributions and expectations.

        Since our measurement only detects one polarization, we estimate the total number
        of counts to construct the relative probability distribution.
        """
        time.sleep(
            self.integration_time
        )  # give the buffer time to fill up, otherwise buffer will throw a warning

        if self.test_mode:
            singles_array = [int(self.shots / 2), 0, 0, 0]
        else:
            singles_array = self.buffer.singles(self.integration_time)

        print()
        print("Measured Singles:")
        print(singles_array)

        counts = singles_array[self.ttag_channel]  # set channel for timetagger device

        print("num counts : ", counts)
        print("max_shots_allowed : ", self.shots)
        print("detection probability : ", counts / self.shots)

        return np.concatenate(
            (
                np.ones((int(min(counts, self.shots)), 1), dtype=int),
                np.zeros((int(max(self.shots - counts, 0)), 1), dtype=int),
            )
        )
