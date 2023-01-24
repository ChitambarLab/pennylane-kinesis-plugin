from pennylane import QubitDevice
import pennylane as qml
import numpy as np

import clr 
import sys
import time
import System

import ttag
import time
from datetime import datetime

# from .version import __version__
from pennylane_kinesis.version import __version__


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


class KinesisCageRotatorDevice(QubitDevice):
    """A PennyLane plugin device for integrating with Kinesis Cage Rotational Stages.

    This plugin device assumes a single qubit experiment with time-tagger detector device.
    """

    name = "Kinesis Cage Rotator Device"
    short_name = "kinesis.cage.rotator"
    pennylane_requires = "0.28.0"
    version = __version__
    author = "Brian Doolittle"

    operations = {"HWPRotStage", "QWPRotStage"}
    observables = {"Projector", "Identity", "PauliZ"}

    incidence_rate = 100000
    # init_state = np.array([1, 0])

    def __init__(
        self,
        wires,
        shots=1000,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        analytic=None,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        # self._state = self.init_state

    def deviceList(self):
        """Builds device list and prints detected devices.
        """
        #Build an internal list of all connected devices.
        DeviceManagerCLI.BuildDeviceList()
        #Get the device list size.
        DeviceManagerCLI.GetDeviceListSize() 
        #Get a list of all connected devices, by serial number.
        devices = DeviceManagerCLI.GetDeviceList()
        
        if len(devices) == 0:
            print("No devices detected.")
            # sys.exit()
        else:
            print("%d devices detected!" % len(devices))
            print("----------------------------------------------")
            i=1
            for device in devices:
                print("Device " + str(i) + ": S/N "+device)
                i += 1
    
    # Connect and initialize device with S/N SN
    # Returns a tuple: the device CageRotator object and configuration information
    def deviceSetup(self, SN):
        # Create CageRotator and connect to it
        device = CageRotator.CreateCageRotator(SN)
        device.Connect(SN) # Makes a connection to device with S/N SN
        if device.IsConnected:
            print("Connected to device S/N " + SN)
        else:
            print("Failed to connect device S/N " + SN)
        
        # If device is not initalized, wait for the given timeout period
        if not device.IsSettingsInitialized():
            try:
                #Wait for the device settings to be initialized. 
                #Argument is timeout period (ms).
                device.WaitForSettingsInitialized(5000) 
            except:
                print("Did not initalize: Device S/N " + SN)
                device.Disconnect(False)

        if device.IsSettingsInitialized():
            print("Initalized device S/N " + SN)
        else:
            print("Failed to initalize device S/N " + SN)
            device.Disconnect(False)
        
        #Start the device polling:
        #The polling loop requests regular status requests to the motor to ensure the 
        #program keeps track of the device.

        #The polling loop will request a status update every ms milliseconds.
        #NOTE the ms time interval starts on receipt of the previous status, so 
        #status messages will NOT stack up
        device.StartPolling(250)

        #Needs a delay so that the current enabled state can be obtained
        time.sleep(0.5) # wait for 500 ms

        device.EnableDevice() #Enable the device, otherwise any move is ignored

        #Needs a delay so that the device can be enabled
        time.sleep(0.5) # wait for 500 ms

        #Call LoadMotorConfiguration on the device to initialize the
        #DeviceUnitConverter object required for real world unit parameters.
        #Loads configuration information into channel
        deviceConfig = device.LoadMotorConfiguration(SN)
        #aHWPSettings = aHWP.MotorDeviceSettings # Not sure what this does yet

        #Set rotation mode
        # device.SetRotationModes(0x01,0x00) # unlimited direction and quickest
        
        #Set velocity
        # If waveplates are unreliable, can set velocity at near half to keep 
        # current load down so not over burden the USB hub (wimp mode)
        # Default value is 10 degrees/s, half is 5 degrees/s
        wimpMode = True
        if wimpMode:
            deviceVel = 5
        else:
            deviceVel = 10
        
        device.SetVelocityParams(System.Decimal(deviceVel), System.Decimal(deviceVel))
        device.SetHomingVelocity(System.Decimal(deviceVel))

        deviceInfo = device.GetDeviceInfo()
        print(deviceInfo.Name + " (S/N " + str(deviceInfo.SerialNumber) + ") is ready to use.")
        print()
        
        return (device, deviceConfig)
    

    #Inputs: CageRotator object device
    #Outputs: None
    def deviceHome(self, device):
        deviceInfo = device.GetDeviceInfo()
        deviceSN = str(deviceInfo.SerialNumber)
        
        #if False: #set it to home every time, just as a precaution
        if device.Status.IsHomed:
            print("Device S/N " + deviceSN + " is already homed")
            return
        else:    
            #Sends the motor to its home position. 
            # Argument is the timeout. Function will wait until the move completes 
            # or the timeout elapses, whichever comes first. 
            device.Home(60000)
            if device.Status.IsHomed:
                print("Homed device S/N " + deviceSN)
            else:
                print("Failed to home device S/N " + deviceSN)
                device.Disconnect(False)
    
        
    #Inputs: CageRotator object device, desired angular position (in degrees)
    #Outputs: None
    def deviceMoveToAngle(self, device, angle):
        deviceInfo = device.GetDeviceInfo()
        deviceSN = str(deviceInfo.SerialNumber)
        device.MoveTo(System.Decimal(angle), 60000) #Moves to position in degrees
        print("Moved device S/N " + deviceSN + " to position " + str(angle) + "\u00B0")

    def reset(self):
        super().reset()
        self._state = self.init_state

    @classmethod
    def capabilities(self):
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
        all_ops = operations + rotations if rotations else operations
        for operation in all_ops:
            mat = qml.matrix(operation)
            self._state = mat @ self._state

    def generate_samples(self):
        p0 = np.linalg.norm(self._state[0]) ** 2

        return np.random.choice([0, 1], size=(self.shots, 1), p=[p0, 1 - p0])


if __name__=="__main__":
    dev = KinesisCageRotatorDevice(wires=[0])

    #Initialize device list
    dev.deviceList()

    #Connect and initlaize devices 
    print()
    print("Status Updates:")
    print("----------------------------------------------")

    print("Reading configuration file...")
    print()

    configFile = "./src/pennylane_kinesis/2022_02_21_colin_wp_config.txt"

    with open(configFile) as f:
        linesConfig = f.readlines()

    wpDetails = [['','','',''],['','','',''], ['','','',''], ['','','',''], ['','','','']]

    i=0
    for line in linesConfig:
        entries = linesConfig[i].split("\t")
        wpDetails[i][0] = entries[0]
        wpDetails[i][1] = entries[1]
        wpDetails[i][2] = entries[2]
        wpDetails[i][3] = entries[3][:-1] #remove newline at end of line
        i += 1

    print("Configuration File:")
    print("----------------------------------------------")
    for line in wpDetails:
        print(*line)
    print()

    aHWPsn = wpDetails[1][1]
    aQWPsn = wpDetails[2][1]
    bHWPsn = wpDetails[3][1]
    bQWPsn = wpDetails[4][1]

    aHWPoff = float(wpDetails[1][2])
    aQWPoff = float(wpDetails[2][2])
    bHWPoff = float(wpDetails[3][2])
    bQWPoff = float(wpDetails[4][2])

    aHWP, aHWPConfig = dev.deviceSetup(aHWPsn) #a waveplates are 1550
    aQWP, aQWPConfig = dev.deviceSetup(aQWPsn)

    # Home all devices
    print("Homing all devices...")
    dev.deviceHome(aHWP)
    dev.deviceHome(aQWP)
    print()

    # Move HWP and QWP to desired angles (degrees)
    aHWPAngle = 0
    aQWPAngle = 0
    dev.deviceMoveToAngle(aHWP, aHWPAngle)
    dev.deviceMoveToAngle(aQWP, aQWPAngle)

    #######################################################
    # End of Program
    #######################################################

    # Code must gracefully disconnect from device at its conclusion.
    # Otherwise, the Connect() method will throw an error when the code is rerun
    aHWP.Disconnect(False) #Disconnects the device. If argument is True, shuts down device. 
    aQWP.Disconnect(False) 