import argparse
import os
from pennylane_kinesis import KinesisCageRotatorDevice, KinesisMotors, TimeTaggerType, swabian
import time
from pylablib.devices import Thorlabs

_WIMP_MODE = False
_TEST_MODE = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Manual alignment",
        description="Manually align the rotation stages to the desired angles.",
    )
    motors = KinesisMotors()

    def device_setup(device: Thorlabs.KinesisMotor):
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
        if _WIMP_MODE:
            deviceVel = 5
        else:
            deviceVel = 25

        if _TEST_MODE:
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

    for wp in motors.wps:
        device_setup(wp)

    tt = swabian.Counter()
    tt.detector_setup()

    def handle_input(move_type, motor):
        print("Enter an angle to move to (in degrees)")
        angle = float(input())
        if move_type == "a":
            move_absolute(motor, angle)
        elif move_type == "r":
            move_relative(motor, angle)
        else:
            print("Invalid input.")

    def move_absolute(motor, angle):
        motors.wps[int(motor) - 1].move_to(angle)
        motors.wps[int(motor) - 1].wait_for_stop()
        print(f"Moved motor {motor} to {angle} degrees.")
        print(f"Current angle: {motors.wps[int(motor)-1].get_position()}.")

    def move_relative(motor, angle):
        motors.wps[int(motor) - 1].move_by(angle)
        motors.wps[int(motor) - 1].wait_for_stop()
        print(f"Moved motor {motor} by {angle} degrees.")
        print(f"Current angle: {motors.wps[int(motor)-1].get_position()}.")

    print("Started motors.")

    try:
        while True:
            print("Manual Control of motors.")
            print("Press q to quit.")
            print("Press 1 to move preparation HWP")
            print("Press 2 to move preparation QWP")
            print("Press 3 to move measurement HWP")
            print("Press 4 to move measurement QWP")
            counts = tt.get_counts()

            key = input()
            if key == "q":
                for wp in range(len(motors.wps)):
                    print(f"WP {wp+1} is at {motors.wps[wp].get_position()} degrees.")
                break
            elif key in ["1", "2", "3", "4"]:
                print("Press a to move to an absolute angle or r to move to a relative angle.")
                move_type = input()
                if move_type in ["a", "r"]:
                    handle_input(move_type, key)
    except KeyboardInterrupt:
        for wp in range(len(motors.wps)):
            print(f"WP {wp+1} is at {motors.wps[wp].get_position()} degrees.")
        exit()
