import sys
import glob
import serial
import elliptec
from pennylane_kinesis import ElliptecMotors
import numpy as np
from pennylane_kinesis.file_utilities import read_config
import argparse
import time

ports = [f"/dev/ttyUSB{i}" for i in range(4)]

# controllers = [elliptec.Controller(ports[i], timeout=3, write_timeout=3) for i in range(len(ports))]


def move_to_angle(device, angle, offset_angle=0):
    angle_ret = None
    while angle_ret is None:
        angle_ret = device.set_angle(
            (angle + offset_angle) % 360
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


def get_mount(mount_num):
    print("Scanning devices...")
    wp_details = read_config()

    for control in controllers:
        scanned = elliptec.scan_for_devices(control, start_address=0, stop_address=0)
        # Each motor has it's own controller, so it should ideally only return
        # one device when scanned
        if len(scanned) > 0:
            device_serial = scanned[0]["info"]["Serial No."]
            print(device_serial)
            print(wp_details[mount_num]["serial_number"])
            if device_serial == wp_details[mount_num]["serial_number"]:
                print(f"Found device with serial {device_serial}")
                return elliptec.Rotator(control, address="0", debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Manual alignment",
        description="Manually align the rotation stages to the desired angles.",
    )
    parser.add_argument("-m", "--mount", default=0, type=int, help="Which mount to rotate.")
    parser.add_argument(
        "--all", default=False, action="store_true", help="Set all waveplates to background (or calibrated zero angle)."
    )
    parser.add_argument("--zero", default=False, action="store_true", help="Set waveplate to calibrated zero angle.")
    parser.add_argument("--angle", default=None, type=float, help="Set waveplate to angle + calibrated offset.")
    parser.add_argument(
        "--background", default=False, action="store_true", help="Set waveplate to calibrated zero angle."
    )
    parser.add_argument("--raw", default=False, action="store_true", help="ignore offsets and set to raw angle.")
    args = parser.parse_args()

    if not args.all:
        # rotators = [get_mount(args.mount)]
        # print(rotators)
        # rotator = rotators[0]
        # rotator.home()
        motors = ElliptecMotors()
        rotators = [motors.prep_hwp, motors.prep_qwp, motors.meas_hwp, motors.meas_qwp]
        rotator = rotators[args.mount]
    else:
        motors = ElliptecMotors()
        rotators = [motors.prep_hwp, motors.prep_qwp, motors.meas_hwp, motors.meas_qwp]

        for rotator in rotators:
            rotator.home()

    wp_details = read_config()

    # wp_details = motors.wp_details[args.mount]
    print(wp_details[args.mount])
    print(rotator.info["Serial No."])

    # move_to_angle(rotator, 0, wp_details["angle_offset"])
    conf = read_config()
    rotator.home()
    print(wp_details[args.mount]["serial_number"])
    if args.all:
        if args.background:
            for i, rotator in enumerate(rotators):
                print(wp_details[i]["background_angle"])
                move_to_angle(rotator, 0, wp_details[i]["background_angle"])
                print(f"Set device {args.mount} to {rotator.get_angle()}")
        else:
            for i, rotator in enumerate(rotators):
                print(wp_details[i]["angle_offset"])
                move_to_angle(rotator, 0, wp_details[i]["angle_offset"])
                print(f"Set device {args.mount} to {rotator.get_angle()}")

    else:
        mount = int(args.mount)
        if args.raw:
            move_to_angle(rotator, args.angle, 0)
        else:
            print(wp_details[args.mount]["angle_offset"])
            move_to_angle(rotator, args.angle, wp_details[args.mount]["angle_offset"])

        print(f"Set device {args.mount} to {rotator.get_angle()}")


def serial_ports():
    """Lists serial port names

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of the serial ports available on the system
    """
    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob("/dev/tty[A-Za-z]*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.*")
    else:
        raise EnvironmentError("Unsupported platform")

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


print(serial_ports())
