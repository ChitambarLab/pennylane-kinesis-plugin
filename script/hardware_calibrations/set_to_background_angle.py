import sys
import glob
import serial
import elliptec
from pennylane_kinesis import read_config
import argparse

ports = [f"/dev/ttyUSB{i}" for i in range(4)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount", "-m", default=0, type=int)
    parser.add_argument("--zero", default=False, action="store_true")
    parser.add_argument("--angle", default=None, type=float)
    parser.add_argument("--raw", default=False, action="store_true")
    args = parser.parse_args()
    conf = read_config("./src/pennylane_kinesis/2025_09_12_timur_elliptec_config.txt")

    mount = int(args.mount)
    print("background_angle is ", conf[mount]["background_angle"])

    controller = elliptec.Controller(ports[mount], timeout=3, write_timeout=3)
    device = elliptec.Rotator(controller, address=str(0))
    if args.zero:
        device.set_angle(float(conf[mount]["angle_offset"]))
    elif args.angle:
        if args.raw:
            device.set_angle(float(args.angle))
        else:
            device.set_angle(float(args.angle) + float(conf[mount]["angle_offset"]))
    else:
        device.set_angle(float(conf[mount]["background_angle"]))

    print(f"Set device {args.mount} to {device.get_angle()}")


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
