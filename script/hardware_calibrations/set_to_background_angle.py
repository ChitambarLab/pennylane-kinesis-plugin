import sys
import glob
import serial
import elliptec
from pennylane_kinesis.file_utilities import read_config
import argparse

ports = [f"/dev/ttyUSB{i}" for i in range(4)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Manual alignment",
        description="Manually align the rotation stages to the desired angles.",
    )
    parser.add_argument("-m","--mount", default=0, type=int, help="Which mount to rotate.")
    parser.add_argument("--all", default=False, action="store_true", help="Set all waveplates to background (or calibrated zero angle).")
    parser.add_argument("--zero", default=False, action="store_true",help="Set waveplate to calibrated zero angle.")
    parser.add_argument("--angle", default=None, type=float, help="Set waveplate to angle + calibrated offset.")
    parser.add_argument("--raw", default=False, action="store_true", help="Set waveplate to raw angle (use with --angle argument).")
    
    args = parser.parse_args()
    conf = read_config()
    if args.all:
        mounts = []
        for port in ports:
            controller = elliptec.Controller(port, timeout=3, write_timeout=3)
            device = elliptec.Rotator(controller, address=str(0))
            mounts.append(device)
        
        for i, device in enumerate(mounts):
            if args.zero:
                print(f"Setting mount {i} to {float(conf[i]['angle_offset'])}")
                device.set_angle(float(conf[i]["angle_offset"]))
                print(f"Set device {i} to {device.get_angle()}")
            else:
                print(f"Setting {args.mount} to {float(conf[i]['background_angle'])}")
                device.set_angle(float(conf[i]["background_angle"]))
                print(f"Set device {i} to {device.get_angle()}")

    else:
        mount = int(args.mount)

        controller = elliptec.Controller(ports[mount], timeout=3, write_timeout=3)
        device = elliptec.Rotator(controller, address=str(0))

        if args.zero:
            print(f"Setting {args.mount} to {float(conf[mount]['angle_offset'])}")
            device.set_angle(float(conf[mount]["angle_offset"]))
        elif args.angle is not None:
            if args.raw:
                print(f"Setting {args.mount} to {float(args.angle)}")
                device.set_angle(float(args.angle))
            else:
                print(f"Setting {args.mount} to {float(args.angle) + float(conf[mount]['angle_offset'])}")
                device.set_angle(float(args.angle) + float(conf[mount]["angle_offset"]))
        else:
            print(f"Setting {args.mount} to {float(conf[mount]['background_angle'])}")
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
