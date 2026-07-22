import sys
import glob
import serial
import elliptec
from pennylane_kinesis.file_utilities import read_config
import argparse

ports = [f"/dev/ttyUSB{i}" for i in range(4)]


def set_velocity(velocity, device):
    return device.send_instruction(b"sv" + str(args.velocity).encode())


def get_velocity(device):
    return device.send_instruction(b"gv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elliptec Calibration",
        description="Set elliptec mount velocities.",
    )

    parser.add_argument("--mount", "-m", default=0, type=int, help="Which mount to set/get velocity for.")

    parser.add_argument(
        "-gv",
        "--get_velocity",
        default=False,
        action="store_true",
        help="Get the velocity from the motor at address specified.",
    )

    parser.add_argument(
        "-sv",
        "--set_velocity",
        default=False,
        action="store_true",
        help="Set the velocity of the motor at address specified.",
    )

    parser.add_argument(
        "--velocity",
        default=None,
        type=int,
        help="Velocity to set for the motor (with sv).",
    )
    args = parser.parse_args()

    conf = read_config()
    mount = int(args.mount)
    controller = elliptec.Controller(ports[mount], timeout=3, write_timeout=3)
    device = elliptec.Rotator(controller, address=str(0))

    ret = controller.send_instruction(b"sv64")
    print(device.info["Serial No."])
    print(ret)
    # exit()
    if args.get_velocity:
        print(f"Velocity of motor {args.mount} is {get_velocity(device)}")

    elif args.set_velocity:
        device.set_velocity(args.velocity)
        print(f"Set velocity of motor {args.mount} is {set_velocity(str(args.velocity), device)}")
    else:
        print("Need to specify either set or get velocity")

