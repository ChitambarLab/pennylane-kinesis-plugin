import elliptec
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elliptec Setup",
        description="Set up elliptec rotation mounts.",
    )

    parser.add_argument(
        "-a",
        "--address",
        help="What address to set for the newly connected rotation mount (assumes new one is at address 0).",
    )

    parser.add_argument(
        "-c",
        "--change_address",
        nargs=2,
        help="Change the address of a rotation mount (first is original addres, second is new address)",
    )

    parser.add_argument(
        "-p",
        "--port",
        help="Which serial port to use.",
        default="COM4",
    )

    parser.add_argument(
        "-t",
        "--test",
        help="Test if all rotators are connected.",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    controller = elliptec.Controller(args.port, timeout=3, write_timeout=3)

    if args.address:
        device = elliptec.Rotator(controller, address="0")
        print("Setting address to: ", args.address)
        device.change_address(str(args.address))
        print("Homing device.")
        device.home()

    elif args.change_address:
        (old_addr, new_addr) = args.change_address
        device = elliptec.Rotator(controller, address=str(old_addr))
        print("Setting address from", old_addr, "to: ", new_addr)
        device.change_address(str(new_addr))
        print("Homing device.")
        device.home()
    elif args.test:
        config_file = "./src/pennylane_kinesis/2024_06_04_timur_elliptec_config.txt"

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
            rot_mount["angle_offset"] = float(entries[3])
            rot_mount["name"] = entries[4]
            wp_details.append(rot_mount)

        print("Testing all devices.")
        for i in range(1, 5):
            print("Testing device: ", wp_details[i - 1]["name"])
            device = elliptec.Rotator(controller, address=str(i))
            print(
                "Device Serial Numbers match:",
                device.info["Serial No."] == wp_details[i - 1]["serial_number"],
            )
            device.set_angle(wp_details[i - 1]["angle_offset"])
            device.home()
