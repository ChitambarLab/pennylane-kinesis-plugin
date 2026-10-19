import numpy as np
import os
import argparse
from pennylane_kinesis import swabian, datetime_now_string, ElliptecMotors
import time
from pathlib import Path


def move_to_angle(device, angle, offset_angle):
    angle_ret = None
    # print(f"Moving to {angle}")
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

    return angle_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Take counts over a period of time",
        description="Take counts over a period of time using the Swabian Time Tagger.",
    )

    parser.add_argument(
        "-t",
        "--time",
        default=None,
        type=int,
        help="Time to take counts for in seconds.",
    )

    parser.add_argument(
        "-s",
        "--shots",
        default=None,
        type=int,
        help="Number of shots to take. Default: None. (will not truncate counts)",
    )

    parser.add_argument(
        "-i",
        "--integration",
        default=1,
        type=float,
        help="Integration time in seconds.",
    )

    parser.add_argument(
        "-a",
        "--auto",
        default=False,
        action="store_true",
        help="Automatically set the integration time based on calculated shot rate.",
    )

    parser.add_argument("--num_trials", default=None, type=int, help="Number of trials to take counts for.")

    parser.add_argument(
        "--repeatability", default=None, type=int, help=f"Repeatable angle changes on (int) mount. Goes to angle 22.5"
    )

    parser.add_argument(
        "--angle", default=22.5, type=float, help="Use with --repeatability. Which angle to set mount to."
    )

    dir = "./data/counts/"
    Path(dir).mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()

    counter = swabian.Counter(shots=args.shots, integration_time=args.integration)
    counter.detector_setup()
    if args.auto:
        counter.auto_shot_rate()
        counter.auto_integration_time()

    counter.shots = None
    print("Starting to take counts...")
    start_time = time.time()
    date_time = datetime_now_string()
    filepath = f"{dir}counts_{date_time}.txt"
    total_counts = []

    try:
        if args.repeatability is not None:
            motors = ElliptecMotors()
            hwp_sweep_angles = np.linspace(0, 180, 180)
            qwp_sweep_angles = np.linspace(0, 180, 180)
            rotators = [motors.prep_hwp, motors.prep_qwp, motors.meas_hwp, motors.meas_qwp]
            background_angles = [motors.wp_details[i]["background_angle"] for i in range(len(rotators))]
            offset_angles = [motors.wp_details[i]["angle_offset"] for i in range(len(rotators))]
        while True:
            if args.time and time.time() - start_time > args.time:
                break
            if args.num_trials and len(total_counts) >= args.num_trials:
                break

            if args.repeatability is not None:
                move_to_angle(rotators[args.repeatability], 0, offset_angles[args.repeatability])
                move_to_angle(rotators[args.repeatability], args.angle, offset_angles[args.repeatability])

            counts = counter.get_counts()
            total_counts.append(counts)

        print("Saving counts")
        total_counts = np.array(total_counts)
        np.savetxt(filepath, total_counts, fmt=["%d", "%d"], delimiter=",")
    except KeyboardInterrupt:
        print("Saving partial counts")
        total_counts = np.array(total_counts)
        np.savetxt(filepath + ".aborted", total_counts, fmt=["%d", "%d"], delimiter=",")
