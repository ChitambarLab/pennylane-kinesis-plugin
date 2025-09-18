import numpy as np
import os
import argparse
from pennylane_kinesis import swabian, datetime_now_string
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Take counts over a period of time",
        description="Take counts over a period of time using the Swabian Time Tagger.",
    )

    parser.add_argument(
        "-t",
        "--time",
        default=60,
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

    dir = "./data/counts/"
    args = parser.parse_args()

    counter = swabian.Counter(
        shots=args.shots, shot_rate=3000000, integration_time=args.integration
    )
    counter.detector_setup()
    if args.auto:
        counter.auto_shot_rate()
        counter.auto_integration_time()

    print("Starting to take counts...")
    start_time = time.time()
    date_time = datetime_now_string()

    while True:
        counts = counter.get_counts()

        with open(f"{dir}counts_{date_time}.txt", "a") as f:
            f.write(f"{counts[0]},{counts[1]}\n")

        if time.time() - start_time > args.time:
            break
