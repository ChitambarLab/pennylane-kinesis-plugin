from pennylane_kinesis.tomography_utilities import take_tomography_data
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Process tomography collection",
        description="Collect counts for process tomography",
    )

    parser.add_argument(
        "-f",
        "--filepath",
        default="./data/process_tomography/",
        help="Path to the data folder.",
    )

    parser.add_argument(
        "-n",
        "--num_shots",
        default=None,
        type=int,
        help="Number of shots to take.",
    )

    parser.add_argument(
        "-t",
        "--integration_time",
        default=None,
        type=int,
        help="Integration time for the counter in s.",
    )

    parser.add_argument(
        "-r",
        "--repetitions",
        default=1,
        type=int,
        help="Number of times to repeat the measurement (to be averaged).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
    )
    parser.add_argument(
        "--state", default=False, action="store_true", help="If running state tomography instead."
    )

    args = parser.parse_args()

    take_tomography_data(
        args.filepath, args.num_shots, args.integration_time, args.repetitions, args.state
    )
