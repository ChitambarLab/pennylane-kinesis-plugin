"""Print the timetagger counts live (for manual alignment tasks)."""

from pennylane_kinesis import swabian
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration_time", "-t", default=1, type=float)

    args = parser.parse_args()

    counter = swabian.Counter()
    counter.set_integration_time(args.integration_time)
    counter.detector_setup()

    while True:
        print(counter.get_counts())
