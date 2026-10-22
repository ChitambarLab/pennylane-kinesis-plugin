import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import os
import argparse
import time
import numpy as np
import pennylane_kinesis as pk
from pennylane_kinesis.file_utilities import datetime_now_string
import itertools
import cProfile, pstats

try:
    # If you're running this on hardware that doesn't have the swabian timetagger library
    # such as when you're doing numerics work, this import will break
    from pennylane_kinesis.tomography_utilities import take_tomography_data
except:
    pass
import datetime
from pathlib import Path
from tqdm import tqdm

MAXIMUM_BOUND_3_BIT = 24
QUANTUM_COST_BOUND_3_BIT = 12 + 12 / np.sqrt(3)
CLASSICAL_COST_BOUND_3_BIT = 18

QUANTUM_BOUND_3_BIT = QUANTUM_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT
CLASSICAL_BOUND_3_BIT = CLASSICAL_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT

MAXIMUM_BOUND_2_BIT = 8
QUANTUM_COST_BOUND_2_BIT = 4 + 2 * np.sqrt(2)
CLASSICAL_COST_BOUND_2_BIT = 6

QUANTUM_BOUND_2_BIT = QUANTUM_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT
CLASSICAL_BOUND_2_BIT = CLASSICAL_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT

BACKGROUND_2_BIT_MULTIPLIER = 1 / 2 + QUANTUM_COST_BOUND_2_BIT / MAXIMUM_BOUND_2_BIT
BACKGROUND_3_BIT_MULTIPLIER = 1 / 2 + QUANTUM_COST_BOUND_3_BIT / MAXIMUM_BOUND_3_BIT


def print_bar(msg):
    bar_length = len(msg) + 15  # Set the bar length to match the message length
    bar = "#" * bar_length  # Generate the bar dynamically
    padding = (len(bar) - len(msg)) // 2

    formatted_msg = f"{'#' * padding}{msg}{'#' * (len(bar) - len(msg) - padding)}"

    print(bar)
    print(formatted_msg)
    print(bar)


if __name__ == "__main__":
    NUM_STEPS = 60
    NUM_SHOTS = 10000
    NUM_TRIALS = 40

    PROTOCOLS = ["state_discrimination_2wp", "rac2", "state_discrimination_4wp", "rac3"]
    STATE_DISC = ["state_discrimination_2wp", "state_discrimination_4wp"]
    RAC = ["rac2", "rac3"]

    GRAD_METHOD = ["parameter-shift", "finite-diff"]

    datetime_str = datetime_now_string()
    data_path = "./data/measurement_plan_run/" + datetime_str + "/"

    measurement_plan_state = list(itertools.product(STATE_DISC, GRAD_METHOD))
    measurement_plan_rac = list(itertools.product(RAC, GRAD_METHOD))
    measurement_plan = [item for pair in zip(measurement_plan_state, measurement_plan_rac) for item in pair]
    print("Measurement plan:", measurement_plan)
    j = 0
    for i in tqdm(range(NUM_TRIALS)):
        for protocol, grad_method in tqdm(measurement_plan, leave=False):
            print(f"Running trial number {i} optimization for measurement: {protocol},{grad_method}")
            print(protocol)
            print(data_path)
            filepath = data_path + protocol + "/" + grad_method + "/"
            prof = cProfile.Profile()
            print("Starting Profiler")
            prof.enable()

            Path(filepath).mkdir(parents=True, exist_ok=True)

            if protocol.startswith("state_discrimination"):
                if protocol.endswith("2wp"):
                    two_wp = True
                else:
                    two_wp = False

                start_time = time.time()
                state_disc_opt_dict = pk.two_state_discrimination(
                    filepath=filepath,
                    shots=NUM_SHOTS,
                    two_wp=two_wp,
                    numerics=False,
                    step_size=0.2,
                    num_steps=NUM_STEPS,
                    verbose=False,
                    optimizer="adagrad",
                    diff_method=grad_method,
                )

                end_time = time.time()

                print(
                    "took",
                    end_time - start_time,
                    f"seconds to run VQO on {protocol}, adagrad",
                    "and",
                    grad_method,
                    f", trial number {i}",
                )

                time_taken = end_time - start_time

                datetime_str = datetime_now_string()
                f = open(filepath + datetime_str + "runtime.txt", "a")
                f.write(str(end_time - start_time))

            else:
                if protocol.endswith("3"):
                    rac3 = True
                else:
                    rac3 = False

                start_time = time.time()
                state_disc_opt_dict = pk.rac(
                    filepath=filepath,
                    shots=NUM_SHOTS,
                    rac_3=rac3,
                    numerics=False,
                    step_size=0.2,
                    num_steps=NUM_STEPS,
                    verbose=False,
                    optimizer="adagrad",
                    diff_method=grad_method,
                )

                end_time = time.time()

                print(
                    "took",
                    end_time - start_time,
                    f"seconds to run VQO on {protocol}, adagrad",
                    "and",
                    grad_method,
                    f", trial number {i}",
                )

                time_taken = end_time - start_time

                datetime_str = datetime_now_string()
                f = open(filepath + datetime_str + "runtime.txt", "a")
                f.write(str(end_time - start_time))

            prof.disable()
            stats = pstats.Stats(prof)
            stats.print_stats("kinesis_simulator_device.py:")
            stats.print_stats("optimization.py:")

            background_rate = np.sum(state_disc_opt_dict["background_rate"])
            shot_rate = np.sum(state_disc_opt_dict["shot_rate"])
            fig, (ax1, ax2) = plt.subplots(ncols=2)

            fig.suptitle(f"VQO of {protocol} with \n" + "adagrad and " + grad_method)

            if protocol.startswith("rac"):
                if protocol.endswith("3"):
                    CLASSICAL_BOUND = CLASSICAL_BOUND_3_BIT
                    QUANTUM_BOUND = QUANTUM_BOUND_3_BIT
                    BACKGROUND_MULTIPLIER = BACKGROUND_3_BIT_MULTIPLIER
                    MAXIMUM_BOUND = MAXIMUM_BOUND_3_BIT

                else:
                    CLASSICAL_BOUND = CLASSICAL_BOUND_2_BIT
                    QUANTUM_BOUND = QUANTUM_BOUND_2_BIT
                    BACKGROUND_MULTIPLIER = BACKGROUND_2_BIT_MULTIPLIER
                    MAXIMUM_BOUND = MAXIMUM_BOUND_2_BIT

                num_steps = len(state_disc_opt_dict["cost_vals"])
                ax1.plot(
                    range(num_steps),
                    [CLASSICAL_BOUND] * num_steps,
                    label="Classical Bound",
                )

                ax1.plot(
                    range(num_steps),
                    [QUANTUM_BOUND] * num_steps,
                    label="Quantum Bound",
                )
                ax1.plot(
                    range(num_steps),
                    np.array(state_disc_opt_dict["cost_vals"]) * -1 / MAXIMUM_BOUND,
                    label="Optimization Trajectory",
                )

                ax2.semilogy(
                    range(num_steps),
                    [QUANTUM_BOUND - CLASSICAL_BOUND] * num_steps,
                    color="orange",
                )
                ax2.semilogy(
                    range(num_steps),
                    [QUANTUM_BOUND] * num_steps - np.array(state_disc_opt_dict["cost_vals"]),
                    label=f"Optimization Trajectory",
                )
                ax2.semilogy(
                    range(num_steps),
                    [(background_rate / shot_rate) * BACKGROUND_MULTIPLIER] * num_steps,
                    "k--",
                    label=f"Background Limit \n({background_rate:.2f} cps)",
                )
                ax2.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

                ax1.set_ylabel("Success Probability")
                ax1.set_xlabel("Optimization Iteration")

                ax2.set_ylabel("Distance to Quantum Bound")
                ax2.set_xlabel("Optimization Iteration")

                fig.legend(loc="lower center", bbox_to_anchor=(0.32, 0.15), fontsize=10)
                fig.tight_layout()

                plt.grid()
            else:
                ax1.plot(
                    range(len(state_disc_opt_dict["cost_vals"])),
                    -1 * qnp.array(state_disc_opt_dict["cost_vals"]),
                    label="Optimization Trajectory",
                )

                ax1.plot(
                    range(len(state_disc_opt_dict["cost_vals"])),
                    [2] * len(state_disc_opt_dict["cost_vals"]),
                    label="Bound",
                )

                ax2.semilogy(
                    range(len(state_disc_opt_dict["cost_vals"])),
                    [2] + qnp.array(state_disc_opt_dict["cost_vals"]),
                    label="_Optimization trajectory",
                )
                ax2.plot(
                    range((len(state_disc_opt_dict["cost_vals"]))),
                    [background_rate / shot_rate] * len(state_disc_opt_dict["cost_vals"]),
                    "k--",
                    label=f"Background Limit \n({background_rate:.2f} cps)",
                )

                ax1.set_ylabel(r"Two-State Discrimination")
                ax1.set_xlabel("Optimization Iteration")

                ax2.set_ylabel("Distance to Optimal State Discrimination Score")
                ax2.set_xlabel("Optimization Iteration")

                plt.grid()
                fig.legend(loc="lower center", bbox_to_anchor=(0.32, 0.15), fontsize=10)

                plt.tight_layout()

            filename = protocol
            filename += "-" + "adagrad"
            filename += "-" + grad_method
            filename += "-" + str(NUM_STEPS)

            output_file = pk.winapi_path(filepath + datetime_str + filename + ".png")
            plt.savefig(output_file)
            plt.close()
