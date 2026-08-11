import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt
import sys
import argparse
import os
import pennylane_kinesis as pk
import time
import numpy as np
from pathlib import Path

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


def plot_data(opt_dict, rac_3, filepath, plot_filename):
    if rac_3:
        QUANTUM_BOUND = QUANTUM_BOUND_3_BIT
        QUANTUM_COST_BOUND = QUANTUM_COST_BOUND_3_BIT
        CLASSICAL_BOUND = CLASSICAL_BOUND_3_BIT
        BACKGROUND_MULTIPLIER = BACKGROUND_3_BIT_MULTIPLIER
        MAXIMUM_BOUND = MAXIMUM_BOUND_3_BIT
        title = "RAC Optimization (3 bits)"
    else:
        QUANTUM_BOUND = QUANTUM_BOUND_2_BIT
        QUANTUM_COST_BOUND = QUANTUM_COST_BOUND_2_BIT
        CLASSICAL_BOUND = CLASSICAL_BOUND_2_BIT
        BACKGROUND_MULTIPLIER = BACKGROUND_2_BIT_MULTIPLIER
        MAXIMUM_BOUND = MAXIMUM_BOUND_2_BIT
        title = "RAC Optimization (2 bits)"

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    # fig, ax2 = plt.subplots()

    fig.suptitle(
        title,  # \n (Dashed) Mean Adam GD vs. (Solid) Mean Standard GD",
        fontweight="bold",
        fontsize=14,
    )

    prob_success = (qnp.array(opt_dict["cost_vals"]) * -1) / MAXIMUM_BOUND
    num_steps = len(opt_dict["cost_vals"])
    background_rate = qnp.sum(opt_dict["background_rate"])
    shot_rate = qnp.sum(qnp.array(opt_dict["shot_rate"]))
    ax1.plot(
        range(num_steps),
        prob_success,
        # label=f"Average Optimization Trajectory ({len(opt_dicts)} trials)",
    )
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

    # ax1.fill_between(
    #     range(num_steps),
    #     (mean_cost_vals - std_err_cost),
    #     (mean_cost_vals + std_err_cost),
    #     alpha=1 / 4,
    #     color="C1",
    #     # label="Standard Error",
    # )

    # ax1.plot(
    #     range(num_steps),
    #     [QUANTUM_BOUND - (background_rate / shot_rate) * BACKGROUND_MULTIPLIER]
    #     * num_steps,
    #     "k--",
    # )

    # ax2.semilogy(
    #     range(num_steps),
    #     [QUANTUM_BOUND - best_cost_val] * num_steps,
    #     alpha=3 / 4,
    #     label="Best Cost",
    #     color="blue",
    # )

    ax2.semilogy(
        range(num_steps),
        # [QUANTUM_BOUND] * num_steps - mean_cost_vals,
        [QUANTUM_BOUND] * num_steps - prob_success,
        label=f"Optimization Trajectory",
    )
    ax2.semilogy(
        range(num_steps),
        [QUANTUM_BOUND - CLASSICAL_BOUND] * num_steps,
        color="orange",
        # label="Classical Limit",
    )
    # ax2.semilogy(
    #     range(num_steps),
    #     [(np.mean(background_rates)/np.mean(shot_rates))]*num_steps,
    #     "k--",
    #     label=f"Background Limit \n(Average {np.mean(background_rates):.2f} cps)",
    # )
    ax2.semilogy(
        range(num_steps),
        [(background_rate / shot_rate) * BACKGROUND_MULTIPLIER] * num_steps,
        "k--",
        label=f"Background Limit \n(Average {background_rate} cps)",
    )
    # ax2.fill_between(
    #     range(num_steps),
    #     [QUANTUM_BOUND] * num_steps - (mean_cost_vals - std_err_cost),
    #     [QUANTUM_BOUND] * num_steps - (mean_cost_vals + std_err_cost),
    #     alpha=1 / 4,
    #     label="Standard Error",
    #     color="C1",
    # )

    ax2.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

    ax1.set_ylabel("Success Probability")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Quantum Bound")
    ax2.set_xlabel("Optimization Iteration")

    fig.legend(loc="lower center", bbox_to_anchor=(0.32, 0.15))
    fig.tight_layout()

    plt.grid()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    Path.mkdir(Path(filepath + "/runtime_plots/"), exist_ok=True)
    plt.savefig(Path(filepath + f"/runtime_plots/{plot_filename}"))
    plt.close()


"""
Optimizes the random access coding advantage.

Arguments:
* `numerics` - runs on simulator device rather than on hardware 
"""

hwp_period = qnp.pi / 2
qwp_period = qnp.pi


def print_bar(msg):
    bar_length = len(msg) + 15  # Set the bar length to match the message length
    bar = "#" * bar_length  # Generate the bar dynamically
    padding = (len(bar) - len(msg)) // 2

    formatted_msg = f"{'#' * padding}{msg}{'#' * (len(bar) - len(msg) - padding)}"

    print(bar)
    print(formatted_msg)
    print(bar)


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


def hwp_unitary(settings, wires):
    zero_angle = 0  # set according to hardware

    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(zero_angle, wires=[wires[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Two state discrimination",
        description="Optimizes the state discrimination success probability for two states.",
    )

    parser.add_argument(
        "-2",
        "--two_wp",
        default=False,
        action="store_true",
        help="If only using two waveplates (fixed measurement waveplates).",
    )
    parser.add_argument(
        "-n",
        "--numerics",
        default=False,
        action="store_true",
        help="If using the simulator backend.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="More verbose output.",
    )

    parser.add_argument("--num_trials", default=40, help="number of trials to run.")

    parser.add_argument("-s", "--step_size", default=0.08, help="Step size to use.")

    parser.add_argument("-p", "--plot", default=None, help="Only plot given filename, do not run")

    parser.add_argument("--num_steps", default=20, help="Num steps to use per optimization.")

    parser.add_argument(
        "--background",
        default=[300, 100],
        nargs="+",
        type=int,
        help="Number of background counts, one per channel.",
    )

    parser.add_argument("--profile", default=False, action="store_true", help="Profile the code.")

    parser.add_argument(
        "-d",
        "--diff",
        default="parameter-shift",
        help="Gradient method to use. Can be: 'parameter-shift' or 'finite-diff'",
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        default="gradient-descent",
        help="Optimizer to use.\n Can be: 'gradient-descent' 'adam', 'momentum' or 'adagrad'",
    )

    parser.add_argument(
        "--shots",
        default=10000,
        type=int,
        help="Number of shots to take.",
    )

    parser.add_argument(
        "--test",
        default=False,
    )

    parser.add_argument(
        "--rac_3",
        default=False,
        action="store_true",
        help="If using the 3 qubit random access code.",
    )

    args = parser.parse_args()

    filepath = "./data/random_access_coding/"
    if args.numerics:
        filepath += "numerics/"
    else:
        filepath += "hardware/"

    num_trials = int(args.num_trials)
    datetime_str = pk.datetime_now_string()

    filepath = filepath + args.diff + "/" + args.optimizer + "/"
    script_start_time = pk.datetime_now_string()
    filepath += script_start_time + "/"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    pk.make_dir(filepath + "runtimes/")

    def noise_fn(settings, wires):
        kraus_ops = fiber_ops_03_08
        qml.QubitChannel(kraus_ops, wires=wires[0])

    for i in range(num_trials):
        shot_rate = qnp.array([200000, 200000])
        print_bar(f"Starting trial {i} out of {num_trials}")
        start_time = time.time()
        if args.numerics:
            data_path = pk.make_dir(filepath + "numerics/")
            shots = args.shots
            dev_name = "kinesis.simulator"
            additional_dev_kwargs = {
                "name": dev_name,
                "shots": shots,
                "shot_rate": shot_rate,
                "qwp_domain_period": qwp_period,
                "hwp_domain_period": hwp_period,
                "background_counts": args.background,
            }
        else:
            motors = pk.elliptec_rotator_device.ElliptecMotors(test_mode=args.test)
            data_path = pk.make_dir(filepath + "hardware/")
            shots = args.shots
            dev_name = "elliptec.rotator"
            additional_dev_kwargs = {
                "motors": motors,
                "name": dev_name,
                "filepath": filepath,
                "test_mode": args.test,
                "debug": args.verbose,
            }

        pk.make_dir(data_path)

        datetime_str = pk.datetime_now_string()

        # network ansatz class manages circuit construction and settings
        if args.rac_3:
            rac_ansatz = qnetvo.NetworkAnsatz(
                [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
                [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
                [
                    qnetvo.MeasureNode(
                        num_in=3,
                        num_out=2,
                        wires=[0],
                        ansatz_fn=arb_unitary,
                        num_settings=2,
                    )
                ],
                dev_kwargs={
                    "name": dev_name,
                    "shot_rate": shot_rate,
                    "shots": shots,
                    **additional_dev_kwargs,
                },
            )
        else:
            rac_ansatz = qnetvo.NetworkAnsatz(
                [qnetvo.PrepareNode(num_in=4, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
                [qnetvo.NoiseNode(wires=[0], ansatz_fn=noise_fn)],
                [qnetvo.MeasureNode(num_in=2, wires=[0], ansatz_fn=hwp_unitary, num_settings=1)],
                dev_kwargs={
                    "name": dev_name,
                    "shot_rate": shot_rate,
                    "shots": shots,
                    **additional_dev_kwargs,
                },
            )

        # Take background and shot_rate data
        device = qml.device(**rac_ansatz.dev_kwargs)
        shot_rate = device.shot_rate.tolist()
        background_rate = device.take_background()

        _qnode_kwargs = {"diff_method": args.diff}
        # construct a linear inequality cost function that
        # corresponds to the minimal qubit random access code.
        if args.rac_3:
            rac_cost = qnetvo.linear_probs_cost_fn(
                rac_ansatz,
                qnp.array(
                    [
                        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                    ]
                ),
                qnode_kwargs=_qnode_kwargs,
            )
        else:
            rac_cost = qnetvo.linear_probs_cost_fn(
                rac_ansatz,
                qnp.array(
                    [
                        [1, 0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 0, 0, 0, 1, 1],
                    ]
                ),
                qnode_kwargs=_qnode_kwargs,
            )

        behavior_fn = qnetvo.behavior_fn(rac_ansatz, qnode_kwargs=_qnode_kwargs)

        if args.rac_3:
            domain_periods = [hwp_period, qwp_period] * 11
        else:
            domain_periods = [hwp_period, qwp_period] * 4 + [hwp_period] * 2

        rac_opt_dict = pk.optimize(
            lambda settings: rac_cost(*settings),
            qnp.array(rac_ansatz.rand_network_settings()),
            background_rate=background_rate,
            shot_rate=shot_rate,
            verbose=args.verbose,
            behavior_fn=behavior_fn,
            shots=int(args.shots),
            numerics=args.numerics,
            step_size=float(args.step_size),
            num_steps=int(args.num_steps),
            diff_method=args.diff,
            background_counts=args.background,
            optimizer=args.optimizer,
            filepath=data_path,
            filename="optimization_" + datetime_str,
            domain_periods=domain_periods,
        )
        if not args.numerics:
            motors.close()

        end_time = time.time()
        time_taken = end_time - start_time

        print_bar(
            f"Took {time_taken} seconds to run VQO on RAC with {args.optimizer} and {args.diff} with {str(args.step_size)}."
        )
        f = open(filepath + "runtimes/" + datetime_str + "runtime.txt", "a")
        f.write(str(time_taken))
        f.close()
        time_remaining = time_taken * (num_trials - i)
        print_bar(
            f"At this rate, it will take {time_remaining:.3f} seconds or {time_remaining / 60:.3f} minutes or {time_remaining / 3600:.3f} hours."
        )

        filename = "rac"
        if args.rac_3:
            filename += "_3"
        else:
            filename += "_2"
        filename += "-" + args.diff
        filename += "-" + args.optimizer
        filename += "-" + str(args.num_steps)
        filename += "-" + str(args.step_size)
        if args.numerics:
            filename += "-numerics"
        if args.two_wp:
            filename += "-twowp"

        plot_data(rac_opt_dict, args.rac_3, filepath, datetime_str + filename + ".png")
        time.sleep(0.1)
