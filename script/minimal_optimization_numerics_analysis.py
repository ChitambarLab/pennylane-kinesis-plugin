import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt

from dask.distributed import Client

import pennylane_kinesis as pk

num_shots = 10000
hwp_period = qnp.pi / 2
qwp_period = qnp.pi / 2


dev = qml.device(
    "kinesis.simulator",
    wires=[0],
    shots=num_shots,
    qwp_domain_period=qwp_period,
    hwp_domain_period=hwp_period,
)

filepath = "./data/min_optimization_numerics_analysis/"


scan_range = qnp.arange(0, qnp.pi, 1)


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


def optimize_ansatz_fn(network_ansatz, inequality, domain_periods, **opt_kwargs):
    def opt_fn(placeholder_param):
        init_settings = qnp.array(network_ansatz.rand_network_settings())
        cost = qnetvo.linear_probs_cost_fn(network_ansatz, inequality)
        opt_dict = pk.optimize(
            lambda settings: cost(*settings),
            init_settings,
            domain_periods=domain_periods,
            **opt_kwargs,
        )
        opt_dict["min_cost"] = min(opt_dict["cost_vals"])

        return opt_dict

    return opt_fn


if __name__ == "__main__":
    n_workers = 4
    client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

    """
    optimizing state discrimination
    """
    state_disc_min_costs = qnp.zeros((len(scan_range), len(scan_range), len(scan_range)))

    state_disc_opt_kwargs = {
        "step_size": 0.07,
        "num_steps": 30,
        "verbose": False,
    }

    for i, alpha in enumerate(scan_range):
        for j, beta in enumerate(scan_range):
            for k, gamma in enumerate(scan_range):
                print("alpha,beta,gamma : ", alpha, ",", beta, ",", gamma, ",")

                state_disc_ansatz = qnetvo.NetworkAnsatz(
                    [
                        qnetvo.PrepareNode(
                            num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2
                        )
                    ],
                    [
                        qnetvo.NoiseNode(
                            wires=[0],
                            ansatz_fn=lambda settings, wires: qml.ArbitraryUnitary(
                                [alpha, beta, gamma], wires=[wires[0]]
                            ),
                        )
                    ],
                    # [qnetvo.MeasureNode(num_out=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
                    [qnetvo.MeasureNode(wires=[0], num_out=2)],
                    dev_kwargs={
                        "name": "kinesis.simulator",
                        "shots": num_shots,
                        "background_counts": 0,
                    },
                )

                state_disc_opt_fn = optimize_ansatz_fn(
                    state_disc_ansatz,
                    qnp.eye(2),
                    [hwp_period, qwp_period] * 3,
                    **state_disc_opt_kwargs,
                )

                state_disc_opt_jobs = client.map(state_disc_opt_fn, range(n_workers))
                state_disc_opt_dicts = client.gather(state_disc_opt_jobs)

                for opt_dict in state_disc_opt_dicts:
                    if state_disc_min_costs[i, j, k] > opt_dict["min_cost"]:
                        state_disc_min_costs[i, j, k] = opt_dict["min_cost"]

                print("state disc min cost: ", state_disc_min_costs[i, j, k])
                print("state disc cost vals : ", state_disc_opt_dicts[0]["cost_vals"])

    pk.write_json(
        {
            "scan_range": scan_range.tolist(),
            "min_costs": state_disc_min_costs.tolist(),
            "opt_kwargs": state_disc_opt_kwargs,
            "shots": num_shots,
            "qwp_domain_period": qwp_period,
        },
        filepath + "twostate_discrimination/" + pk.datetime_now_string(),
    )

    """
    optimizing dimensionality witnessing
    """
    client.restart()

    dim_wit_min_costs = qnp.zeros((len(scan_range), len(scan_range), len(scan_range)))
    dim_wit_opt_kwargs = {
        "step_size": 0.03,
        "num_steps": 40,
        "verbose": False,
    }

    for i, alpha in enumerate(scan_range):
        for j, beta in enumerate(scan_range):
            for k, gamma in enumerate(scan_range):
                print("alpha,beta,gamma : ", alpha, ",", beta, ",", gamma, ",")

                dim_wit_ansatz = qnetvo.NetworkAnsatz(
                    [
                        qnetvo.PrepareNode(
                            num_in=3, wires=[0], ansatz_fn=arb_unitary, num_settings=2
                        )
                    ],
                    # [qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1)],
                    [
                        qnetvo.NoiseNode(
                            wires=[0],
                            ansatz_fn=lambda settings, wires: qml.ArbitraryUnitary(
                                [alpha, beta, gamma], wires=[wires[0]]
                            ),
                        )
                    ],
                    # [qnetvo.MeasureNode(num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
                    [
                        qnetvo.MeasureNode(
                            num_out=2, num_in=2, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1
                        )
                    ],
                    dev_kwargs={
                        "name": "kinesis.simulator",
                        "shots": num_shots,
                        "background_counts": 0,
                    },
                )

                dim_wit_opt_fn = optimize_ansatz_fn(
                    dim_wit_ansatz,
                    qnp.array(
                        [
                            [1, 1, 1, -1, -1, 0],
                            [-1, -1, -1, 1, 1, 0],
                        ]
                    ),
                    [hwp_period, qwp_period] * 3 + [hwp_period] * 2,
                    **dim_wit_opt_kwargs,
                )

                dim_wit_opt_jobs = client.map(dim_wit_opt_fn, range(n_workers))
                dim_wit_opt_dicts = client.gather(dim_wit_opt_jobs)

                for opt_dict in dim_wit_opt_dicts:
                    if dim_wit_min_costs[i, j, k] > opt_dict["min_cost"]:
                        dim_wit_min_costs[i, j, k] = opt_dict["min_cost"]

                print("min cost: ", dim_wit_min_costs[i, j, k])
                print("cos vals : ", dim_wit_opt_dicts[0]["cost_vals"])

    pk.write_json(
        {
            "scan_range": scan_range.tolist(),
            "min_costs": dim_wit_min_costs.tolist(),
            "opt_kwargs": dim_wit_opt_kwargs,
            "shots": num_shots,
            "qwp_domain_period": qwp_period,
        },
        filepath + "dimensionality_witnessing/" + pk.datetime_now_string(),
    )

    """
    optimizing random access coding
    """
    client.restart()

    rac_min_costs = qnp.zeros((len(scan_range), len(scan_range), len(scan_range)))

    rac_opt_kwargs = {
        "step_size": 0.05,
        "num_steps": 40,
        "verbose": False,
    }

    for i, alpha in enumerate(scan_range):
        for j, beta in enumerate(scan_range):
            for k, gamma in enumerate(scan_range):
                print("alpha,beta,gamma : ", alpha, ",", beta, ",", gamma, ",")

                rac_ansatz = qnetvo.NetworkAnsatz(
                    [
                        qnetvo.PrepareNode(
                            num_in=4, wires=[0], ansatz_fn=arb_unitary, num_settings=2
                        )
                    ],
                    # [qnetvo.PrepareNode(num_in=4, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1)],
                    [
                        qnetvo.NoiseNode(
                            wires=[0],
                            ansatz_fn=lambda settings, wires: qml.ArbitraryUnitary(
                                [alpha, beta, gamma], wires=[wires[0]]
                            ),
                        )
                    ],
                    # [qnetvo.MeasureNode(num_out=2, num_in=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
                    [
                        qnetvo.MeasureNode(
                            num_out=2, num_in=2, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1
                        )
                    ],
                    dev_kwargs={
                        "name": "kinesis.simulator",
                        "shots": num_shots,
                        "background_counts": 0,
                    },
                )

                rac_opt_fn = optimize_ansatz_fn(
                    rac_ansatz,
                    qnp.array(
                        [
                            [1, 0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0, 0, 1, 1],
                        ]
                    ),
                    [hwp_period, qwp_period] * 4 + [hwp_period] * 2,
                    **rac_opt_kwargs,
                )

                rac_opt_jobs = client.map(rac_opt_fn, range(n_workers))
                rac_opt_dicts = client.gather(rac_opt_jobs)

                for opt_dict in rac_opt_dicts:
                    if rac_min_costs[i, j, k] > opt_dict["min_cost"]:
                        rac_min_costs[i, j, k] = opt_dict["min_cost"]

                print("min cost: ", rac_min_costs[i, j, k])
                print("cost vals : ", rac_opt_dicts[0]["cost_vals"])

    pk.write_json(
        {
            "scan_range": scan_range.tolist(),
            "min_costs": rac_min_costs.tolist(),
            "opt_kwargs": rac_opt_kwargs,
            "shots": num_shots,
            "qwp_domain_period": qwp_period,
        },
        filepath + "random_access_coding/" + pk.datetime_now_string(),
    )

    print(rac_min_costs)
