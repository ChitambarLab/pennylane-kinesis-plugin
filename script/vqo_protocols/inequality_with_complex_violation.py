import pennylane as qml
from pennylane import numpy as qnp
import qnetvo
from matplotlib import pyplot as plt

from dask.distributed import Client

import pennylane_kinesis as pk

hwp_period = qnp.pi / 2
qwp_period = qnp.pi


dev_kwargs = {
    "name": "kinesis.simulator",
    "shots": None,
    "background_counts": 0,
    "qwp_domain_period": qwp_period,
    "hwp_domain_period": hwp_period,
}

filepath = "./data/inequality_with_complex_violation/"


def arb_unitary(settings, wires):
    pk.HWPRotStage(settings[0], wires=[wires[0]])
    pk.QWPRotStage(settings[1], wires=[wires[0]])


def optimize_ansatz_fn(network_ansatz, inequality, domain_periods, **opt_kwargs):
    def opt_fn(placeholder_param):
        init_settings = qnp.array(network_ansatz.rand_network_settings())
        cost = qnetvo.linear_probs_cost_fn(network_ansatz, inequality, qnode_kwargs={"diff_method": "parameter-shift"})
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

    real_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1)],
        [qnetvo.MeasureNode(num_in=3, num_out=2, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1)],
        dev_kwargs=dev_kwargs,
    )

    cplx_state_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_in=3, num_out=2, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1)],
        dev_kwargs=dev_kwargs,
    )

    cplx_meas_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=pk.HWPRotStage, num_settings=1)],
        [qnetvo.MeasureNode(num_in=3, num_out=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        dev_kwargs=dev_kwargs,
    )

    cplx_ansatz = qnetvo.NetworkAnsatz(
        [qnetvo.PrepareNode(num_in=8, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        [qnetvo.MeasureNode(num_in=3, num_out=2, wires=[0], ansatz_fn=arb_unitary, num_settings=2)],
        dev_kwargs=dev_kwargs,
    )

    # inequalities = [
    #     (4, qnp.array([
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 0, 0, 0, 0],
    #     ])),
    #     (7, qnp.array([
    #         [0, 0, 1, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 1, 1],
    #     ])),
    #     (5, qnp.array([
    #         [0, 0, 0, 0, 1, 0, 1, 0, 1],
    #         [1, 1, 0, 0, 0, 1, 0, 0, 0],
    #     ])),
    #     (5, qnp.array([
    #         [0, 0, 1, 0, 1, 0, 1, 0, 0],
    #         [1, 0, 0, 0, 0, 1, 0, 1, 0],
    #     ])),
    #     (7, qnp.array([
    #         [0, 0, 0, 0, 1, 1, 1, 0, 1],
    #         [1, 1, 1, 1, 0, 0, 0, 1, 0],
    #     ])),
    # ]

    # large = (7, qnp.array([
    #     [0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0],
    #     [0,  1,  1,  1,  0,  0,  1,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    # ]))

    rac_3bit_tight = (
        8,
        qnp.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            ]
        ),
        "rac_3bit_tight/",
    )

    rac_3bit = (
        18,
        qnp.array(
            [
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            ]
        ),
        "rac_3bit/",
    )

    for inequality in [rac_3bit_tight, rac_3bit]:
        real_opt_fn = optimize_ansatz_fn(
            real_ansatz,
            inequality[1],
            [hwp_period] * 11,
            step_size=0.03,
            num_steps=100,
            verbose=False,
        )

        real_opt_jobs = client.map(real_opt_fn, range(n_workers))
        real_opt_dicts = client.gather(real_opt_jobs)

        real_min_cost = 0
        for opt_dict in real_opt_dicts:
            if real_min_cost > opt_dict["min_cost"]:
                real_min_cost = opt_dict["min_cost"]

        print("inequality: ", inequality)
        print("classical bound: ", inequality[0])
        print("real max violation: ", real_min_cost)

        cplx_state_opt_fn = optimize_ansatz_fn(
            cplx_state_ansatz,
            inequality[1],
            [hwp_period, qwp_period] * 8 + [hwp_period] * 3,
            step_size=0.03,
            num_steps=100,
            verbose=False,
        )

        cplx_state_opt_jobs = client.map(cplx_state_opt_fn, range(n_workers))
        cplx_state_opt_dicts = client.gather(cplx_state_opt_jobs)

        cplx_state_min_cost = 0
        for opt_dict in cplx_state_opt_dicts:
            if cplx_state_min_cost > opt_dict["min_cost"]:
                cplx_state_min_cost = opt_dict["min_cost"]

        print("cplx state max violation: ", cplx_state_min_cost)

        cplx_meas_opt_fn = optimize_ansatz_fn(
            cplx_meas_ansatz,
            inequality[1],
            [hwp_period] * 8 + [hwp_period, qwp_period] * 3,
            step_size=0.03,
            num_steps=100,
            verbose=False,
        )

        cplx_meas_opt_jobs = client.map(cplx_meas_opt_fn, range(n_workers))
        cplx_meas_opt_dicts = client.gather(cplx_meas_opt_jobs)

        cplx_meas_min_cost = 0
        for opt_dict in cplx_meas_opt_dicts:
            if cplx_meas_min_cost > opt_dict["min_cost"]:
                cplx_meas_min_cost = opt_dict["min_cost"]

        print("cplx meas max violation: ", cplx_meas_min_cost)

        cplx_opt_fn = optimize_ansatz_fn(
            cplx_ansatz,
            inequality[1],
            [hwp_period, qwp_period] * 11,
            step_size=0.03,
            num_steps=100,
            verbose=False,
        )

        cplx_opt_jobs = client.map(cplx_opt_fn, range(n_workers))
        cplx_opt_dicts = client.gather(cplx_opt_jobs)

        cplx_min_cost = 0
        for opt_dict in cplx_opt_dicts:
            if cplx_min_cost > opt_dict["min_cost"]:
                cplx_min_cost = opt_dict["min_cost"]

        print("cplx max violation: ", cplx_min_cost)

        pk.write_json(
            {
                "classical_bound": inequality[0],
                "game_matrix": inequality[1].tolist(),
                "real_min_cost": real_min_cost,
                "cplx_state_min_cost": cplx_state_min_cost,
                "cplx_meas_min_cost": cplx_meas_min_cost,
                "cplx_min_cost": cplx_min_cost,
            },
            filepath + inequality[2] + pk.datetime_now_string(),
        )
