import numpy as np
from matplotlib import pyplot as plt
import sys

import pennylane_kinesis as pk

"""
Optimizes a quantum violation of the simplest qubit dimensionality witness.

By default the script will run on hardware using 4 wave plates.

Use the `numerics` argument to run the script using the simulator backend.
Use the `three_wp` argument to run the script with fixed mneasurement waveplates.

```
$ python script/qubit_dimensionality_witnessing.py numerics
```
"""

if __name__ == "__main__":
    THREE_WP = False if len(sys.argv) == 1 else "three_wp" in sys.argv[1:]
    NUMERICS = False if len(sys.argv) == 1 else "numerics" in sys.argv[1:]

    dim_wit_opt_dict = pk.qubit_dimensionality_witnessing(
        filepath="./data/qubit_dimensionality_witnessing/",
        shots=10000,
        shot_rate=3000000,
        three_wp=THREE_WP,
        numerics=NUMERICS,
        step_size=0.02,
        num_steps=150,
        use_adam=True,
        verbose=True,
        fixed_settings=[0, 0],  # zero angle for the measurement QWP
    )

    """
    Plotting
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    fig.suptitle("VQO of Dimensionality Witness")

    ax1.plot(
        range(len(dim_wit_opt_dict["cost_vals"])),
        -1 * np.array(dim_wit_opt_dict["cost_vals"]),
        label="Optimization Trajectory",
    )
    ax1.plot(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [3] * len(dim_wit_opt_dict["cost_vals"]),
        label="Classical Bound",
    )

    ax1.plot(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [1 + 2 * np.sqrt(2)] * len(dim_wit_opt_dict["cost_vals"]),
        label="Quantum Bound",
    )

    ax1.legend()

    ax2.semilogy(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [1 + 2 * np.sqrt(2)] + np.array(dim_wit_opt_dict["cost_vals"]),
        label="optimization trajectory",
    )
    ax2.semilogy(
        range(len(dim_wit_opt_dict["cost_vals"])),
        [1 + 2 * np.sqrt(2) - 3] * len(dim_wit_opt_dict["cost_vals"]),
        label="classical bound",
    )

    ax1.set_ylabel(r"Dimensionality Witness Score")
    ax1.set_xlabel("Optimization Iteration")

    ax2.set_ylabel("Distance to Optimal Dimensionality Witness Score")
    ax2.set_xlabel("Optimization Iteration")

    plt.tight_layout()

    plt.show()
