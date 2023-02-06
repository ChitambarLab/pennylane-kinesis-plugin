from matplotlib import pyplot as plt
import numpy as np



def plot_trajectory_on_cost_contour(scan_dict, opt_dict):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.grid()

    fig.suptitle("Variational Quantum Optimization of\nFiber Transformation Inversion", fontweight="bold", fontsize=24)

    label_size = 14

    # make contour plot of cost landscape
    CS = ax1.contour(scan_dict["x_grid"], scan_dict["y_grid"], scan_dict["z_grid"])
    ax1.clabel(CS, inline=True, fontsize=10)
    ax1.set_xlabel(r"HWP Angle (Radians)", fontsize=label_size)
    ax1.set_ylabel(r"QWP Angle (Radians)", fontsize=label_size)

    ax1.set_xticks(
        np.arange(0,np.pi/2+0.1, np.pi/6), [r"$0$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$"]
    )
    ax1.set_yticks(
        np.arange(0,np.pi+0.1, np.pi/6),
        [r"$0$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$", r"$\frac{2\pi}{3}$", r"$\frac{5\pi}{6}$", r"$\pi$"]
    )

    # plot trajectory of optimization
    theta_settings = []
    phi_settings = []
    for settings in opt_dict["settings_history"]:
        theta_settings += [settings[0]]
        phi_settings += [settings[1]]

    ax1.plot(theta_settings, phi_settings, marker="o", label="Gradient Descent")
    ax1.plot(theta_settings[0], phi_settings[0], "rx", markersize=8, label="Starting Point")
    ax1.plot(
        theta_settings[-1],
        phi_settings[-1],
        color="lime",
        marker="v",
        linestyle="",
        markersize=8,
        label="Ending Point",
    )

    # plot log scale of optimization on right
    ax2.semilogy(
        range(len(opt_dict["cost_vals"])),
        opt_dict["cost_vals"],
        label="Gradient Descent",
        marker="o",
    )
    ax2.plot(0, opt_dict["cost_vals"][0], "rx", markersize=8, label="Starting Point")
    ax2.plot(
        len(opt_dict["cost_vals"]),
        opt_dict["cost_vals"][-1],
        color="lime",
        marker="v",
        linestyle="",
        markersize=8,
        label="Ending Point",
    )

    ax2.grid()
    ax2.set_ylabel("Detector Extinction Rate (Cost)", fontsize=label_size)
    ax2.set_xlabel("Gradient Descent Iteration", fontsize=label_size)

    plt.legend()

    plt.tight_layout()

    plt.show()

