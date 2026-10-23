"""Plot the timetagger counts live (for manual alignment tasks)."""

from pennylane_kinesis import swabian
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration_time", "-t", default=1, type=float)

    args = parser.parse_args()

    counter = swabian.Counter()
    counter.set_integration_time(args.integration_time)
    counter.detector_setup()

    fig = plt.figure()
    ax = fig.add_subplot()

    y_range = [-1, 100000]
    xs = [0]
    ys = [[0, 0]]
    ax.set_ylim(y_range[0], y_range[1])

    max_len = 100

    (line,) = ax.plot(xs, ys[0][0], color="red")
    (line2,) = ax.plot(xs, ys[0][1], color="blue")
    txt = ax.text(0.8, 0.8, f"H:{ys[0][0]}\nV:{ys[0][1]}", transform=ax.transAxes)
    diff_txt = ax.text(0.8, 0.7, f"Difference:{abs(ys[0][0] - ys[0][1])}", transform=ax.transAxes)

    plt.grid()

    def animate(frame, xs, ys):
        counts = counter.get_counts()

        ys.append(counts)
        ys = ys[-max_len:]
        plot_ys = np.array(ys)
        xs = range(plot_ys.shape[0])

        ylimits = ax.get_ylim()
        if counts[0] < ylimits[0]:
            new_limit = abs(counts[0] - 0.1 * counts[0])
            ax.set_ylim(-new_limit, new_limit)
        if counts[0] > ylimits[1]:
            new_limit = counts[0] + 0.1 * counts[0]
            ax.set_ylim(-new_limit, new_limit)

        if counts[1] < ylimits[0]:
            new_limit = abs(counts[1] - 0.1 * counts[1])
            ax.set_ylim(-new_limit, new_limit)
        if counts[1] > ylimits[1]:
            new_limit = counts[1] + 0.1 * counts[1]
            ax.set_ylim(-new_limit, new_limit)

        ax.set_xticks(xs)

        line.set_data(xs, plot_ys[:, 0])
        line2.set_data(xs, plot_ys[:, 1])
        txt.set_text(f"H:{plot_ys[-1:, 0]}\nV:{plot_ys[-1:, 1]}")
        diff_txt.set_text(f"Difference:{abs(plot_ys[-1:, 0] - plot_ys[-1:, 1])}")

        return (line, line2)

    ani = animation.FuncAnimation(
        fig=fig,  # figure handle
        func=animate,  # function animate
        fargs=(xs, ys),  # function arguments
        interval=args.integration_time * 1000,  # function call interval in milliseconds,
        save_count=max_len,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Keyboard interrupt occurred")
        exit()
