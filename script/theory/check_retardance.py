import numpy as np
import scipy as sp
from pennylane_kinesis.operators import VariableHWPRotStage, VariableQWPRotStageA, VariableQWPRotStageB
import matplotlib.pyplot as plt
from qutip import Bloch
from pathlib import Path

# from qutip.ipynbtools import plot_animation
import imageio


def bloch_to_coords(rho: np.array):
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = 2 * rho[0, 0] - 1
    return np.array([x, y, z])


def qwp_mat(theta, retardance=np.pi/2):
    # off_retardance = retardance * np.pi / 2

    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(-1.0j * retardance/2)

    # If anything is not tensorflow, it has to be casted and then

    mat = [
        [c**2 + e * s**2, (1 - e) * c * s],
        [(1 - e) * c * s, e * c**2 + s**2],
    ]
    return np.array(mat)


def hwp_mat(theta, retardance=np.pi):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.exp(-1.0j * retardance/2)

    # If anything is not tensorflow, it has to be casted and then

    mat = [
        [c**2 + e * s**2, (1 - e) * c * s],
        [(1 - e) * c * s, e * c**2 + s**2],
    ]
    return np.array(mat)


def plot_setup(result=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    return fig, ax


def plot_result(result, n, fig=None, ax=None):
    global sphere
    if fig is None or ax is None:
        fig, axes = plot_setup(result)

    if not sphere:
        sphere = Bloch(axes=axes)
        sphere.vector_color = ["r"]

    sphere.clear()
    sphere.add_vectors(result[n])
    sphere.make_sphere()
    return ax.artists


def make_bloch_animation(retardance=1.0051):
    input_state = np.array([1, 0])
    input_rho = np.outer(input_state, input_state.conj().T)

    vectors = []
    vectors_1 = []
    angles = np.linspace(0, 2 * np.pi, 100)

    for i, angle in enumerate(angles):
        qwp = qwp_mat(angle, retardance=retardance)
        qwp_1 = qwp_mat(angle)
        vectors.append(
            bloch_to_coords(qwp @ qwp @ input_rho @ qwp.conj().T @ qwp.conj().T)
        )
        vectors_1.append(
            bloch_to_coords(qwp_1 @ qwp_1 @ input_rho @ qwp_1.conj().T @ qwp_1.conj().T)
        )

    folder_path = f"./bloch_animation/bloch_animation_{retardance:.4f}"

    Path.mkdir(Path(folder_path), exist_ok=True, parents=True)

    angles = np.linspace(0, 360, 100)
    for i, angle in enumerate(angles):
        fig, ax = plot_setup()
        sphere = Bloch(fig, None, view=[-angle, 45])
        sphere.vector_color = ["r"]
        sphere.add_points(np.array(vectors).T, meth="l", alpha=0.8)
        sphere.add_points(np.array(vectors_1).T, meth="l", colors="r", alpha=0.75)
        # sphere.add_vectors(vectors[i])
        sphere.make_sphere()
        plt.savefig(f"{folder_path}/{i}.png")
        plt.close()

    image_paths = Path(folder_path).glob("*.png")
    image_paths_sorted = sorted(image_paths, key=lambda x: float(x.stem))
    images = []
    for filename in image_paths_sorted:
        try:
            images.append(imageio.imread(filename))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    imageio.mimsave(f"{folder_path}/animation.gif", images, loop=0, duration=5)


def check_density_matrix(retardance=1.0052):
    qwp_1 = qwp_mat(np.pi / 4, 1)

    qwp = qwp_mat(np.pi / 4, retardance)
    input_state = np.array([1, 0])
    input_rho = np.outer(input_state, input_state.conj().T)

    output_state_1 = qwp_1 @ input_rho @ qwp_1.conj().T

    print(output_state_1)
    output_state = qwp @ input_rho @ qwp.conj().T
    print(output_state)

    expected_0_counts_1 = np.real(np.trace(input_rho @ output_state_1))
    expected_0_counts = np.real(np.trace(input_rho @ output_state))

    expected_1_counts = 1 - expected_0_counts
    expected_1_counts_1 = 1 - expected_0_counts_1

    print(
        f"Expected 0 counts: {expected_0_counts * 10000}\n Expected 1 counts: {expected_1_counts * 10000} \n Expected 0 counts perfect: {expected_0_counts_1 * 10000} \n Expected 1 counts perfect: {expected_1_counts_1 * 10000}"
    )

    output_rho = qwp @ qwp @ input_rho @ qwp.conj().T @ qwp.conj().T

    expected_0_counts = np.real(np.trace(input_rho @ output_rho))

    print(f"QWP: \n{qwp}")

    qwp_half = qwp_mat(np.pi / 4, retardance)
    qwp_half_neg = qwp_mat(-np.pi / 4, retardance)

    hwp = hwp_mat(np.pi / 4, retardance)
    hwp_1 = hwp_mat(np.pi / 4, 1)

    print(f"qwp2:\n{qwp_half @ qwp_half}")

    qwp_half_1 = qwp_mat(np.pi / 4, 1)

    print(f"qwp2:\n{qwp_half_1 @ qwp_half_1}")

    expected_0 = hwp_1 @ qwp_half_1 @ hwp_1 @ qwp_half_1 @ input_state
    expected_0 = np.outer(expected_0, expected_0.conj().T)
    print(
        f"Expected 1 counts: {(1 - np.real(np.trace(input_rho @ expected_0))) * 10000}"
    )

    expected_0 = hwp @ qwp_half @ hwp @ qwp_half @ input_state
    expected_0 = np.outer(expected_0, expected_0.conj().T)
    print(
        f"Expected 1 counts imperfect: {(1 - np.real(np.trace(input_rho @ expected_0))) * 10000}"
    )


def simulate_angle_sweep(retardance=1,phase=0):
    # angles = np.linspace(211*np.pi/180, np.pi+211*np.pi/180, 180)
    angles = np.linspace(0,np.pi,180)
    input_state = np.array([1, 0])
    # input_state = 1/np.sqrt(2)*np.array([1,1])
    input_rho = np.outer(input_state, input_state.conj().T)
    counts = []
    for i, angle in enumerate(angles):
        qwp = qwp_mat(angle+phase, retardance)
        output_state = qwp @ input_state

        output_state = np.outer(output_state, output_state.conj().T)
        zero_counts = np.real(np.trace(input_rho @ output_state))
        one_counts = 1 - zero_counts
        counts.append([zero_counts, one_counts])
    counts = np.array(counts)
    
    return angles, counts

def simulate_angle_sweep_hwp(retardance=1,phase=0):
    angles = np.linspace(0, (2 * np.pi), 360)
    input_state = np.array([1, 0])
    input_rho = np.outer(input_state, input_state.conj().T)
    counts = []
    for i, angle in enumerate(angles):
        qwp = qwp_mat(angle+phase, retardance)
        output_state = qwp @ input_state

        output_state = np.outer(output_state, output_state.conj().T)
        zero_counts = np.real(np.trace(input_rho @ output_state))
        one_counts = 1 - zero_counts
        counts.append([zero_counts, one_counts])
    counts = np.array(counts)
    
    return angles, counts

def plot_angle_sweep(retardance=1):
    folder_name = "./sim_angle_sweeps"
    Path.mkdir(Path(folder_name), exist_ok=True, parents=True)
    angles = np.linspace(0, 2 * np.pi, 360)
    input_state = 1/np.sqrt(2)*np.array([1, 1])
    input_rho = np.outer(input_state, input_state.conj().T)
    retardances = np.linspace(0.98, 1.1, 50)
    for j, retardance in enumerate(retardances):
        counts = []
        for i, angle in enumerate(angles):
            qwp = qwp_mat(angle, retardance)
            output_state = qwp @ input_state

            output_state = np.outer(output_state, output_state.conj().T)
            zero_counts = np.real(np.trace(input_rho @ output_state))
            one_counts = 1 - zero_counts
            counts.append([zero_counts, one_counts])
        # counts = np.array(counts)
    
        counts = np.array(counts)
            # return angles, counts

        fig, ax = plt.subplots()

        ax.plot(angles, counts[:, 0], label="H counts")
        ax.plot(angles, counts[:, 1], label="V counts")

        ax.set_ylabel("Normalized Counts")
        ax.set_xlabel("Angle of rotation (radians)")
        ax.legend()
        ax.set_title(f"QWP sweep (retardance {retardance:.4f})")
        plt.savefig(folder_name + f"/{retardance:.4f}.png")
        plt.close()

    image_paths = Path(folder_name).glob("*.png")
    image_paths_sorted = sorted(image_paths, key=lambda x: float(x.stem))
    images = []
    for filename in image_paths_sorted:
        try:
            images.append(imageio.imread(filename))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    imageio.mimsave(f"{folder_name}/animation.gif", images, loop=0, duration=5)



if __name__ == "__main__":
    # qwp = VariableQWPRotStage(45, None, retardance=np.pi / 2)
    # make_bloch_animation()
    # check_density_matrix()
    # simulate_angle_sweep()
    plot_angle_sweep()
