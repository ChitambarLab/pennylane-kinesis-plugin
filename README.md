# Pennylane-Kinesis Plugin
A PennyLane device plugin for controlling Thorlabs Elliptec and Kinesis rotational stages for variational optimization. It also provides a simulator device that simulates waveplates and photon detectors.

This is part of the software stack that is used as part of our work "Variational Quantum Optimization of Noisy Polarization Qubit Experiments", currently found on the [arxiv]() TODO: ADD ARXIV LINK. See the pre-print for more information about the protocols being executed here.

The main PennyLane/Thorlabs Elliptec integration can be found in `src/pennylane-kinesis`, and the scripts used to set up various protocols and experiments are in `script/vqo_protocols` which utilizes protocol descriptions in `src/pennylane_kinesis/script_utilities.py`. The rest of the scripts are ones we used for calibrating our hardware and gathering setup-specific data; these may or may not be useful to the average viewer but is included for posterity.

## Requirements

If using a Swabian timetagger, please install the software package provided [here](https://www.swabianinstruments.com/time-tagger/downloads/) and follow the corresponding instructions. This installs the Python libraries required for this project.


## Development Environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency and project management. We recommend using following the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set up the command line tool. 

You can create the environment, install the dependencies, and install the project by running `uv sync` in the directory.

You can then use `uv run /path/to/script`
or
```
$ source .venv/bin/activate
(pennylane-kinesis) $ python /path/to/script
```
to run various scripts using the library.

## Example run

As a basic example, you can run the random access coding protocol on the simulator device by

```
(pennylane-kinesis) $ python ./script/vqo_protocols/random_access_coding.py -n 
````

This will perform 40 optimizations with 60 steps each starting from random parameters for each waveplate. You can control the number of trials with the `--num_trials` argument.

You can then plot the average results by

```
(pennylane-kinesis) $ python ./script/plotting/plot_rac.py -f /path/to/data/folder
````
As an example, 

```
(pennylane-kinesis) $ python ./script/plotting/plot_rac.py -f data/random_access_coding/rac_2/numerics/parameter-shift/adagrad/2026-10-19T17-03-23Z
```` 

## Running Tests
You can run the testing for this library by running `uv run pytest`.

## Code Formatting
You can format the code in the codebase by running `uv run ruff format`.