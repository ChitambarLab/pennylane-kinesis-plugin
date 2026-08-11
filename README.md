# Pennylane-Kinesis Plugin
A PennyLane device plugin for controlling Thorlabs Elliptec and Kinesis rotational stages for variational optimization.

This is part of the software stack that is used as part of our work "Variational Quantum Optimization of Noisy Polarization Qubit Experiments", currently found on the [arxiv]() TODO: ADD ARXIV LINK.

The main PennyLane/Thorlabs Elliptec integration can be found in `src/pennylane-kinesis`, and the scripts used to set up various protocols and experiments are in `script/vqo_protocols` which utilizes protocol descriptions in `src/pennylane_kinesis/script_utilities.py`. There are some scripts that we used for calibrating our hardware and gathering setup-specific data in `script/hardware_calibrations`; these may or may not be useful to the average viewer. 

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

## Running Tests
You can run the testing for this library by running `uv run pytest`.

## Code Formatting
You can format the code in the codebase by running `uv format`.