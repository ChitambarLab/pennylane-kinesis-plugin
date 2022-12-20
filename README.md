# Pennylane-Kinesis Plugin
A PennyLane device plugin for controlling rotational stages using Thorlabs Kinesis Software.

## Development Environment

This project uses the [Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary) distribution of Python to ensure a consistent development environment.
Follow the [Anaconda installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) to set up the `conda` command line tool.
For more details, see the [managing environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) page in the `conda` documentation.

To create the dev environment, navigate to the root directory of the ``pennylane-kinesis`` repository and follow these steps:

1. Create the `pennylane-kinesis-dev` conda environment.

```
(base) $ conda env create -f environment.yml
```

2. Activate the `pennylane-kinesis-dev` conda environment.

```
(base) $ conda activate pennylane-kinesis-dev
```

3. Install pennylane-kinesis plugin in editable mode.

```
(pennylane-kinesis-dev) $ pip install -e .
```

## Running Tests

```
(pennylane-kinesis-dev) $ python -m pytest
```

## Code Formatting

```
(pennylane-kinesis-dev) $ black -l 100 test src
```
