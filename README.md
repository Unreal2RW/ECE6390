# ECE 6390 Course Code

This repository contains reusable course tools and homework implementations for ECE 6390.

## Repository structure

- `src/`: reusable toolbox code that can support multiple assignments and course topics.
- `Homeworks/`: scripts and assignment-specific helpers organized by homework.

## Current implementation

- `src/astrodynamics/`: constants, two-body orbit calculations, numerical orbit propagation, perturbation models, and look-angle geometry.
- `src/antenna/`: position-vector array factors, element patterns, and directivity and gain calculations.
- `src/rf/`: wavelength and power conversions, free-space loss, and Friis link budgets.
- `Homeworks/HW1/`: the Homework 1 solution and its supporting functions for solar-pressure perturbations and look-angle analysis.

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- Matplotlib
- tqdm

The reusable modules use SI units and radians unless a function's documentation says otherwise. Array directivity is normalized by a numerical integral of the specified element and array power pattern over the sphere; angular sampling must resolve the pattern. Link budgets take total transmit-array power and directional linear gains.

To make the toolbox importable from homework directories, install it from the repository root:

```bash
python -m pip install -e .
```

For example, reusable modules can then be imported as `src.astrodynamics.geometry`, `src.antenna.arrays`, `src.antenna.patterns`, and `src.rf.links`.

## Run the current homework

From the repository root:

```bash
python Homeworks/HW1/solve_hw1.py
```

The script prints progress and numerical results in the terminal. Generated figures and other result artifacts are not stored in this repository.
