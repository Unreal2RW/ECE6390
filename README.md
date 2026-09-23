# ECE 6390 Course Code

This repository contains reusable course tools and homework implementations for ECE 6390.

## Repository structure

- `src/`: reusable toolbox code that can support multiple assignments and course topics.
- `Homeworks/`: scripts and assignment-specific helpers organized by homework.

## Current implementation

- `src/astrodynamics/`: constants, two-body orbit calculations, numerical orbit propagation, perturbation models, and look-angle geometry.
- `Homeworks/HW1/`: the Homework 1 solution and its supporting functions for solar-pressure perturbations and look-angle analysis.

## Requirements

- Python 3.10 or later
- SciPy
- Matplotlib
- tqdm

## Run the current homework

From the repository root:

```bash
python Homeworks/HW1/solve_hw1.py
```

The script prints progress and numerical results in the terminal. Generated figures and other result artifacts are not stored in this repository.
