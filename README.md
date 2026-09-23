# ECE 6390 Astrodynamics Toolbox and HW1

This repository contains a small Python astrodynamics toolbox based on the ECE 6390 lecture methods and the scripts used for Homework 1 on solar-radiation-pressure perturbations and ground-station look angles.

## Contents

- `src/astrodynamics/`: course-based constants, two-body orbit formulas, ASD3 numerical propagation, solar-pressure perturbations, and ASD5 look-angle geometry.
- `Homeworks/HW1/solve_hw1.py`: terminal entry point for the two homework calculations.
- `Homeworks/HW1/hw1_helpers.py`: HW1-specific simulation, search, optimization, and plotting helpers.

## Requirements

- Python 3.10 or later
- SciPy
- Matplotlib
- tqdm

## Run

From the repository root:

```bash
python Homeworks/HW1/solve_hw1.py
```

The integration step is configured through `DEFAULT_STEP_S` in `src/astrodynamics/propagation.py`. Phase 2 can use either the original coordinate search or the SciPy search by setting `DESIGN_METHOD` in `Homeworks/HW1/solve_hw1.py` to `"naive"` or `"scipy"`.

The script prints progress and numerical results in the terminal. It also generates the density-versus-drift and optimized look-angle figures locally; generated artifacts are not stored in this repository.
