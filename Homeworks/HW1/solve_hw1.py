"""Solve ECE 6390 HW1 using the course astrodynamics toolbox.

Run in conda environment ``foo`` with ``conda run -n foo python solve_hw1.py``.
The script prints its progress and results and saves one density-versus-drift
PNG in this folder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from tqdm import tqdm


COURSE_ROOT = Path(__file__).resolve().parents[2]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from hw1_helpers import AXIS_OFFSET_BOUNDS_KM, DENSITY_MAX_KG_M2, DENSITY_MIN_KG_M2, ECCENTRICITY_BOUNDS, GEO_RADIUS_M, PHASE_BOUNDS_DEG, density_for_drift, design_orbit_naive, design_orbit_scipy, initial_design_state, look_angle_history, save_density_plot, save_look_angle_plot, simulate_year  # noqa: E402
from src.astrodynamics.propagation import DEFAULT_STEP_S  # noqa: E402


DESIGN_METHOD = "naive"


def main() -> None:
    print(f"GEO radius: {GEO_RADIUS_M/1000:.3f} km")
    print(f"Orbit step: {DEFAULT_STEP_S:g} s (set DEFAULT_STEP_S in the toolbox)")
    print("Model: equatorial two-body gravity + rotating solar pressure; no eclipse or other perturbations")
    print("Atlanta: 33.7758 deg N, 84.39738 deg W; pressure reflectivity: 0.5")
    print("Circular case: initial SRP acceleration is radially outward (phase = 0 deg)")
    baseline = simulate_year(math.inf, 0.0, DEFAULT_STEP_S)
    print(f"Zero-pressure numerical drift: {baseline.max_deviation_deg:.6f} deg")

    print(f"Phase 1: density search from {DENSITY_MIN_KG_M2:g} to {DENSITY_MAX_KG_M2:g} kg/m^2 for a circular GEO start")
    print("Progress counts year-long simulations at tested rho_A; D = max |Az(t) - Az(0)|")
    print("Each target uses 2 endpoints, 15 geometric midpoint trials, and 1 final check (18 simulations)")
    density_samples: dict[float, float] = {}
    thresholds: dict[float, float] = {}
    for target_deg in (1.0, 5.0, 15.0):
        with tqdm(total=18, desc=f"D={target_deg:g} deg density search", unit="year simulation") as progress:
            density, run = density_for_drift(target_deg, DEFAULT_STEP_S, density_samples, progress)
        thresholds[target_deg] = density
        print(f"  target D={target_deg:g} deg: rho_A={density:.5f} kg/m^2; computed D={run.max_deviation_deg:.5f} deg")
    plot_path = Path(__file__).with_name("hw1_density_vs_azimuth_drift.png")
    save_density_plot(density_samples, thresholds, plot_path)
    print(f"Saved density-versus-drift plot: {plot_path}")

    design_density = 0.5
    print("Phase 2: fix rho_A=0.5 kg/m^2 and minimize D = max |Az(t) - Az(0)|")
    print(f"Search bounds: GEO semi-major-axis offset {AXIS_OFFSET_BOUNDS_KM} km, eccentricity {ECCENTRICITY_BOUNDS}, initial SRP phase {PHASE_BOUNDS_DEG} deg")
    print("Every objective evaluation is a one-year simulation using the toolbox's ASD3 propagator")
    if DESIGN_METHOD == "naive":
        print("Design method: original coordinate search")
        design_function = design_orbit_naive
    elif DESIGN_METHOD == "scipy":
        print("Design method: SciPy differential evolution followed by Powell refinement")
        design_function = design_orbit_scipy
    else:
        raise ValueError('DESIGN_METHOD must be "naive" or "scipy"')
    with tqdm(desc=f"{DESIGN_METHOD} orbit design", unit="year simulation") as progress:
        design = design_function(design_density, DEFAULT_STEP_S, progress)
    result = design.pointing
    print("Designed 0.5 kg/m^2 orbit:")
    print(f"  a = {(GEO_RADIUS_M + design.axis_offset_km*1000.0)/1000:.3f} km (GEO offset {design.axis_offset_km:+.3f} km), e = {design.eccentricity:.6f}")
    print(f"  initial SRP acceleration phase = {design.phase_deg:.3f} deg from outward radius")
    print(f"  objective evaluations = {design.evaluations}")
    print(f"  optimized peak azimuth deviation D = {result.max_deviation_deg:.4f} deg")
    print(f"  azimuth = {result.min_az_deg:.4f} to {result.max_az_deg:.4f} deg; range = {result.azimuth_range_deg:.4f} deg")
    print(f"  elevation = {result.min_el_deg:.4f} to {result.max_el_deg:.4f} deg")
    print(f"  minimum orbital radius = {result.min_radius_m/1000:.3f} km")
    final_state = initial_design_state(design.axis_offset_km, design.eccentricity)
    history = look_angle_history(design_density, math.radians(design.phase_deg), DEFAULT_STEP_S, final_state)
    look_angle_plot_path = Path(__file__).with_name("hw1_optimized_look_angles.png")
    save_look_angle_plot(history, look_angle_plot_path)
    print(f"Saved required one-year look-angle plot: {look_angle_plot_path}")


if __name__ == "__main__":
    main()
