"""HW1-specific simulation, density search, plotting, and orbit design."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.optimize import differential_evolution, minimize

from src.astrodynamics.constants import EARTH_RADIUS_M, SIDEREAL_DAY_S, SOLAR_YEAR_S
from src.astrodynamics.look_angles import equatorial_orbit_look_angles
from src.astrodynamics.orbits import radius_for_period
from src.astrodynamics.perturbations import rotating_solar_pressure
from src.astrodynamics.propagation import OrbitState, circular_orbit_state, periapsis_orbit_state, propagate_planar


ATLANTA_LAT_RAD = math.radians(33.7758)
ATLANTA_LON_RAD = math.radians(-84.39738)
REFLECTIVITY = 0.5
GEO_RADIUS_M = radius_for_period(SIDEREAL_DAY_S)
OBSERVATION_INTERVAL_S = 600.0
DENSITY_MIN_KG_M2 = 0.2
DENSITY_MAX_KG_M2 = 100.0
AXIS_OFFSET_BOUNDS_KM = (-30.0, 30.0)
ECCENTRICITY_BOUNDS = (0.0, 0.08)
PHASE_BOUNDS_DEG = (0.0, 360.0)


@dataclass(frozen=True)
class PointingResult:
    """One-year pointing envelope for one simulated orbit."""

    initial_az_deg: float
    max_deviation_deg: float
    azimuth_range_deg: float
    min_az_deg: float
    max_az_deg: float
    min_el_deg: float
    max_el_deg: float
    min_radius_m: float


@dataclass(frozen=True)
class DesignResult:
    """Optimized orbit parameters and their one-year pointing result."""

    axis_offset_km: float
    eccentricity: float
    phase_deg: float
    pointing: PointingResult
    evaluations: int


@dataclass(frozen=True)
class LookAngleHistory:
    """Sampled look angles for plotting one propagated orbit."""

    time_days: tuple[float, ...]
    azimuth_deg: tuple[float, ...]
    elevation_deg: tuple[float, ...]


def initial_geo_state() -> OrbitState:
    """Return an ideal circular-GEO state from the toolbox."""
    return circular_orbit_state(GEO_RADIUS_M)


def initial_design_state(axis_offset_km: float, eccentricity: float) -> OrbitState:
    """Return a perigee-start state for one HW1 design candidate."""
    state = periapsis_orbit_state(GEO_RADIUS_M + axis_offset_km * 1000.0, eccentricity)
    if state.radius_m <= EARTH_RADIUS_M:
        raise ValueError("Perigee must remain above Earth's surface")
    return state


def pointing(state: OrbitState):
    """Return Atlanta look angles for an equatorial orbit state."""
    return equatorial_orbit_look_angles(state, ATLANTA_LAT_RAD, ATLANTA_LON_RAD, ATLANTA_LON_RAD)


def simulate_year(area_density_kg_m2: float, solar_phase_rad: float, step_s: float, initial_state: OrbitState | None = None) -> PointingResult:
    """Propagate one solar year and measure the Atlanta pointing envelope."""
    if area_density_kg_m2 <= 0.0 or step_s <= 0.0:
        raise ValueError("Area density and integration step must be positive")
    initial = initial_geo_state() if initial_state is None else initial_state
    solar = rotating_solar_pressure(area_density_kg_m2, REFLECTIVITY, direction_at_epoch_rad=solar_phase_rad)
    initial_angles = pointing(initial)
    az0, el0 = initial_angles.azimuth_rad, initial_angles.elevation_rad
    previous_az, unwrapped_az = az0, az0
    minimum_az = maximum_az = az0
    minimum_el = maximum_el = el0
    minimum_radius = initial.radius_m
    observation_stride = max(1, int(OBSERVATION_INTERVAL_S / step_s))
    states = propagate_planar(initial.radius_m, initial.radial_speed_m_s, initial.tangential_speed_m_s(), SOLAR_YEAR_S, step_s, acceleration=solar)
    for step_number, state in enumerate(states):
        if step_number == 0:
            continue
        minimum_radius = min(minimum_radius, state.radius_m)
        if state.radius_m <= EARTH_RADIUS_M:
            raise ValueError("Trajectory intersects Earth")
        if step_number % observation_stride != 0 and state.time_s < SOLAR_YEAR_S:
            continue
        angles = pointing(state)
        az, el = angles.azimuth_rad, angles.elevation_rad
        unwrapped_az += math.atan2(math.sin(az - previous_az), math.cos(az - previous_az))
        previous_az = az
        minimum_az = min(minimum_az, unwrapped_az)
        maximum_az = max(maximum_az, unwrapped_az)
        minimum_el = min(minimum_el, el)
        maximum_el = max(maximum_el, el)
    initial_az_deg = math.degrees(az0)
    max_deviation_deg = math.degrees(max(abs(minimum_az - az0), abs(maximum_az - az0)))
    azimuth_range_deg = math.degrees(maximum_az - minimum_az)
    return PointingResult(initial_az_deg, max_deviation_deg, azimuth_range_deg, math.degrees(minimum_az), math.degrees(maximum_az), math.degrees(minimum_el), math.degrees(maximum_el), minimum_radius)


def density_for_drift(target_deg: float, step_s: float, samples: dict[float, float], progress: Any = None) -> tuple[float, PointingResult]:
    """Find the circular-GEO density giving a target peak azimuth deviation."""
    def measured(density: float) -> PointingResult:
        try:
            run = simulate_year(density, 0.0, step_s)
        except ValueError as exc:
            if str(exc) != "Trajectory intersects Earth":
                raise
            if progress is not None:
                progress.set_postfix_str(f"rho={density:.4g} kg/m^2: Earth crossing; search stopped")
            raise RuntimeError(f"Cannot bracket D={target_deg:g} deg: rho_A={density:.4g} kg/m^2 intersects Earth at step {step_s:g} s; check step-size convergence") from exc
        samples[density] = run.max_deviation_deg
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"rho={density:.4g} kg/m^2, D={run.max_deviation_deg:.4g} deg")
        return run

    low, high = DENSITY_MIN_KG_M2, DENSITY_MAX_KG_M2
    if measured(low).max_deviation_deg < target_deg:
        raise RuntimeError("Lower density bound did not reach the requested drift")
    if measured(high).max_deviation_deg > target_deg:
        raise RuntimeError("Upper density bound still exceeds the requested drift")
    for _ in range(15):
        middle = math.sqrt(low * high)
        if measured(middle).max_deviation_deg > target_deg:
            low = middle
        else:
            high = middle
    threshold = math.sqrt(low * high)
    return threshold, measured(threshold)


def save_density_plot(samples: dict[float, float], thresholds: dict[float, float], output_path: Path) -> None:
    """Save evaluated circular-GEO densities versus peak azimuth deviation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    densities = sorted(samples)
    deviations = [samples[density] for density in densities]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.scatter(densities, deviations, s=24, label="One-year simulations")
    for target_deg, density in thresholds.items():
        axis.axhline(target_deg, color="0.75", linewidth=0.8)
        axis.scatter([density], [samples[density]], s=60, label=f"{target_deg:g} deg: {density:.4g} kg/m^2")
    axis.set_xscale("log")
    axis.set_xlabel("Area density rho_A (kg/m^2)")
    axis.set_ylabel("Peak azimuth deviation D (deg)")
    axis.set_title("Circular GEO: D = max |azimuth(t) - azimuth(0)| over one year")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def look_angle_history(area_density_kg_m2: float, solar_phase_rad: float, step_s: float, initial_state: OrbitState) -> LookAngleHistory:
    """Propagate one year and retain sampled Atlanta azimuth and elevation."""
    solar = rotating_solar_pressure(area_density_kg_m2, REFLECTIVITY, direction_at_epoch_rad=solar_phase_rad)
    sample_every = max(1, int(OBSERVATION_INTERVAL_S / step_s))
    states = propagate_planar(initial_state.radius_m, initial_state.radial_speed_m_s, initial_state.tangential_speed_m_s(), SOLAR_YEAR_S, step_s, acceleration=solar, sample_every=sample_every)
    time_days: list[float] = []
    azimuth_deg: list[float] = []
    elevation_deg: list[float] = []
    previous_az = None
    unwrapped_az = None
    for state in states:
        if state.radius_m <= EARTH_RADIUS_M:
            raise ValueError("Trajectory intersects Earth")
        angles = pointing(state)
        if previous_az is None:
            unwrapped_az = angles.azimuth_rad
        else:
            unwrapped_az += math.atan2(math.sin(angles.azimuth_rad - previous_az), math.cos(angles.azimuth_rad - previous_az))
        previous_az = angles.azimuth_rad
        time_days.append(state.time_s / 86400.0)
        azimuth_deg.append(math.degrees(unwrapped_az))
        elevation_deg.append(math.degrees(angles.elevation_rad))
    return LookAngleHistory(tuple(time_days), tuple(azimuth_deg), tuple(elevation_deg))


def save_look_angle_plot(history: LookAngleHistory, output_path: Path) -> None:
    """Save one year of optimized-orbit azimuth and elevation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].plot(history.time_days, history.azimuth_deg, linewidth=0.8)
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].set_title("Optimized 0.5 kg/m^2 orbit: one year of Atlanta look angles")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(history.time_days, history.elevation_deg, linewidth=0.8)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Elevation (deg)")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def design_orbit_naive(area_density_kg_m2: float, step_s: float, progress: Any = None) -> DesignResult:
    """Use the original bounded coordinate search to minimize peak azimuth deviation."""
    seed_e = 0.011 / area_density_kg_m2
    best_parameters = (0.0, seed_e, 180.0)
    cache: dict[tuple[float, float, float], tuple[float, PointingResult | None]] = {}

    def objective(parameters: tuple[float, float, float]) -> float:
        axis_offset_km, eccentricity, phase_deg = parameters
        if not (AXIS_OFFSET_BOUNDS_KM[0] <= axis_offset_km <= AXIS_OFFSET_BOUNDS_KM[1] and ECCENTRICITY_BOUNDS[0] <= eccentricity <= ECCENTRICITY_BOUNDS[1]):
            return math.inf
        phase_deg %= 360.0
        key = (round(axis_offset_km, 7), round(eccentricity, 9), round(phase_deg, 7))
        if key in cache:
            return cache[key][0]
        try:
            initial = initial_design_state(axis_offset_km, eccentricity)
            run = simulate_year(area_density_kg_m2, math.radians(phase_deg), step_s, initial)
            value = run.max_deviation_deg if run.min_el_deg > 0.0 else math.inf
        except ValueError:
            run, value = None, math.inf
        cache[key] = (value, run)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"best D={min(item[0] for item in cache.values()):.3f} deg")
        return value

    best_value = objective(best_parameters)
    for offset_km in range(-10, 11, 2):
        trial = (float(offset_km), seed_e, 180.0)
        value = objective(trial)
        if value < best_value:
            best_parameters, best_value = trial, value
    increments = [1.0, 0.004, 12.0]
    for _ in range(16):
        improved = False
        for dimension in range(3):
            for direction in (-1.0, 1.0):
                trial = list(best_parameters)
                trial[dimension] += direction * increments[dimension]
                candidate = (float(trial[0]), float(trial[1]), float(trial[2]) % 360.0)
                value = objective(candidate)
                if value < best_value:
                    best_parameters, best_value = candidate, value
                    improved = True
        if not improved:
            increments = [increment / 2.0 for increment in increments]
        if increments[0] < 0.06 and increments[1] < 0.00025 and increments[2] < 0.75:
            break
    axis_offset_km, eccentricity, phase_deg = best_parameters
    initial = initial_design_state(axis_offset_km, eccentricity)
    final = simulate_year(area_density_kg_m2, math.radians(phase_deg), step_s, initial)
    return DesignResult(axis_offset_km, eccentricity, phase_deg, final, len(cache))


def design_orbit_scipy(area_density_kg_m2: float, step_s: float, progress: Any = None) -> DesignResult:
    """Use differential evolution and Powell refinement to minimize peak azimuth deviation."""
    cache: dict[tuple[float, float, float], tuple[float, PointingResult | None]] = {}

    def objective(parameters) -> float:
        axis_offset_km = float(parameters[0])
        eccentricity = float(parameters[1])
        phase_deg = float(parameters[2]) % 360.0
        key = (round(axis_offset_km, 7), round(eccentricity, 9), round(phase_deg, 7))
        if key in cache:
            return cache[key][0]
        try:
            initial = initial_design_state(axis_offset_km, eccentricity)
            run = simulate_year(area_density_kg_m2, math.radians(phase_deg), step_s, initial)
            value = run.max_deviation_deg if run.min_el_deg > 0.0 else 1.0e6
        except ValueError:
            run, value = None, 1.0e6
        cache[key] = (value, run)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"best D={min(item[0] for item in cache.values()):.3f} deg")
        return value

    bounds = (AXIS_OFFSET_BOUNDS_KM, ECCENTRICITY_BOUNDS, PHASE_BOUNDS_DEG)
    seed = (0.0, 0.011 / area_density_kg_m2, 180.0)
    global_result = differential_evolution(objective, bounds, seed=6390, maxiter=4, popsize=4, tol=0.02, polish=False, workers=1, updating="immediate")
    local_start = seed if objective(seed) < global_result.fun else global_result.x
    local_result = minimize(objective, local_start, method="Powell", bounds=bounds, options={"maxiter": 10, "maxfev": 60, "xtol": 0.05, "ftol": 1.0e-3})
    best_parameters = local_result.x if local_result.fun <= objective(global_result.x) else global_result.x
    axis_offset_km = float(best_parameters[0])
    eccentricity = float(best_parameters[1])
    phase_deg = float(best_parameters[2]) % 360.0
    initial = initial_design_state(axis_offset_km, eccentricity)
    final = simulate_year(area_density_kg_m2, math.radians(phase_deg), step_s, initial)
    return DesignResult(axis_offset_km, eccentricity, phase_deg, final, len(cache))
