"""Solar radiation pressure model used by HW1 and ASD4.

Functions
---------
solar_pressure_magnitude: Acceleration from HW1 pressure, reflectivity, and
    spacecraft area density.
rotating_solar_pressure: Make a planar acceleration callback for
    propagation.propagate_planar. Its returned callback resolves the rotating
    inertial force into radial and tangential components at each step.

This assignment model assumes a constant pressure at 1 AU, an equatorial force,
and no eclipse. The epoch angle describes the acceleration direction (away from
the Sun), not the direction from the spacecraft toward the Sun.
"""

import math
from collections.abc import Callable

from .constants import SOLAR_PRESSURE_N_M2, SOLAR_YEAR_S


def solar_pressure_magnitude(area_density_kg_m2: float, reflectivity: float = 0.5) -> float:
    """Return (9.08 microN/m^2) * reflectivity / area density, in m/s^2."""
    if area_density_kg_m2 <= 0.0 or reflectivity < 0.0:
        raise ValueError("Require positive area density and nonnegative reflectivity")
    return SOLAR_PRESSURE_N_M2 * reflectivity / area_density_kg_m2


def rotating_solar_pressure(area_density_kg_m2: float, reflectivity: float = 0.5,
                            direction_at_epoch_rad: float = 0.0,
                            angular_rate_rad_s: float = 2.0 * math.pi / SOLAR_YEAR_S) -> Callable:
    """Return a callback giving (radial, tangential) acceleration in m/s^2.

    A positive angular rate rotates the force counterclockwise in the inertial
    equatorial plane. Supply a negative rate to reverse that convention.
    """
    magnitude = solar_pressure_magnitude(area_density_kg_m2, reflectivity)

    def acceleration(time_s: float, radius_m: float, angle_rad: float) -> tuple[float, float]:
        direction = direction_at_epoch_rad + angular_rate_rad_s * time_s
        relative_angle = direction - angle_rad
        return magnitude * math.cos(relative_angle), magnitude * math.sin(relative_angle)

    return acceleration
