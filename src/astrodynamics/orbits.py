"""Analytic two-body orbit calculations from ASD1 and ASD2.

Functions
---------
circular_angular_rate: Angular speed of a circular orbit.
circular_speed: Tangential speed of a circular orbit.
orbital_period: Period from circular radius or elliptical semi-major axis.
radius_for_period: Circular radius for a requested orbital period.
ellipse_from_apsides: Semi-major axis and eccentricity from apsis radii.
vis_viva_speed: Speed at a radius on a Keplerian ellipse.
hohmann_transfer: Two burn magnitudes and flight time between circular orbits.

All distances are in meters, times in seconds, speeds in m/s, and angles in
radians. Radii are measured from the center of the attracting body.
"""

import math

from .constants import EARTH_MU_M3_S2


def circular_angular_rate(radius_m: float, mu_m3_s2: float = EARTH_MU_M3_S2) -> float:
    """Return sqrt(mu/r^3) in rad/s."""
    return math.sqrt(mu_m3_s2 / radius_m**3)


def circular_speed(radius_m: float, mu_m3_s2: float = EARTH_MU_M3_S2) -> float:
    """Return sqrt(mu/r) in m/s."""
    return math.sqrt(mu_m3_s2 / radius_m)


def orbital_period(semi_major_axis_m: float, mu_m3_s2: float = EARTH_MU_M3_S2) -> float:
    """Return the Keplerian period in seconds."""
    return 2.0 * math.pi * math.sqrt(semi_major_axis_m**3 / mu_m3_s2)


def radius_for_period(period_s: float, mu_m3_s2: float = EARTH_MU_M3_S2) -> float:
    """Return the radius of a circular orbit with the specified period."""
    return (mu_m3_s2 * (period_s / (2.0 * math.pi))**2) ** (1.0 / 3.0)


def ellipse_from_apsides(periapsis_m: float, apoapsis_m: float) -> tuple[float, float]:
    """Return (semi-major axis in meters, eccentricity)."""
    if periapsis_m <= 0.0 or apoapsis_m < periapsis_m:
        raise ValueError("Require 0 < periapsis_m <= apoapsis_m")
    return (periapsis_m + apoapsis_m) / 2.0, (apoapsis_m - periapsis_m) / (apoapsis_m + periapsis_m)


def vis_viva_speed(radius_m: float, semi_major_axis_m: float,
                   mu_m3_s2: float = EARTH_MU_M3_S2) -> float:
    """Return sqrt(mu(2/r - 1/a)) in m/s."""
    return math.sqrt(mu_m3_s2 * (2.0 / radius_m - 1.0 / semi_major_axis_m))


def hohmann_transfer(start_radius_m: float, target_radius_m: float,
                     mu_m3_s2: float = EARTH_MU_M3_S2) -> tuple[float, float, float]:
    """Return (first burn, second burn, transfer time), all positive magnitudes.

    The burns are in m/s and transfer time is in seconds. The model assumes
    coplanar circular endpoints and two instantaneous tangential burns.
    """
    if start_radius_m <= 0.0 or target_radius_m <= 0.0:
        raise ValueError("Orbit radii must be positive")
    transfer_axis = (start_radius_m + target_radius_m) / 2.0
    first_burn = abs(vis_viva_speed(start_radius_m, transfer_axis, mu_m3_s2)
                     - circular_speed(start_radius_m, mu_m3_s2))
    second_burn = abs(circular_speed(target_radius_m, mu_m3_s2)
                      - vis_viva_speed(target_radius_m, transfer_axis, mu_m3_s2))
    return first_burn, second_burn, orbital_period(transfer_axis, mu_m3_s2) / 2.0
