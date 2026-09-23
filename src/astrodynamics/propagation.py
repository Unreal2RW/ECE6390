"""ASD3 planar finite-difference orbit propagation.

Classes
-------
OrbitState: One planar orbit sample with time, polar position, and velocities.
    tangential_speed_m_s: Tangential speed derived from radius and angular rate.
    specific_energy: Two-body specific mechanical energy for drift checks.
    specific_angular_momentum: Two-body angular momentum for drift checks.

Functions
---------
circular_orbit_state: Construct a circular planar-orbit state.
periapsis_orbit_state: Construct a state at an ellipse's periapsis.
propagate_planar: Yield sampled states from the lecture's four difference
    equations, optionally adding radial and tangential acceleration.
DEFAULT_STEP_S: Course-project default integration step in seconds.

SI units and radians are used throughout. The optional acceleration callback
receives (time_s, radius_m, angle_rad) and returns (radial_m_s2,
tangential_m_s2). For a long run, sample_every limits stored output without
changing the integration step.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .constants import EARTH_MU_M3_S2


DEFAULT_STEP_S = 2.0


@dataclass(frozen=True)
class OrbitState:
    """One state of a planar orbit in polar coordinates."""

    time_s: float
    radius_m: float
    angle_rad: float
    radial_speed_m_s: float
    angular_rate_rad_s: float

    def tangential_speed_m_s(self) -> float:
        """Return radius times angular rate."""
        return self.radius_m * self.angular_rate_rad_s

    def specific_energy(self, mu_m3_s2: float = EARTH_MU_M3_S2) -> float:
        """Return two-body specific energy in J/kg."""
        speed_squared = self.radial_speed_m_s**2 + self.tangential_speed_m_s()**2
        return speed_squared / 2.0 - mu_m3_s2 / self.radius_m

    def specific_angular_momentum(self) -> float:
        """Return two-body specific angular momentum in m^2/s."""
        return self.radius_m**2 * self.angular_rate_rad_s


def circular_orbit_state(radius_m: float, angle_rad: float = 0.0, mu_m3_s2: float = EARTH_MU_M3_S2) -> OrbitState:
    """Return a zero-radial-speed state for a circular orbit."""
    if radius_m <= 0.0 or mu_m3_s2 <= 0.0:
        raise ValueError("Require positive radius and mu")
    angular_rate_rad_s = (mu_m3_s2 / radius_m**3) ** 0.5
    return OrbitState(0.0, radius_m, angle_rad, 0.0, angular_rate_rad_s)


def periapsis_orbit_state(semi_major_axis_m: float, eccentricity: float, angle_rad: float = 0.0, mu_m3_s2: float = EARTH_MU_M3_S2) -> OrbitState:
    """Return a zero-radial-speed state at an ellipse's periapsis."""
    if semi_major_axis_m <= 0.0 or mu_m3_s2 <= 0.0 or not 0.0 <= eccentricity < 1.0:
        raise ValueError("Require positive semi-major axis and mu with 0 <= eccentricity < 1")
    radius_m = semi_major_axis_m * (1.0 - eccentricity)
    tangential_speed_m_s = (mu_m3_s2 * (2.0 / radius_m - 1.0 / semi_major_axis_m)) ** 0.5
    return OrbitState(0.0, radius_m, angle_rad, 0.0, tangential_speed_m_s / radius_m)


def propagate_planar(radius_m: float, radial_speed_m_s: float, tangential_speed_m_s: float, duration_s: float, step_s: float = DEFAULT_STEP_S, *, mu_m3_s2: float = EARTH_MU_M3_S2, acceleration: Callable[[float, float, float], tuple[float, float]] | None = None, sample_every: int = 1) -> Iterator[OrbitState]:
    """Yield the initial state and selected subsequent states through duration_s.

    The lecture model starts at inertial angle zero. Adjust DEFAULT_STEP_S and
    halve it manually to check convergence for a different orbit or
    perturbation. With a nonzero perturbing acceleration, two-body energy and
    angular momentum need not remain constant.
    """
    if radius_m <= 0.0 or duration_s < 0.0 or step_s <= 0.0 or mu_m3_s2 <= 0.0:
        raise ValueError("Require positive radius, step, and mu; duration must be nonnegative")
    if not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")

    time_s, angle_rad = 0.0, 0.0
    angular_rate_rad_s = tangential_speed_m_s / radius_m
    yield OrbitState(time_s, radius_m, angle_rad, radial_speed_m_s, angular_rate_rad_s)

    step_number = 0
    while time_s < duration_s:
        dt = min(step_s, duration_s - time_s)
        dr = radial_speed_m_s * dt
        dtheta = angular_rate_rad_s * dt
        midpoint_radius = radius_m + dr / 2.0
        radial_accel, tangential_accel = (acceleration(time_s, radius_m, angle_rad)
                                          if acceleration is not None else (0.0, 0.0))
        next_dr = dr + midpoint_radius * dtheta**2
        next_dr += (radial_accel - mu_m3_s2 / radius_m**2) * dt**2
        next_dtheta = dtheta - 2.0 * dr * dtheta / midpoint_radius
        next_dtheta += tangential_accel * dt**2 / midpoint_radius

        radius_m += dr
        angle_rad += dtheta
        time_s += dt
        radial_speed_m_s = next_dr / dt
        angular_rate_rad_s = next_dtheta / dt
        step_number += 1
        if step_number % sample_every == 0 or time_s >= duration_s:
            yield OrbitState(time_s, radius_m, angle_rad, radial_speed_m_s, angular_rate_rad_s)
