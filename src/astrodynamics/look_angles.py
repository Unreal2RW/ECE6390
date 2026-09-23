"""ASD5 Earth-station visibility, pointing, and slant-range geometry.

Classes
-------
LookAngles: Central angle, elevation, azimuth, and slant range for one view.
    above_elevation_mask: Whether elevation meets a chosen mask angle.

Functions
---------
central_angle: Earth-centered angle between station and satellite subpoint.
equatorial_subpoint_longitude: Earth-fixed longitude for an ASD3 equatorial
    trajectory whose inertial angle starts at zero.
equatorial_orbit_look_angles: Calculate look angles directly from an
    equatorial OrbitState.
look_angles: Calculate pointing, visibility geometry, and slant range.

Angles are in radians, distances in meters, and longitude is east-positive.
Azimuth is clockwise from north in [0, 2*pi); it is NaN directly at zenith,
where a compass bearing is undefined.
"""

import math
from dataclasses import dataclass

from .constants import EARTH_RADIUS_M, EARTH_ROTATION_RATE_RAD_S
from .propagation import OrbitState


@dataclass(frozen=True)
class LookAngles:
    """Geometry of a satellite as seen from one Earth station."""

    central_angle_rad: float
    elevation_rad: float
    azimuth_rad: float
    slant_range_m: float

    def above_elevation_mask(self, minimum_elevation_rad: float = 0.0) -> bool:
        """Return whether the satellite meets a geometric or operational mask."""
        return self.elevation_rad >= minimum_elevation_rad


def central_angle(station_lat_rad: float, station_lon_rad: float,
                  subpoint_lat_rad: float, subpoint_lon_rad: float) -> float:
    """Return the Earth-centered station-to-subpoint separation in radians."""
    delta_lon = subpoint_lon_rad - station_lon_rad
    cosine = (math.sin(station_lat_rad) * math.sin(subpoint_lat_rad)
              + math.cos(station_lat_rad) * math.cos(subpoint_lat_rad) * math.cos(delta_lon))
    return math.acos(max(-1.0, min(1.0, cosine)))


def equatorial_subpoint_longitude(inertial_angle_rad: float, time_s: float,
                                  initial_longitude_rad: float,
                                  earth_rotation_rate_rad_s: float = EARTH_ROTATION_RATE_RAD_S) -> float:
    """Return Earth-fixed longitude, wrapped into [-pi, pi].

    The propagation starts at inertial angle zero; initial_longitude_rad is its
    chosen subpoint longitude at time zero.
    """
    longitude = initial_longitude_rad + inertial_angle_rad - earth_rotation_rate_rad_s * time_s
    return math.atan2(math.sin(longitude), math.cos(longitude))


def look_angles(station_lat_rad: float, station_lon_rad: float,
                subpoint_lat_rad: float, subpoint_lon_rad: float,
                satellite_radius_m: float, earth_radius_m: float = EARTH_RADIUS_M) -> LookAngles:
    """Return central angle, elevation, azimuth, and slant range.

    Positive elevation means the satellite is above a spherical Earth horizon.
    No terrain, refraction, or operational elevation mask is assumed.
    """
    if earth_radius_m <= 0.0 or satellite_radius_m <= earth_radius_m:
        raise ValueError("Require satellite radius above a positive Earth radius")

    gamma = central_angle(station_lat_rad, station_lon_rad, subpoint_lat_rad, subpoint_lon_rad)
    slant_range = math.sqrt(earth_radius_m**2 + satellite_radius_m**2
                            - 2.0 * earth_radius_m * satellite_radius_m * math.cos(gamma))
    elevation = math.atan2(satellite_radius_m * math.cos(gamma) - earth_radius_m,
                           satellite_radius_m * math.sin(gamma))

    delta_lon = subpoint_lon_rad - station_lon_rad
    east = math.sin(delta_lon) * math.cos(subpoint_lat_rad)
    north = (math.cos(station_lat_rad) * math.sin(subpoint_lat_rad)
             - math.sin(station_lat_rad) * math.cos(subpoint_lat_rad) * math.cos(delta_lon))
    azimuth = math.nan if east == 0.0 and north == 0.0 else math.atan2(east, north) % (2.0 * math.pi)
    return LookAngles(gamma, elevation, azimuth, slant_range)


def equatorial_orbit_look_angles(state: OrbitState, station_lat_rad: float, station_lon_rad: float, initial_subpoint_lon_rad: float, earth_radius_m: float = EARTH_RADIUS_M) -> LookAngles:
    """Return ground-station look angles for an equatorial propagated state."""
    subpoint_lon_rad = equatorial_subpoint_longitude(state.angle_rad, state.time_s, initial_subpoint_lon_rad)
    return look_angles(station_lat_rad, station_lon_rad, 0.0, subpoint_lon_rad, state.radius_m, earth_radius_m)
