"""Vectorized spherical-Earth geometry for satellite observations.

Public functions
----------------
earth_fixed_position: Convert geocentric latitude, longitude, and radius to xyz.
geo_position: Earth-fixed position of a geostationary satellite.
satellite_to_ground: Directions from satellite to ground points and slant ranges.
ground_visibility: Test ground-point visibility against an elevation mask.
nadir_array_frame: Define a right-handed, nadir-pointing GEO array frame.
project_to_array_frame: Express Earth-fixed vectors in that array frame.

Positions and ranges are in meters; angles are in radians. Latitudes are
geocentric, longitude is east-positive, and vector coordinates occupy the last
axis. Latitude and longitude arrays broadcast, so georeferenced grids retain
their original shape. Convert degree-valued inputs explicitly with np.deg2rad.

The Cartesian construction is derived from the course look-angle geometry; the
existing scalar look_angles API remains available for antenna pointing.
"""

import numpy as np


def earth_fixed_position(latitude_rad, longitude_rad, radius_m):
    """Return xyz in meters, with shape broadcast(latitude, longitude, radius) + (3,).

    The Earth-fixed x axis is at zero longitude on the equator, y is at 90 deg
    east, and z points north. A spherical surface uses a common radius_m.
    """
    latitude, longitude, radius = np.broadcast_arrays(latitude_rad, longitude_rad, radius_m)
    if np.any(np.abs(latitude) > np.pi / 2) or np.any(radius <= 0):
        raise ValueError("Latitude must be within +/- pi/2 and radius must be positive")
    radial_xy = radius * np.cos(latitude)
    return np.stack((radial_xy * np.cos(longitude), radial_xy * np.sin(longitude), radius * np.sin(latitude)), axis=-1)


def geo_position(longitude_rad, orbital_radius_m):
    """Return Earth-fixed xyz for GEO longitude and center-to-satellite radius."""
    return earth_fixed_position(0.0, longitude_rad, orbital_radius_m)


def satellite_to_ground(satellite_position_m, ground_positions_m):
    """Return (unit directions, ranges_m), pointing from satellite to ground.

    Satellite and ground xyz arrays use a final axis of length three and must
    broadcast. Directions have the broadcast shape plus (3,); ranges omit it.
    """
    satellite = np.asarray(satellite_position_m, dtype=float)
    ground = np.asarray(ground_positions_m, dtype=float)
    if satellite.shape[-1:] != (3,) or ground.shape[-1:] != (3,):
        raise ValueError("Position vectors must have a final axis of length three")
    displacement = ground - satellite
    ranges = np.linalg.norm(displacement, axis=-1)
    if np.any(ranges == 0):
        raise ValueError("Satellite and ground positions must be distinct")
    return displacement / ranges[..., None], ranges


def ground_visibility(satellite_position_m, ground_positions_m, minimum_elevation_rad=0.0):
    """Return a boolean visibility mask for points on a spherical Earth.

    The ground-position radius defines local up. The default includes the
    geometric horizon; a positive mask requires additional elevation.
    """
    if not 0 <= minimum_elevation_rad <= np.pi / 2:
        raise ValueError("Minimum elevation must be between 0 and pi/2 radians")
    ground = np.asarray(ground_positions_m, dtype=float)
    satellite_to_point, _ = satellite_to_ground(satellite_position_m, ground)
    radius = np.linalg.norm(ground, axis=-1)
    if np.any(radius == 0):
        raise ValueError("Ground positions must be away from Earth's center")
    local_up = ground / radius[..., None]
    elevation_sine = np.sum(-satellite_to_point * local_up, axis=-1)
    return elevation_sine >= np.sin(minimum_elevation_rad)


def nadir_array_frame(geo_longitude_rad):
    """Return GEO frame axes as rows in Earth-fixed xyz coordinates.

    Local +x points east, +y points south, and +z points toward Earth center.
    Thus x cross y is z and the array's positive half-space faces Earth.
    This construction assumes an equatorial geostationary satellite.
    """
    longitude = float(geo_longitude_rad)
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    south = np.array([0.0, 0.0, -1.0])
    nadir = np.array([-np.cos(longitude), -np.sin(longitude), 0.0])
    return np.stack((east, south, nadir))


def project_to_array_frame(vectors_earth_fixed, frame_axes):
    """Project xyz vectors onto frame axes stored as the rows of a 3x3 matrix."""
    vectors = np.asarray(vectors_earth_fixed, dtype=float)
    frame = np.asarray(frame_axes, dtype=float)
    if vectors.shape[-1:] != (3,) or frame.shape != (3, 3):
        raise ValueError("Vectors need a final axis of three and frame_axes must be 3x3")
    return np.einsum("...j,ij->...i", vectors, frame, optimize=False)
