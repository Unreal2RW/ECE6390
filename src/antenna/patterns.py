"""Power patterns and gain normalized by total radiated power.

Public functions
----------------
ideal_half_space_power: Unit element power in +z, zero in -z.
field_to_power: Relative total power from an array factor and element pattern.
integrate_array_power: Numerically integrate the relative pattern over 4*pi sr.
directional_directivity: Normalize relative directional power by that integral.
directional_gain: Apply a stated radiation efficiency to directivity.

The normalization follows D = 4*pi*U / integral(U dOmega). Numerical midpoint
integration in azimuth and either cos(theta) or theta evaluates the spherical integral;
it is an approximation whose angular sampling must resolve the pattern. Identical
equally oriented element patterns and the ideal array-factor model are assumed.
"""

import numpy as np

from .arrays import _array_factor_block, _check_chunks, _prepare_array


def ideal_half_space_power(directions):
    """Return 1 for array-frame z >= 0 and 0 for z < 0, for directions (..., 3)."""
    directions = np.asarray(directions, dtype=float)
    if directions.shape[-1:] != (3,):
        raise ValueError("Directions must have a final axis of length three")
    return np.where(directions[..., 2] >= 0, 1.0, 0.0)


def field_to_power(array_factors, element_power=1.0):
    """Return relative power |AF|^2 times the element's relative power pattern."""
    element_power = np.asarray(element_power, dtype=float)
    if np.any(~np.isfinite(element_power)) or np.any(element_power < 0):
        raise ValueError("Element power pattern must be finite and nonnegative")
    return np.abs(array_factors)**2 * element_power


def integrate_array_power(positions_m, wavelength_m, weights=None, element_power_pattern=None, polar_samples=128, azimuth_samples=256, direction_chunk=256, element_chunk=256, progress=None, polar_grid="cosine"):
    """Return integral(element_power(s)*|AF(s)|^2 dOmega) over the sphere.

    element_power_pattern is a vectorized callable accepting array-frame unit
    directions with shape (n, 3) and returning n relative power values. None
    means an isotropic full-sphere element. Choose even polar_samples for the
    ideal half-space boundary. Sampling resolution is independent of chunk size.
    An optional progress object receives update(n) after each direction block.
    The theta grid gives even angular resolution near a narrow main beam.
    """
    positions, weights = _prepare_array(positions_m, wavelength_m, weights)
    _check_chunks(direction_chunk, element_chunk)
    if not isinstance(polar_samples, int) or polar_samples < 1 or not isinstance(azimuth_samples, int) or azimuth_samples < 1:
        raise ValueError("Sample counts must be positive integers")
    sample_count = polar_samples * azimuth_samples
    wavenumber = 2 * np.pi / wavelength_m
    power_sum = 0.0
    for start in range(0, sample_count, direction_chunk):
        indices = np.arange(start, min(start + direction_chunk, sample_count))
        if polar_grid == "theta":
            theta = np.pi * (indices // azimuth_samples + 0.5) / polar_samples
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            angular_weight = sin_theta
        else:
            cos_theta = -1.0 + 2.0 * (indices // azimuth_samples + 0.5) / polar_samples
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            angular_weight = 1.0
        phi = 2.0 * np.pi * (indices % azimuth_samples + 0.5) / azimuth_samples
        directions = np.stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=-1)
        af = _array_factor_block(positions, directions, wavenumber, weights, element_chunk)
        if element_power_pattern is None:
            element_power = 1.0
        else:
            element_power = np.asarray(element_power_pattern(directions), dtype=float)
            if element_power.shape not in ((), (len(directions),)) or np.any(~np.isfinite(element_power)) or np.any(element_power < 0):
                raise ValueError("Element pattern must return a finite nonnegative scalar or one value per direction")
        power_sum += np.sum(np.abs(af)**2 * element_power * angular_weight)
        if progress is not None:
            progress.update(len(directions))
    integral = float(((2 * np.pi**2 if polar_grid == "theta" else 4 * np.pi) / sample_count) * power_sum)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("The integrated radiation pattern must be finite and positive")
    return integral


def directional_directivity(relative_power, integrated_power):
    """Return linear D = 4*pi*relative_power/integrated_power, not dBi."""
    if not np.isfinite(integrated_power) or integrated_power <= 0:
        raise ValueError("integrated_power must be finite and positive")
    relative_power = np.asarray(relative_power, dtype=float)
    if np.any(~np.isfinite(relative_power)) or np.any(relative_power < 0):
        raise ValueError("relative_power must be finite and nonnegative")
    return 4 * np.pi * relative_power / integrated_power


def directional_gain(directivity, radiation_efficiency=1.0):
    """Return linear gain D*efficiency; efficiency is relative to accepted power."""
    if not 0 <= radiation_efficiency <= 1:
        raise ValueError("Radiation efficiency must be between zero and one")
    directivity = np.asarray(directivity, dtype=float)
    if np.any(~np.isfinite(directivity)) or np.any(directivity < 0):
        raise ValueError("Directivity must be finite and nonnegative")
    return directivity * radiation_efficiency
