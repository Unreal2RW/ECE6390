"""Planar geometry and general position-vector antenna array factors.

Public functions
----------------
planar_grid_positions: Rectangular xy-grid element positions in meters.
excitation_weights: Combine element amplitudes, phases, and an optional mask.
steering_phases: Feed phases that align fields in a selected unit direction.
array_factor: Evaluate complex field factors at arbitrary unit directions.

The course convention is exp(+j*2*pi*f*t) in time and exp(-j*k*r) for
outward propagation. The relative array sum is
    AF(s) = sum_i a_i exp(j*(k*r_i dot s + beta_i)).
Positions and unit directions use the same Cartesian frame; phase is in radians,
wavelength and positions in meters. The last coordinate axis has length three.
The array factor is a relative field, not an absolute gain or a power value.
"""

import numpy as np


def planar_grid_positions(rows, columns, spacing_x_m, spacing_y_m, centered=True):
    """Return (rows*columns, 3) xy-plane positions, with columns varying fastest.

    rows follow +y and columns follow +x. Centering translates the grid so its
    geometric center is at the origin; it does not change the power pattern.
    """
    if not isinstance(rows, int) or not isinstance(columns, int) or rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive integers")
    if spacing_x_m <= 0 or spacing_y_m <= 0:
        raise ValueError("Grid spacings must be positive")
    x = np.arange(columns, dtype=float)
    y = np.arange(rows, dtype=float)
    if centered:
        x -= (columns - 1) / 2
        y -= (rows - 1) / 2
    xx, yy = np.meshgrid(x * spacing_x_m, y * spacing_y_m)
    return np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)


def excitation_weights(amplitudes, phases_rad, active_mask=None):
    """Return complex weights a*exp(j*beta); masked-off entries are zero.

    Amplitudes are real nonnegative relative field amplitudes. Scalar or array
    inputs broadcast; grid-shaped output may be passed directly to array_factor.
    """
    amplitudes, phases = np.broadcast_arrays(np.asarray(amplitudes, dtype=float), np.asarray(phases_rad, dtype=float))
    if np.any(~np.isfinite(amplitudes)) or np.any(amplitudes < 0) or np.any(~np.isfinite(phases)):
        raise ValueError("Amplitudes must be finite and nonnegative; phases must be finite")
    weights = amplitudes * np.exp(1j * phases)
    return weights if active_mask is None else np.where(np.asarray(active_mask, dtype=bool), weights, 0.0)


def steering_phases(positions_m, direction, wavelength_m):
    """Return beta_i = -k*r_i dot direction, in radians, for one unit direction."""
    positions, _ = _prepare_array(positions_m, wavelength_m, None)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("Steering direction must be a finite three-component unit vector")
    return -(2 * np.pi / wavelength_m) * (positions @ direction)


def array_factor(positions_m, directions, wavelength_m, weights=None, direction_chunk=256, element_chunk=256):
    """Return the complex AF with the same leading shape as directions.

    positions_m has shape (elements, 3), directions has shape (..., 3), and
    weights contains one complex excitation per element (ones by default).
    Both direction and element blocks are bounded, including for large grids.
    """
    positions, weights = _prepare_array(positions_m, wavelength_m, weights)
    directions = np.asarray(directions, dtype=float)
    if directions.shape[-1:] != (3,):
        raise ValueError("Directions must have a final axis of length three")
    lengths = np.linalg.norm(directions, axis=-1)
    if not np.all(np.isclose(lengths, 1.0, rtol=1e-7, atol=1e-9)):
        raise ValueError("Each observation direction must be a finite unit vector")
    _check_chunks(direction_chunk, element_chunk)
    flat_directions = directions.reshape(-1, 3)
    result = np.empty(flat_directions.shape[0], dtype=complex)
    for start in range(0, len(flat_directions), direction_chunk):
        stop = start + direction_chunk
        result[start:stop] = _array_factor_block(positions, flat_directions[start:stop], 2 * np.pi / wavelength_m, weights, element_chunk)
    return result.reshape(directions.shape[:-1])


def _prepare_array(positions_m, wavelength_m, weights):
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be a nonempty finite (elements, 3) array")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive")
    weights = np.ones(len(positions), dtype=complex) if weights is None else np.asarray(weights, dtype=complex).reshape(-1)
    if len(weights) != len(positions) or not np.all(np.isfinite(weights)):
        raise ValueError("Provide one finite excitation per position")
    return positions, weights


def _check_chunks(direction_chunk, element_chunk):
    if not isinstance(direction_chunk, int) or direction_chunk < 1 or not isinstance(element_chunk, int) or element_chunk < 1:
        raise ValueError("Chunk sizes must be positive integers")


def _array_factor_block(positions, directions, wavenumber, weights, element_chunk):
    result = np.zeros(len(directions), dtype=complex)
    for start in range(0, len(positions), element_chunk):
        stop = start + element_chunk
        phase = wavenumber * (directions @ positions[start:stop].T)
        result += np.exp(1j * phase) @ weights[start:stop]
    return result
