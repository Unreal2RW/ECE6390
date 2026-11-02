"""Wavelength, power conversions, and free-space Friis link calculations.

Public globals
--------------
SPEED_OF_LIGHT_M_S: Lecture approximation for propagation in free space.

Public functions
----------------
wavelength_m, wavenumber_rad_m: Convert frequency in Hz to wavelength and k.
linear_to_db, db_to_linear: Convert linear power ratios and dB.
watts_to_dbw, dbw_to_watts, watts_to_dbm, dbm_to_watts: Power conversions.
dbw_to_dbm, dbm_to_dbw: Convert the two logarithmic power reference levels.
free_space_path_loss_db: Positive spreading loss for range and wavelength.
friis_received_power_w: Receive-antenna output power from total transmit power.
required_transmit_power_w: Per-location transmit power needed for a threshold.
minimum_total_transmit_power_w: Largest requirement over selected locations.
total_element_power_w: Sum explicitly supplied per-element accepted powers.

Friis uses directional linear gains, distance and wavelength in meters, and
powers in watts. Its transmit input is the total power accepted by the whole
transmit array, not power per element. For an ideal lossless matched array this
equals total radiated power. A supplied transmit gain includes any efficiency
already accounted for; do not apply the same loss again.
"""

import numpy as np

SPEED_OF_LIGHT_M_S = 3.0e8


def wavelength_m(frequency_hz, speed_m_s=SPEED_OF_LIGHT_M_S):
    """Return wavelength in meters for frequency in Hz and wave speed in m/s."""
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0) or not np.isfinite(speed_m_s) or speed_m_s <= 0:
        raise ValueError("Frequency and wave speed must be finite and positive")
    return speed_m_s / frequency


def wavenumber_rad_m(frequency_hz, speed_m_s=SPEED_OF_LIGHT_M_S):
    """Return k = 2*pi/lambda in radians per meter."""
    return 2 * np.pi / wavelength_m(frequency_hz, speed_m_s)


def linear_to_db(linear_power_ratio):
    """Return 10*log10 of a nonnegative power ratio; zero becomes -inf."""
    ratio = np.asarray(linear_power_ratio, dtype=float)
    if np.any(np.isnan(ratio)) or np.any(ratio < 0):
        raise ValueError("Power ratios must be nonnegative and not NaN")
    with np.errstate(divide="ignore"):
        return 10 * np.log10(ratio)


def db_to_linear(value_db):
    """Return the linear power ratio for a value in dB; -inf becomes zero."""
    return np.power(10.0, np.asarray(value_db, dtype=float) / 10.0)


def watts_to_dbw(power_w):
    """Convert watts to dBW, relative to 1 W."""
    return linear_to_db(power_w)


def dbw_to_watts(power_dbw):
    """Convert dBW to watts."""
    return db_to_linear(power_dbw)


def watts_to_dbm(power_w):
    """Convert watts to dBm, relative to 1 mW."""
    return linear_to_db(np.asarray(power_w, dtype=float) / 1e-3)


def dbm_to_watts(power_dbm):
    """Convert dBm to watts."""
    return 1e-3 * db_to_linear(power_dbm)


def dbw_to_dbm(power_dbw):
    """Convert dBW to dBm."""
    return np.asarray(power_dbw, dtype=float) + 30.0


def dbm_to_dbw(power_dbm):
    """Convert dBm to dBW."""
    return np.asarray(power_dbm, dtype=float) - 30.0


def free_space_path_loss_db(range_m, wavelength_m):
    """Return positive-convention loss 20*log10(4*pi*range/lambda), in dB."""
    distance, wavelength = np.broadcast_arrays(np.asarray(range_m, dtype=float), np.asarray(wavelength_m, dtype=float))
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0) or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("Range and wavelength must be finite and positive")
    return 20 * np.log10(4 * np.pi * distance / wavelength)


def _friis_inputs(transmit_gain, receive_gain, range_m, wavelength_m):
    tx_gain, rx_gain, distance, wavelength = np.broadcast_arrays(np.asarray(transmit_gain, dtype=float), np.asarray(receive_gain, dtype=float), np.asarray(range_m, dtype=float), np.asarray(wavelength_m, dtype=float))
    if np.any(~np.isfinite(tx_gain)) or np.any(tx_gain < 0) or np.any(~np.isfinite(rx_gain)) or np.any(rx_gain < 0):
        raise ValueError("Linear gains must be finite and nonnegative")
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0) or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("Range and wavelength must be finite and positive")
    return tx_gain * rx_gain * (wavelength / (4 * np.pi * distance))**2


def friis_received_power_w(total_transmit_power_w, transmit_gain, receive_gain, range_m, wavelength_m):
    """Return receiver-output watts from total antenna-input watts using Friis."""
    transmit_power = np.asarray(total_transmit_power_w, dtype=float)
    if np.any(~np.isfinite(transmit_power)) or np.any(transmit_power < 0):
        raise ValueError("Total transmit power must be finite and nonnegative")
    return transmit_power * _friis_inputs(transmit_gain, receive_gain, range_m, wavelength_m)


def required_transmit_power_w(receiver_threshold_w, transmit_gain, receive_gain, range_m, wavelength_m):
    """Return total antenna-input watts required at each location.

    A positive threshold at a zero-gain location yields infinity. Inputs
    broadcast and threshold is power at the receive-antenna output.
    """
    threshold = np.asarray(receiver_threshold_w, dtype=float)
    if np.any(~np.isfinite(threshold)) or np.any(threshold < 0):
        raise ValueError("Receiver thresholds must be finite and nonnegative")
    factor = _friis_inputs(transmit_gain, receive_gain, range_m, wavelength_m)
    threshold, factor = np.broadcast_arrays(threshold, factor)
    required = np.full(threshold.shape, np.inf)
    np.divide(threshold, factor, out=required, where=factor > 0)
    return np.where(threshold == 0, 0.0, required)


def minimum_total_transmit_power_w(receiver_threshold_w, transmit_gain, receive_gain, range_m, wavelength_m, target_mask=None):
    """Return minimum total watts meeting the threshold at all selected points.

    target_mask broadcasts to the per-location result; None selects all. The
    return is infinity if a selected positive threshold has zero link gain.
    """
    required = required_transmit_power_w(receiver_threshold_w, transmit_gain, receive_gain, range_m, wavelength_m)
    if target_mask is None:
        selected = np.ones(required.shape, dtype=bool)
    else:
        required, selected = np.broadcast_arrays(required, np.asarray(target_mask, dtype=bool))
    if not np.any(selected):
        raise ValueError("Select at least one target location")
    return float(np.max(required[selected]))


def total_element_power_w(element_powers_w):
    """Sum explicitly specified accepted powers of all feeds, in watts.

    This sum uses stated feed powers; relative field weights alone do not set
    absolute watts. For equal active feeds it is active-count times feed power.
    """
    power = np.asarray(element_powers_w, dtype=float)
    if np.any(~np.isfinite(power)) or np.any(power < 0):
        raise ValueError("Element feed powers must be finite and nonnegative")
    return float(np.sum(power))
