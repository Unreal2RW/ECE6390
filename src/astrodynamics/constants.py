"""Course constants shared by the astrodynamics calculations.

Globals
-------
EARTH_RADIUS_M: Working Earth radius from the course notes, in meters.
EARTH_MU_M3_S2: Earth gravitational parameter, in m^3/s^2.
SIDEREAL_DAY_S: Approximate sidereal day used for GEO and Earth rotation.
SOLAR_YEAR_S: 365.25-day year used for the rotating HW1 solar direction.
SOLAR_PRESSURE_N_M2: HW1 solar-pressure force per area at 1 AU.
EARTH_ROTATION_RATE_RAD_S: Earth rotation angle per second.
"""

import math

EARTH_RADIUS_M = 6_378_000.0
EARTH_MU_M3_S2 = 3.986e14
SIDEREAL_DAY_S = 86_164.0
SOLAR_YEAR_S = 365.25 * 86_400.0
SOLAR_PRESSURE_N_M2 = 9.08e-6
EARTH_ROTATION_RATE_RAD_S = 2.0 * math.pi / SIDEREAL_DAY_S
