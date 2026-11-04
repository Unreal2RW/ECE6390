"""Design HW3 coverage from a ground ellipse and rectangular antenna grids."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import savemat


COURSE_ROOT = Path(__file__).resolve().parents[2]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from hw3_helpers import *


def main():
    start_time = perf_counter()
    print(f"Frequency: {FREQUENCY_HZ/1e9:g} GHz; GEO radius: {GEO_RADIUS_M/1000:g} km; Earth radius: {EARTH_RADIUS_M/1000:g} km")

    print("Phase 1: load the Georgia map")
    phase_start = perf_counter()
    ground_map = load_georgia_map(Path(__file__).with_name("GAmap.mat"))
    inside = ground_map.georgia
    print(f"  map shape: {inside.shape}; Georgia pixels: {np.count_nonzero(inside)}")
    print(f"  Georgia latitude: {np.min(ground_map.latitude_deg[inside]):.6f} to {np.max(ground_map.latitude_deg[inside]):.6f} deg")
    print(f"  Georgia longitude: {np.min(ground_map.longitude_deg[inside]):.6f} to {np.max(ground_map.longitude_deg[inside]):.6f} deg (east-positive)")
    print(f"  elapsed: {perf_counter() - phase_start:.2f} s")

    print("Phase 2: fit a candidate ellipse and compare rectangular arrays")
    phase_start = perf_counter()
    satellite_longitude_deg, geometry, array, target_ellipse, ellipse_center, search_score, search_summary, longitude_search = search_rectangles(ground_map)
    print(f"  candidate -80 dBm ellipse center: {ellipse_center[0]:.4f} deg N, {ellipse_center[1]:.4f} deg E")
    print(f"  GEO longitudes visible from all Georgia: {longitude_search[0][0]:.3f} to {longitude_search[0][1]:.3f} deg")
    print(f"  searched {len(longitude_search[1])} longitudes in that interval at {LONGITUDE_STEP_DEG:g} deg spacing, plus the ellipse-center longitude")
    print(f"  candidate ground ellipse: {np.count_nonzero(target_ellipse & ~inside & geometry.visible)} outside pixels on the map")
    print(f"  selected rectangular grid: {array.active_mask.shape}, {array.cut_kind}; active elements: {search_score[1]}; outside pixels: {search_score[0]}")
    if array.cut_kind == "lattice":
        print(f"  active row/column strides: {array.row_stride}/{array.column_stride} grid positions")
    print(f"  arrays at or below ellipse leakage: {search_summary[0]}")
    print(f"  best coverage tried: {search_summary[1][0]} outside pixels with {search_summary[1][1]} active elements")
    print(f"  fewest active tried: {search_summary[2][1]} elements with {search_summary[2][0]} outside pixels")
    print(f"  elapsed: {perf_counter() - phase_start:.2f} s")

    print("Phase 3: normalize the selected pattern and calculate Friis received power")
    phase_start = perf_counter()
    ga_range_km = geometry.ranges_m[inside]/1000
    result = array_coverage(ground_map, geometry, array)
    print(f"  elapsed: {perf_counter() - phase_start:.2f} s")
    ga_gain_dbi = linear_to_db(result.transmit_gain[inside])
    received_dbm = watts_to_dbm(result.received_power_w)
    ga_received_dbm = received_dbm[inside]
    threshold_w = float(dbm_to_watts(RECEIVER_THRESHOLD_DBM))
    outside_count = np.count_nonzero(~inside & (result.received_power_w > threshold_w))
    georgia_area_km2 = np.count_nonzero(inside)*ground_map.pixel_area_km2
    outside_area_km2 = outside_count*ground_map.pixel_area_km2
    weakest = np.flatnonzero(inside)[np.argmin(ga_received_dbm)]
    print(f"  GEO longitude: {satellite_longitude_deg:g} deg; array shape: {array.active_mask.shape}; active elements: {np.count_nonzero(array.active_mask)}")
    print(f"  beam center from nadir: theta={np.rad2deg(np.arccos(array.steer_direction[2])):.3f} deg, phi={np.rad2deg(np.arctan2(array.steer_direction[1], array.steer_direction[0])):.3f} deg (east toward south)")
    print(f"  Georgia slant range: {np.min(ga_range_km):.3f} to {np.max(ga_range_km):.3f} km")
    print(f"  Georgia transmit gain: {np.min(ga_gain_dbi):.3f} to {np.max(ga_gain_dbi):.3f} dBi")
    print(f"  required total transmit power: {result.total_transmit_power_w:.6g} W ({float(watts_to_dbm(result.total_transmit_power_w)):.3f} dBm)")
    print(f"  Georgia received power: {np.min(ga_received_dbm):.6f} to {np.max(ga_received_dbm):.6f} dBm")
    print(f"  weakest Georgia pixel: ({ground_map.latitude_deg.flat[weakest]:.6f}, {ground_map.longitude_deg.flat[weakest]:.6f}) deg latitude/longitude")
    print(f"  outside pixels above {RECEIVER_THRESHOLD_DBM:g} dBm: {outside_count}; area: {outside_area_km2:.1f} km² within the map ({100*outside_area_km2/georgia_area_km2:.1f}% of Georgia's {georgia_area_km2:.1f} km² raster area)")

    print("Phase 4: save the final received-power raster and two plots")
    phase_start = perf_counter()
    raster_path = Path(__file__).with_name("hw3_received_power.mat")
    coverage_path = Path(__file__).with_name("hw3_received_power.png")
    array_path = Path(__file__).with_name("hw3_array.png")
    savemat(raster_path, {"received_power_dbm": received_dbm, "georgia_mask": inside, "target_ellipse": target_ellipse, "longitude_deg": ground_map.longitude_deg, "latitude_deg": ground_map.latitude_deg, "element_on": array.active_mask, "element_phase_rad": array.phases_rad, "steer_direction": array.steer_direction, "total_transmit_power_w": result.total_transmit_power_w, "satellite_longitude_deg": satellite_longitude_deg}, do_compression=True)
    save_received_power_plot(ground_map, result, satellite_longitude_deg, target_ellipse, coverage_path)
    save_array_plot(array, array_path)
    print(f"  saved received-power raster: {raster_path}")
    print(f"  saved received-power map: {coverage_path}")
    print(f"  saved element layout and phases: {array_path}")
    print(f"  elapsed: {perf_counter() - phase_start:.2f} s")
    print(f"Total elapsed time: {perf_counter() - start_time:.2f} s")
    return ground_map, geometry, array, result


if __name__ == "__main__":
    main()
