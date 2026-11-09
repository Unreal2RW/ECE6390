"""HW3 ground target, rectangular arrays, coverage, and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from tqdm import tqdm

from src.antenna.arrays import planar_grid_positions, steering_phases
from src.astrodynamics.geometry import earth_fixed_position, geo_position, ground_visibility, nadir_array_frame, project_to_array_frame, satellite_to_ground
from src.rf.links import db_to_linear, dbm_to_watts, friis_received_power_w, linear_to_db, minimum_total_transmit_power_w, watts_to_dbm, wavelength_m


FREQUENCY_HZ = 30.0e9
GEO_RADIUS_M = 42_400_000.0
EARTH_RADIUS_M = GEO_RADIUS_M - 36_000_000.0
RECEIVE_GAIN_DBI = 10.0
RECEIVER_THRESHOLD_DBM = -80.0
POLAR_SAMPLES = 720
AZIMUTH_SAMPLES = 360
SIZE_FACTORS = np.linspace(0.6, 1.4, 9)
CORNER_CUT_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.45)
FRAME_WIDTH_FRACTIONS = (0.05, 0.1, 0.2, 0.3)
LATTICE_STRIDES = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class GroundMap:
    georgia: np.ndarray
    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    pixel_area_km2: float


@dataclass(frozen=True)
class GroundGeometry:
    directions_array: np.ndarray
    ranges_m: np.ndarray
    visible: np.ndarray


@dataclass(frozen=True)
class ArrayDesign:
    active_mask: np.ndarray
    phases_rad: np.ndarray
    steer_direction: np.ndarray
    cut_kind: str
    cut_rows: int
    cut_columns: int
    row_stride: int
    column_stride: int


@dataclass(frozen=True)
class CoverageResult:
    transmit_gain: np.ndarray
    received_power_w: np.ndarray
    total_transmit_power_w: float


def load_georgia_map(path: Path) -> GroundMap:
    """Read this file's grayscale mask and convert latitude to north-positive."""
    ga = loadmat(path, simplify_cells=True)["GA"]
    georgia = np.asarray(ga["data"]) >= 128
    longitude = np.asarray(ga["lon"], dtype=float)
    latitude = -np.asarray(ga["lat"], dtype=float)
    return GroundMap(georgia, longitude, latitude, float(ga["cellsize"])**2/1e6)


def ground_geometry(ground_map: GroundMap, satellite_longitude_deg: float) -> GroundGeometry:
    """Use ASD spherical geometry; array axes are east, south, and nadir."""
    longitude_rad = np.deg2rad(satellite_longitude_deg)
    ground = earth_fixed_position(np.deg2rad(ground_map.latitude_deg), np.deg2rad(ground_map.longitude_deg), EARTH_RADIUS_M)
    satellite = geo_position(longitude_rad, GEO_RADIUS_M)
    directions, ranges = satellite_to_ground(satellite, ground)
    visible = ground_visibility(satellite, ground)
    frame = nadir_array_frame(longitude_rad)
    return GroundGeometry(project_to_array_frame(directions, frame), ranges, visible)


def make_array(shape: tuple[int, int], steer_direction: np.ndarray, cut_kind: str = "full", fraction: float = 0.0, row_stride: int = 1, column_stride: int = 1) -> ArrayDesign:
    """Build a lambda/2 rectangular grid with equal-amplitude on elements."""
    wavelength = float(wavelength_m(FREQUENCY_HZ))
    positions = planar_grid_positions(shape[0], shape[1], wavelength/2, wavelength/2)
    active = np.ones(shape, dtype=bool)
    cut_rows = cut_columns = 0
    if cut_kind == "corners":
        cut_rows = int(round(shape[0]*fraction))
        cut_columns = int(round(shape[1]*fraction))
        active[:cut_rows, :cut_columns] = False
        active[:cut_rows, -cut_columns:] = False
        active[-cut_rows:, :cut_columns] = False
        active[-cut_rows:, -cut_columns:] = False
    elif cut_kind == "frame":
        cut_rows = int(round(shape[0]*(1 - 2*fraction)))
        cut_columns = int(round(shape[1]*(1 - 2*fraction)))
        row = (shape[0] - cut_rows)//2
        column = (shape[1] - cut_columns)//2
        active[row:row + cut_rows, column:column + cut_columns] = False
    elif cut_kind == "lattice":
        active[:] = False
        row_start = ((shape[0] - 1) % row_stride)//2
        column_start = ((shape[1] - 1) % column_stride)//2
        active[row_start::row_stride, column_start::column_stride] = True
    phases = steering_phases(positions, steer_direction, wavelength).reshape(shape)
    return ArrayDesign(active, phases, steer_direction, cut_kind, cut_rows, cut_columns, row_stride, column_stride)


def line_power(offset: np.ndarray, count: int, stride: int = 1) -> np.ndarray:
    phase = stride*offset/2
    denominator = np.sinc(phase)
    ratio = np.divide(np.sinc(count*phase), denominator, out=np.ones_like(phase), where=np.abs(denominator) > 1e-12)
    return ratio**2


def rectangular_power(directions: np.ndarray, shape: tuple[int, int], steer_direction: np.ndarray) -> np.ndarray:
    """Normalized power from the finite element sum in Pratt B.10-B.11."""
    return line_power(directions[..., 0] - steer_direction[0], shape[1])*line_power(directions[..., 1] - steer_direction[1], shape[0])


def block_field(dx: np.ndarray, dy: np.ndarray, shape: tuple[int, int], row: int, rows: int, column: int, columns: int) -> np.ndarray:
    """Sum one contiguous block of the rectangular grid."""
    x_center = column + (columns - 1)/2 - (shape[1] - 1)/2
    y_center = row + (rows - 1)/2 - (shape[0] - 1)/2
    amplitude = rows*columns*np.sinc(columns*dx/2)*np.sinc(rows*dy/2)/(np.sinc(dx/2)*np.sinc(dy/2))
    return amplitude*np.exp(1j*np.pi*(x_center*dx + y_center*dy))


def array_pattern_power(directions: np.ndarray, array: ArrayDesign) -> np.ndarray:
    """Actual power for the selected on/off mask, normalized by its on count."""
    shape = array.active_mask.shape
    if array.cut_kind == "full":
        return rectangular_power(directions, shape, array.steer_direction)
    if array.cut_kind == "lattice":
        columns_on = int(np.count_nonzero(np.any(array.active_mask, axis=0)))
        rows_on = int(np.count_nonzero(np.any(array.active_mask, axis=1)))
        return line_power(directions[..., 0] - array.steer_direction[0], columns_on, array.column_stride)*line_power(directions[..., 1] - array.steer_direction[1], rows_on, array.row_stride)
    dx = directions[..., 0] - array.steer_direction[0]
    dy = directions[..., 1] - array.steer_direction[1]
    field = block_field(dx, dy, shape, 0, shape[0], 0, shape[1])
    if array.cut_kind == "corners":
        for row in (0, shape[0] - array.cut_rows):
            for column in (0, shape[1] - array.cut_columns):
                field -= block_field(dx, dy, shape, row, array.cut_rows, column, array.cut_columns)
    elif array.cut_kind == "frame":
        row = (shape[0] - array.cut_rows)//2
        column = (shape[1] - array.cut_columns)//2
        field -= block_field(dx, dy, shape, row, array.cut_rows, column, array.cut_columns)
    return np.abs(field/np.count_nonzero(array.active_mask))**2


def target_coverage_ellipse(ground_map: GroundMap):
    """Fit a compact rotated ground ellipse before selecting the satellite or array."""
    inside = ground_map.georgia
    origin_lon = float(np.mean(ground_map.longitude_deg[inside]))
    origin_lat = float(np.mean(ground_map.latitude_deg[inside]))
    x = EARTH_RADIUS_M*np.cos(np.deg2rad(origin_lat))*np.deg2rad(ground_map.longitude_deg - origin_lon)
    y = EARTH_RADIUS_M*np.deg2rad(ground_map.latitude_deg - origin_lat)
    xy = np.stack((x, y), axis=-1)
    points = xy[inside]
    edge = points[ConvexHull(points).vertices]
    q = np.vstack((edge.T, np.ones(len(edge))))
    weights = np.full(len(edge), 1/len(edge))
    for _ in range(1000):
        matrix = (q*weights) @ q.T
        distances = np.sum(q*np.linalg.solve(matrix, q), axis=0)
        index = int(np.argmax(distances))
        if distances[index] <= 3.0001:
            break
        step = (distances[index] - 3)/(3*(distances[index] - 1))
        weights *= 1 - step
        weights[index] += step
    center = weights @ edge
    shifted = edge - center
    matrix = np.linalg.inv(shifted.T @ (weights[:, None]*shifted))/2
    delta = points - center
    matrix /= (1 + 1e-10)*np.max(np.einsum("ni,ij,nj->n", delta, matrix, delta, optimize=False))
    ellipse = np.einsum("...i,ij,...j->...", xy - center, matrix, xy - center, optimize=False) <= 1
    axes = 1/np.sqrt(np.linalg.eigvalsh(matrix))
    center_lon = origin_lon + np.rad2deg(center[0]/(EARTH_RADIUS_M*np.cos(np.deg2rad(origin_lat))))
    center_lat = origin_lat + np.rad2deg(center[1]/EARTH_RADIUS_M)
    east_scale = np.cos(np.deg2rad(center_lat))/np.cos(np.deg2rad(origin_lat))
    scale = np.array([east_scale, 1.0])
    ground_shape = np.linalg.inv(matrix)*np.outer(scale, scale)
    return center_lat, center_lon, axes, ellipse, ground_shape


def ellipse_longitude(center_lat: float, center_lon: float, ground_shape: np.ndarray, longitude_limits: tuple[float, float]) -> float:
    """Align the local ellipse with the array's east/south direction coordinates."""
    latitude, longitude = np.deg2rad([center_lat, center_lon])
    ground = earth_fixed_position(latitude, longitude, EARTH_RADIUS_M)
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.array([-np.sin(latitude)*np.cos(longitude), -np.sin(latitude)*np.sin(longitude), np.cos(latitude)])
    ground_axes = np.stack((east, north))

    def cross_term(longitude_deg):
        satellite = geo_position(np.deg2rad(longitude_deg), GEO_RADIUS_M)
        direction, distance = satellite_to_ground(satellite, ground)
        frame = nadir_array_frame(np.deg2rad(longitude_deg))
        direction = project_to_array_frame(direction, frame)
        local_axes = project_to_array_frame(ground_axes, frame)
        range_gradient = np.einsum("ij,j->i", local_axes, direction, optimize=False)
        mapping = (local_axes[:, :2].T - np.outer(direction[:2], range_gradient))/distance
        angular_shape = np.einsum("ia,ab,jb->ij", mapping, ground_shape, mapping, optimize=False)
        return float(angular_shape[0, 1])

    return float(brentq(cross_term, longitude_limits[0], longitude_limits[1], xtol=1e-6))


def search_rectangles(ground_map: GroundMap):
    """Map the ground target to a GEO beam, then compare rectangular grids."""
    inside = ground_map.georgia
    center_lat, center_lon, _, ellipse, ground_shape = target_coverage_ellipse(ground_map)
    target_ground = earth_fixed_position(np.deg2rad(center_lat), np.deg2rad(center_lon), EARTH_RADIUS_M)
    ga_lat = np.deg2rad(ground_map.latitude_deg[inside])
    ga_lon = ground_map.longitude_deg[inside]
    visible_halfspan = np.rad2deg(np.arccos(EARTH_RADIUS_M/(GEO_RADIUS_M*np.cos(ga_lat))))
    longitude_limits = (float(np.max(ga_lon - visible_halfspan)), float(np.min(ga_lon + visible_halfspan)))
    longitude_deg = ellipse_longitude(center_lat, center_lon, ground_shape, longitude_limits)
    geometry = ground_geometry(ground_map, longitude_deg)
    satellite = geo_position(np.deg2rad(longitude_deg), GEO_RADIUS_M)
    center_direction, _ = satellite_to_ground(satellite, target_ground)
    steer_direction = project_to_array_frame(center_direction, nadir_array_frame(np.deg2rad(longitude_deg)))
    target_xy = geometry.directions_array[ellipse, :2]
    width_deg = np.rad2deg(np.max(target_xy, axis=0) - np.min(target_xy, axis=0))
    estimate = np.maximum(2, np.rint(102/width_deg).astype(int))
    row_sizes = np.unique(np.maximum(2, np.rint(estimate[1]*SIZE_FACTORS).astype(int)))
    column_sizes = np.unique(np.maximum(2, np.rint(estimate[0]*SIZE_FACTORS).astype(int)))
    candidates = []
    grid_progress = tqdm(total=len(row_sizes)*len(column_sizes), desc="Rectangular search", unit="array")
    for rows in row_sizes:
        for columns in column_sizes:
            shape = (int(rows), int(columns))
            power = rectangular_power(geometry.directions_array, shape, steer_direction)
            relative = power/geometry.ranges_m**2
            weakest_ga = np.min(relative[inside])
            outside_count = int(np.count_nonzero(~inside & geometry.visible & (relative > weakest_ga)))
            score = (outside_count, rows*columns)
            candidates.append((score, longitude_deg, geometry, shape, steer_direction, "full", 0.0))
            grid_progress.update()
    grid_progress.close()
    by_coverage = sorted(candidates, key=lambda item: item[0])[:5]
    by_count = sorted(candidates, key=lambda item: (item[0][1], item[0][0]))[:5]
    seed_count = len({(item[1], item[3]) for item in by_coverage + by_count})
    thinning_progress = tqdm(total=seed_count, desc="Thinning search", unit="grid")
    tried = set()
    for _, longitude_deg, geometry, shape, steer_direction, _, _ in by_coverage + by_count:
        if (longitude_deg, shape) in tried:
            continue
        tried.add((longitude_deg, shape))
        for cut_kind, fractions in (("corners", CORNER_CUT_FRACTIONS), ("frame", FRAME_WIDTH_FRACTIONS)):
            for fraction in fractions:
                array = make_array(shape, steer_direction, cut_kind, fraction)
                relative = array_pattern_power(geometry.directions_array, array)/geometry.ranges_m**2
                weakest_ga = np.min(relative[inside])
                outside_count = int(np.count_nonzero(~inside & geometry.visible & (relative > weakest_ga)))
                score = (outside_count, int(np.count_nonzero(array.active_mask)))
                candidates.append((score, longitude_deg, geometry, shape, steer_direction, cut_kind, fraction))
        earth_sine = EARTH_RADIUS_M/GEO_RADIUS_M
        for row_stride in LATTICE_STRIDES:
            for column_stride in LATTICE_STRIDES:
                if row_stride == column_stride == 1:
                    continue
                if 2/row_stride <= earth_sine + abs(steer_direction[1]) or 2/column_stride <= earth_sine + abs(steer_direction[0]):
                    continue
                array = make_array(shape, steer_direction, "lattice", row_stride=row_stride, column_stride=column_stride)
                relative = array_pattern_power(geometry.directions_array, array)/geometry.ranges_m**2
                weakest_ga = np.min(relative[inside])
                outside_count = int(np.count_nonzero(~inside & geometry.visible & (relative > weakest_ga)))
                score = (outside_count, int(np.count_nonzero(array.active_mask)))
                candidates.append((score, longitude_deg, geometry, shape, steer_direction, "lattice", (row_stride, column_stride)))
        thinning_progress.update()
    thinning_progress.close()
    target_outside = int(np.count_nonzero(ellipse & ~inside))
    feasible = [item for item in candidates if item[0][0] <= target_outside]
    selected = min(feasible, key=lambda item: (item[0][1], item[0][0])) if feasible else min(candidates, key=lambda item: item[0])
    best_coverage = min(candidates, key=lambda item: item[0])
    score, longitude_deg, geometry, shape, steer_direction, cut_kind, fraction = selected
    summary = (len(feasible), best_coverage[0], min(candidates, key=lambda item: (item[0][1], item[0][0]))[0])
    array = make_array(shape, steer_direction, cut_kind, row_stride=fraction[0], column_stride=fraction[1]) if cut_kind == "lattice" else make_array(shape, steer_direction, cut_kind, fraction)
    return longitude_deg, geometry, array, ellipse, (center_lat, center_lon), score, summary, longitude_limits


def array_coverage(ground_map: GroundMap, geometry: GroundGeometry, array: ArrayDesign) -> CoverageResult:
    """Normalize the rectangular pattern, then set power by the weakest GA link."""
    wavelength = float(wavelength_m(FREQUENCY_HZ))
    theta = (np.arange(POLAR_SAMPLES) + 0.5)*np.pi/(2*POLAR_SAMPLES)
    phi = (np.arange(AZIMUTH_SAMPLES) + 0.5)*2*np.pi/AZIMUTH_SAMPLES
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack((np.sin(theta_grid)*np.cos(phi_grid), np.sin(theta_grid)*np.sin(phi_grid), np.cos(theta_grid)), axis=-1)
    pattern = array_pattern_power(directions, array)
    integral = np.pi**2*np.sum(pattern*np.sin(theta_grid))/(POLAR_SAMPLES*AZIMUTH_SAMPLES)
    gain = np.zeros(ground_map.georgia.shape)
    gain[geometry.visible] = 4*np.pi*array_pattern_power(geometry.directions_array[geometry.visible], array)/integral
    receive_gain = float(db_to_linear(RECEIVE_GAIN_DBI))
    threshold_w = float(dbm_to_watts(RECEIVER_THRESHOLD_DBM))
    transmit_power = minimum_total_transmit_power_w(threshold_w, gain, receive_gain, geometry.ranges_m, wavelength, target_mask=ground_map.georgia)
    received_power = friis_received_power_w(transmit_power, gain, receive_gain, geometry.ranges_m, wavelength)
    return CoverageResult(gain, received_power, transmit_power)


def save_received_power_plot(ground_map: GroundMap, result: CoverageResult, satellite_longitude_deg: float, target_ellipse: np.ndarray, output_path: Path) -> None:
    """Compare the candidate ellipse with the actual receiver-threshold contour."""
    received_dbm = np.ma.masked_invalid(watts_to_dbm(result.received_power_w))
    longitude, latitude = ground_map.longitude_deg, ground_map.latitude_deg
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.pcolormesh(longitude, latitude, received_dbm, shading="nearest", cmap="viridis")
    axis.contour(longitude, latitude, ground_map.georgia.astype(float), levels=[0.5], colors="black", linewidths=1.2)
    axis.contour(longitude, latitude, received_dbm, levels=[RECEIVER_THRESHOLD_DBM], colors="white", linewidths=1.2, linestyles="--")
    axis.contour(longitude, latitude, target_ellipse.astype(float), levels=[0.5], colors="orange", linewidths=1.2, linestyles=":")
    boundary_label = Line2D([], [], color="black", linewidth=1.2, label="Georgia boundary")
    threshold_label = Line2D([], [], color="white", linewidth=1.2, linestyle="--", label=f"{RECEIVER_THRESHOLD_DBM:g} dBm threshold")
    target_label = Line2D([], [], color="orange", linewidth=1.2, linestyle=":", label="Candidate ground ellipse")
    axis.legend(handles=[boundary_label, threshold_label, target_label], facecolor="0.85", framealpha=1.0)
    axis.set_xlabel("Longitude (deg, east-positive)")
    axis.set_ylabel("Latitude (deg, north-positive)")
    axis.set_title(f"Received power: GEO {satellite_longitude_deg:g} deg, total TX power {result.total_transmit_power_w:.4g} W")
    axis.set_aspect("equal", adjustable="box")
    figure.colorbar(image, ax=axis, label="Received power (dBm)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_array_plot(array: ArrayDesign, output_path: Path) -> None:
    """Save the on/off element layout and feed phases."""
    rows, columns = array.active_mask.shape
    pitch_mm = 1000 * float(wavelength_m(FREQUENCY_HZ))/2
    x_edges = (np.arange(columns + 1) - columns/2)*pitch_mm
    y_edges = (np.arange(rows + 1) - rows/2)*pitch_mm
    phases_deg = (np.rad2deg(array.phases_rad) + 180) % 360 - 180
    phases_deg = np.ma.masked_where(~array.active_mask, phases_deg)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    active_image = axes[0].pcolormesh(x_edges, y_edges, array.active_mask, shading="flat", cmap=ListedColormap(["0.85", "#2060a0"]), vmin=0, vmax=1, edgecolors="white", linewidth=0.6, antialiased=False)
    rows_on, columns_on = np.nonzero(array.active_mask)
    x_on = (columns_on - (columns - 1)/2)*pitch_mm
    y_on = (rows_on - (rows - 1)/2)*pitch_mm
    axes[0].scatter(x_on, y_on, s=14, marker="s", color="#2060a0", edgecolor="white", linewidth=0.3)
    active_bar = figure.colorbar(active_image, ax=axes[0], ticks=[0, 1])
    active_bar.set_ticklabels(["Off", "On"])
    axes[0].set_title(f"Active elements: {np.count_nonzero(array.active_mask)}")
    phase_image = axes[1].pcolormesh(x_edges, y_edges, phases_deg, shading="flat", cmap="twilight", vmin=-180, vmax=180, edgecolors="white", linewidth=0.6, antialiased=False)
    axes[1].scatter(x_on, y_on, c=np.asarray(phases_deg[array.active_mask]), s=14, marker="s", cmap="twilight", vmin=-180, vmax=180, edgecolor="white", linewidth=0.3)
    axes[1].set_facecolor("0.85")
    axes[1].set_title("Feed phases (gray = off)")
    figure.colorbar(phase_image, ax=axes[1], label="Phase (deg)", ticks=[-180, -90, 0, 90, 180])
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("Array x (mm)")
        axis.set_ylabel("Array y (mm)")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
