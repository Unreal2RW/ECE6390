# HW3: GEO antenna coverage over Georgia

`solve_hw3.py` runs the design and saves the results. `hw3_helpers.py` contains the assignment-specific geometry, array search, coverage calculations, and plotting functions; reusable geometry, antenna, and RF calculations live in `src/`.

## Run

From the repository root:

```bash
python -m pip install -e .
python Homeworks/HW3/solve_hw3.py
```

Supply the assignment's `GAmap.mat` in this directory before running. It is not distributed here. The loader expects a MATLAB structure named `GA` with `data`, `lon`, `lat`, and `cellsize` fields. For this supplied dataset, values at least 128 identify Georgia; the stored latitude is negated to obtain north-positive degrees. Longitude is east-positive, and `cellsize` is in meters.

## Method

1. Fit a compact, rotated ellipse enclosing the Georgia raster as a candidate receiver-threshold footprint.
2. Search GEO longitudes visible from every Georgia pixel. For each longitude, steer toward the ellipse center and compare filled rectangular arrays on a half-wavelength grid.
3. Compare corner cuts, rectangular frames, and regular row/column thinning on shortlisted grids. Active elements have equal feed amplitudes and steering phases. Finite element sums give the actual array patterns; the ellipse is a design target.
4. Normalize the selected pattern over the radiating hemisphere, apply Friis, and set total transmit power so the weakest Georgia pixel receives -80 dBm. Report leakage above that threshold within the supplied raster, including its area and percentage of Georgia's raster area.

The search selects the fewest active elements among tested candidates meeting the ellipse leakage benchmark. If none meet it, it selects the candidate with the least leakage, breaking ties by active count. This finite search does not establish a global minimum.

Physical constants, longitude spacing, rectangle-size factors, thinning choices, and angular integration resolution are at the top of `hw3_helpers.py`. The array frame uses east, south, and nadir axes. The rectangular aperture beamwidth estimate initializes sizes; final coverage uses the array pattern and each pixel's slant range.

## Outputs

- `hw3_received_power.mat`: received power in dBm on the original raster, coordinates, Georgia and target masks, on/off mask, feed phases, steering direction, satellite longitude, and total transmit power.
- `hw3_received_power.png`: received-power map with Georgia, target ellipse, and -80 dBm contours.
- `hw3_array.png`: element on/off layout and feed phases.

Progress bars cover the searches; elapsed times and final design results appear in the terminal. Output files remain local and are ignored by Git.

## Course references

HW3 assignment statement; ASD5 look-angle handout; Pratt, *Satellite Communications*, 3rd edition, Appendix B, especially the aperture field sums and rectangular-aperture discussion on printed page 709. The ellipse fit and finite search are design choices used by this implementation.
