"""Generate the selected frozen normal/depth map for local supplementation."""
import numpy as np
from . import common as b
from .common import REFERENCES, GUARD, evaluation_mask
from .normal_depth import scale_features
from .raw_features import descriptors
from .raw_grid import read_grid
from .phase import (BAND, CALIBRATION_PERCENTILE, CALIBRATION_FLOOR,
                    common_phase_residual, flank_context, donor_background_at_new_points)
RADIUS_MM = 4.0
def score_residual(residual, reference):
    normal = np.hypot(residual[..., 0], residual[..., 1])
    depth = residual[..., 2]
    valid = np.isfinite(residual).all(axis=-1)
    reference_amplitudes = np.stack((normal[reference], np.abs(depth[reference])), axis=-1)
    ref_valid = valid[reference]
    if not ref_valid.any():
        raise ValueError('Missing 4 mm normal/depth reference residuals')
    pooled = np.percentile(reference_amplitudes[ref_valid], CALIBRATION_PERCENTILE, axis=0)
    floor = np.maximum(CALIBRATION_FLOOR * pooled, 1e-06)
    score = np.full(normal.shape, np.nan, dtype=np.float32)
    calibration = []
    for start in range(0, depth.shape[1], BAND):
        cols = slice(start, start + BAND)
        values = reference_amplitudes[:, cols][ref_valid[:, cols]]
        local = np.percentile(values, CALIBRATION_PERCENTILE, axis=0) if len(values) >= 128 else pooled
        local = np.maximum(local, floor).astype(np.float32)
        joint = np.hypot(normal[:, cols] / local[0], np.maximum(depth[:, cols], 0) / local[1])
        score[:, cols] = np.where(valid[:, cols], joint / np.sqrt(2), np.nan)
        calibration.append(dict(x_start_px=start * b.STEP, reference_cells=len(values), normal_depth_p95=local.tolist()))
    return (score, calibration)


def compute(grid, raw, rows_mm, fitted_radius, shape):
    cx, cz = (raw['cx'], raw['cz'])
    same_sheet = np.isfinite(descriptors(grid, cx, cz)[..., 0])
    dx = float(fitted_radius * np.median(np.diff(raw['theta'])))
    dy = float(np.median(np.diff(rows_mm)))
    assert dx > 0 and dy > 0 and np.all(np.diff(rows_mm) > 0)
    half_rows, half_cols = (max(1, round(RADIUS_MM / dy)), max(1, round(RADIUS_MM / dx)))
    field = scale_features(grid, cx, cz, half_rows, half_cols, same_sheet=same_sheet)
    if field.shape[1] < shape[1]:
        field = np.pad(field, ((0, 0), (0, shape[1] - field.shape[1]), (0, 0)), constant_values=np.nan)
    assert field.shape == (*shape, 3)
    geometry_valid = np.isfinite(field).all(axis=-1)
    field[~geometry_valid] = np.nan
    reference = slice(REFERENCES[0] // b.STEP, (REFERENCES[-1] + b.TILE) // b.STEP)
    periods = []
    residual = common_phase_residual(field, reference, periods)
    score, calibration = score_residual(residual, reference)
    score[~evaluation_mask(shape)] = np.nan
    assert not np.isfinite(score[~geometry_valid]).any()
    return (score, dict(window_stride2=[2 * half_rows, 2 * half_cols], geometry_cells=int(geometry_valid.sum()), initial_spacing_mm=[dy, dx], phase_bands=periods, reference_scales=calibration))
def extend_scores(original_score, old_field, new_field, prior):
    old_geometry = np.isfinite(old_field).all(axis=-1)
    new_geometry = np.isfinite(new_field).all(axis=-1)
    old_finite = np.isfinite(original_score)
    assert not (old_finite & ~old_geometry).any()
    assert not (old_geometry & ~new_geometry).any()
    np.testing.assert_array_equal(old_field[old_geometry], new_field[old_geometry])
    new_points = new_geometry & ~old_geometry & evaluation_mask(original_score.shape)
    output = original_score.copy()
    context = flank_context(old_field[..., 0])
    ref0, ref1 = (REFERENCES[0] // b.STEP, (REFERENCES[-1] + b.TILE) // b.STEP)
    guard = GUARD // b.STEP
    intervals = (slice(0, ref0 - guard), slice(ref1 + guard, len(original_score)))
    periods = {r['x_start_px']: r['period_px'] for r in prior['phase_bands']}
    calibrations = {r['x_start_px']: r['normal_depth_p95'] for r in prior['reference_scales']}
    for left in range(0, original_score.shape[1], BAND):
        columns = slice(left, left + BAND)
        start_px = left * b.STEP
        assert periods[start_px] % b.STEP == 0
        period = periods[start_px] // b.STEP
        normal_scale, depth_scale = np.asarray(calibrations[start_px], dtype=np.float32)
        assert normal_scale > 0 and depth_scale > 0
        for rows in intervals:
            yy, xx = np.where(new_points[rows, columns])
            if len(yy) == 0:
                continue
            donors, counts = donor_background_at_new_points(old_field[rows, columns], context[rows, columns], yy, xx, period)
            accepted = (counts >= 2) & np.isfinite(donors).all(axis=-1)
            y, x = (yy[accepted] + rows.start, xx[accepted] + left)
            residual = new_field[y, x] - donors[accepted]
            normal = np.hypot(residual[:, 0], residual[:, 1]) / normal_scale
            depth = np.maximum(residual[:, 2], 0) / depth_scale
            output[y, x] = np.hypot(normal, depth) / np.sqrt(2)
    np.testing.assert_array_equal(output[old_finite], original_score[old_finite])
    added = np.isfinite(output) & ~old_finite
    assert not (added & ~new_points).any()
    return (output, dict(raw_new_geometry_cells=int(new_points.sum()), added_scored_cells=int(added.sum()), missing_frozen_donor_cells=int((new_points & ~added).sum())))

def make_score(source, meta, shape):
    grid, raw = read_grid(source, meta, stride=2)
    rows_mm = meta['y'][raw['grid_rows'].astype(int)] - raw['y_origin']
    original, prior = compute(grid, raw, rows_mm, float(meta['radius']), shape)
    sheet = np.isfinite(descriptors(grid, raw['cx'], raw['cz'])[..., 0])
    half_rows, half_cols = (size // 2 for size in prior['window_stride2'])
    fields = []
    for spread in (2.0, 2.5):
        field = scale_features(grid, raw['cx'], raw['cz'], half_rows, half_cols,
                               same_sheet=sheet, max_radial_spread_mm=spread)
        if field.shape[1] < shape[1]:
            field = np.pad(field, ((0, 0), (0, shape[1]-field.shape[1]), (0, 0)), constant_values=np.nan)
        fields.append(field)
    output, extra = extend_scores(original, fields[0], fields[1], prior)
    return output, dict(prior, **extra)
