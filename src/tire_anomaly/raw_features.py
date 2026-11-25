"""Measured 3D descriptors and fixed local repeat matching."""
import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d
from .raw_grid import MAX_RADIAL_SPREAD_MM
NORMAL_SPAN = 2
LAG_ADJUSTMENTS, REPEATS = (0, -1, 1, -2, 2), (-2, -1, 1, 2)
def descriptors(grid, cx, cz):
    """Two oriented components of the surface normal and signed inward sagitta in mm."""
    d = NORMAL_SPAN
    center = grid[d:-d, d:-d]
    left, right = (grid[d:-d, :-2 * d], grid[d:-d, 2 * d:])
    upper, lower = (grid[:-2 * d, d:-d], grid[2 * d:, d:-d])
    rx, rz = (center[..., 0] - cx, center[..., 2] - cz)
    radius = np.hypot(rx, rz)
    nonzero_radius = np.where(radius > 0, radius, np.nan)
    rx, rz = (rx / nonzero_radius, rz / nonzero_radius)
    same_sheet = np.ones(radius.shape, dtype=bool)
    for neighbor in (left, right, upper, lower):
        neighbor_radius = np.hypot(neighbor[..., 0] - cx, neighbor[..., 2] - cz)
        same_sheet &= np.abs(neighbor_radius - radius) <= MAX_RADIAL_SPREAD_MM
    normal = np.cross(right - left, lower - upper)
    normal_size = np.linalg.norm(normal, axis=-1)
    normal /= np.where(normal_size > 0, normal_size, np.nan)[..., None]
    orientation = np.where(normal[..., 0] * rx + normal[..., 2] * rz >= 0, 1, -1)
    sagitta = (left - center + (right - center) + (upper - center) + (lower - center)) * 0.25
    output = np.full(grid.shape, np.nan, dtype=np.float32)
    output[d:-d, d:-d, 0] = normal[..., 1] * orientation
    output[d:-d, d:-d, 1] = (-rz * normal[..., 0] + rx * normal[..., 2]) * orientation
    output[d:-d, d:-d, 2] = sagitta[..., 0] * rx + sagitta[..., 2] * rz
    output[d:-d, d:-d][~same_sheet] = np.nan
    output[~np.isfinite(output).all(axis=-1)] = np.nan
    return output


def masked_mean(signal, size, axis=None):
    good = np.isfinite(signal)
    if axis is None:
        weight = uniform_filter(good.astype(np.float32), size, mode='constant')
        total = uniform_filter(np.where(good, signal, 0), size, mode='constant')
    else:
        weight = uniform_filter1d(good.astype(np.float32), size, axis=axis, mode='constant')
        total = uniform_filter1d(np.where(good, signal, 0), size, axis=axis, mode='constant')
    return np.divide(total, weight, out=np.full_like(total, np.nan), where=weight >= 0.75)


def shift_rows(values, shift):
    result = np.full_like(values, np.nan)
    if shift > 0:
        result[:-shift] = values[shift:]
    elif shift < 0:
        result[-shift:] = values[:shift]
    else:
        result[:] = values
    return result


def repeat_residual(features, context, period):
    donors = []
    for repeat in REPEATS:
        best_cost = np.full(features.shape[:2], np.inf, dtype=np.float32)
        best = np.full_like(features, np.nan)
        for adjust in LAG_ADJUSTMENTS:
            lag = repeat * period + adjust
            candidate = shift_rows(features, lag)
            diff = np.abs(context - shift_rows(context, lag))
            count = np.isfinite(diff).sum(axis=-1)
            cost = np.divide(np.nansum(diff, axis=-1), count, out=np.full_like(best_cost, np.nan), where=count > 0)
            cost = masked_mean(cost, 9, axis=0)
            better = (cost < best_cost) & np.isfinite(candidate[..., 0])
            best_cost[better] = cost[better]
            best[better] = candidate[better]
        donors.append(best)
    donors = np.stack(donors)
    available = np.isfinite(donors[..., 0]).sum(axis=0) >= 2
    residual = np.full_like(features, np.nan)
    if available.any():
        residual[available] = features[available] - np.nanmedian(donors[:, available, :], axis=0)
    return residual
