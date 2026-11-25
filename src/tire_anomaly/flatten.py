from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.signal import convolve, convolve2d
from scipy.sparse import diags

@dataclass
class Config:
    maxValidX: float = 100.0
    targetNx: int | None = None
    leftPercentile: float = 2.0
    rightPercentile: float = 98.0
    minRowCountFactor: float = 0.5
    maxRowCountFactor: float = 3.0
    fitPointsPerRow: int = 100
    fitMaxRows: int = 5000
    fitIterations: int = 8
    rawSpikeWindow: int = 7
    rawSpikeMadScale: float = 8.0
    rawSpikeMinMm: float = 0.8
    rawSpikeMaxRun: int = 8
    rawCliffJumpMinMm: float = 2.0
    rawCliffMaxWidth: int = 20
    rawCliffReturnToleranceMm: float = 1.0
    whittakerLambda: float = 30.0
    whittakerBlockRows: int = 256
    whittakerGrossErrorMm: float = 1.5
    whittakerModerateMinMm: float = 0.35
    whittakerMadScale: float = 8.0
    whittakerSupportRows: int = 5
    whittakerSupportCols: int = 3
    whittakerMaxIsolatedSupport: int = 2

def matlab_round(x):
    x = np.asarray(x)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def sample_ids(n, limit):
    return matlab_round(np.linspace(1, n, limit)).astype(int) - 1

def pct(v, q):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q, method='linear')) if v.size else np.nan

def fit_circle(x, z, valid, cfg):
    rows = np.flatnonzero(valid.sum(axis=1) >= 3)
    if rows.size > cfg.fitMaxRows:
        rows = rows[sample_ids(rows.size, cfg.fitMaxRows)]
    sx, sz = ([], [])
    for r in rows:
        ids = np.flatnonzero(valid[r])
        if ids.size > cfg.fitPointsPerRow:
            ids = ids[sample_ids(ids.size, cfg.fitPointsPerRow)]
        sx.append(x[r, ids])
        sz.append(z[r, ids])
    if not sx:
        raise ValueError('No scan line with at least three valid points')
    sx = np.concatenate(sx).astype(np.float64, copy=False)
    sz = np.concatenate(sz).astype(np.float64, copy=False)
    a = np.column_stack((sx, sz, np.ones(sx.size)))
    if np.linalg.matrix_rank(a) < 3:
        raise ValueError('Circle fit is degenerate')
    sol = np.linalg.lstsq(a, -(sx * sx + sz * sz), rcond=None)[0]
    cx, cz = -sol[:2] / 2
    eps = np.finfo(float).eps
    radius = np.sqrt(max(eps, sol[0] ** 2 + sol[1] ** 2 - 4 * sol[2])) / 2
    sigma = np.nan
    for _ in range(cfg.fitIterations):
        dx, dz = (sx - cx, sz - cz)
        d = np.maximum(np.hypot(dx, dz), eps)
        e = d - radius
        med = np.median(e)
        sigma = 1.4826 * np.median(np.abs(e - med))
        if not np.isfinite(sigma) or sigma < 1e-10:
            break
        u = (e - med) / (4.685 * sigma)
        w = np.zeros_like(u)
        inside = np.abs(u) < 1
        w[inside] = (1 - u[inside] ** 2) ** 2
        sw = np.sqrt(np.maximum(w, 1e-08))
        j = np.column_stack((-dx / d, -dz / d, -np.ones_like(d)))
        delta = np.linalg.lstsq(j * sw[:, None], -e * sw, rcond=None)[0]
        cx += delta[0]
        cz += delta[1]
        radius += delta[2]
        if np.linalg.norm(delta) < 1e-09:
            break
    if not np.isfinite([cx, cz, radius]).all() or radius <= 0:
        raise ValueError('Circle fit failed')
    return (cx, cz, radius, dict(sampleCount=int(sx.size), rowCount=int(rows.size), robustSigma=float(sigma)))

def moving_median(v, width):
    left, right = (width // 2, (width - 1) // 2)
    a = np.pad(v, (left, right), constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(np.lib.stride_tricks.sliding_window_view(a, width), axis=1)

def normalized_conv(v):
    a = np.asarray(v, dtype=np.float32)
    finite = np.isfinite(a)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    num = convolve(np.where(finite, a, np.float32(0)), kernel, mode='same', method='direct')
    den = convolve(finite.astype(np.float32), kernel, mode='same', method='direct')
    out = num / np.maximum(den, np.float32(1e-06))
    out[den <= 0] = np.nan
    return out.astype(np.float64)

def clean_profile(v, cfg):
    v = np.asarray(v, dtype=np.float64).copy()
    finite = np.isfinite(v)
    if finite.sum() < 3:
        return (v, 0)
    med = moving_median(v, cfg.rawSpikeWindow)
    residual = np.abs(v - med)
    mad = moving_median(residual, cfg.rawSpikeWindow)
    candidate = finite & (residual > np.maximum(cfg.rawSpikeMinMm, cfg.rawSpikeMadScale * mad))
    repair = np.zeros(v.size, bool)
    edges = np.diff(np.r_[False, candidate, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start <= cfg.rawSpikeMaxRun:
            repair[start:stop] = True
    delta = np.diff(v)
    cliffs = np.flatnonzero(np.isfinite(delta) & (np.abs(delta) >= cfg.rawCliffJumpMinMm))
    used = np.zeros(cliffs.size, bool)
    for a, le in enumerate(cliffs):
        if used[a]:
            continue
        for b in range(a + 1, cliffs.size):
            re = cliffs[b]
            if re - le > cfg.rawCliffMaxWidth:
                break
            if np.sign(delta[le]) != np.sign(delta[re]) and abs(v[le] - v[re + 1]) <= cfg.rawCliffReturnToleranceMm:
                repair[le + 1:re + 1] = True
                used[a] = used[b] = True
                break
    count = int(repair.sum())
    if count:
        v[repair] = np.nan
        good = np.flatnonzero(np.isfinite(v))
        if good.size >= 2:
            v[repair] = np.interp(np.flatnonzero(repair), good, v[good], left=np.nan, right=np.nan)
        else:
            raise ValueError('Too few points remain for profile interpolation')
    return (normalized_conv(v), count)

def unwrap_to_grid(x, y, z, valid, cx, cz, radius, cfg, *, output_path=None):
    counts = valid.sum(axis=1)
    positive = counts[counts > 0]
    if not positive.size:
        raise ValueError('No valid points')
    median_count = np.median(positive)
    good = (counts >= max(2, matlab_round(cfg.minRowCountFactor * median_count))) & (counts <= matlab_round(cfg.maxRowCountFactor * median_count))
    row_median = np.full(x.shape[0], np.nan)
    for r in np.flatnonzero(good):
        ids = valid[r]
        row_median[r] = np.median(np.unwrap(np.arctan2(z[r, ids] - cz, x[r, ids] - cx)))
    ref = np.nanmedian(row_median[good])
    left = np.full(x.shape[0], np.nan)
    right = left.copy()

    def angles(r):
        ids = valid[r]
        theta = np.unwrap(np.arctan2(z[r, ids] - cz, x[r, ids] - cx))
        return theta + 2 * np.pi * matlab_round((ref - row_median[r]) / (2 * np.pi))
    for r in np.flatnonzero(good):
        t = angles(r)
        left[r], right[r] = (pct(t, cfg.leftPercentile), pct(t, cfg.rightPercentile))
    rows = np.flatnonzero(good & np.isfinite(left) & np.isfinite(right) & (right > left))
    if not rows.size:
        raise ValueError('No usable scan lines for unwrapping')
    lo, hi = (np.median(left[rows]), np.median(right[rows]))
    if hi <= lo:
        raise ValueError('Invalid common angular interval')
    nx = max(100, int(matlab_round(np.median(counts[rows]) if cfg.targetNx is None else cfg.targetNx)))
    theta_grid = np.linspace(lo, hi, nx)
    shape = (rows.size, nx)
    if output_path is None:
        radial = np.empty(shape, dtype=np.float32)
    else:
        radial = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float32, shape=shape)
    filt = np.exp(-np.arange(-7, 8, dtype=float) ** 2 / (2 * 3.0 ** 2))
    filt /= filt.sum()
    repaired = 0
    for i, r in enumerate(rows):
        radial[i] = np.nan
        t = angles(r)
        rr = np.hypot(x[r, valid[r]] - cx, z[r, valid[r]] - cz)
        order = np.argsort(t, kind='stable')
        t, rr = (t[order], rr[order])
        tu, first, multiplicity = np.unique(t, return_index=True, return_counts=True)
        if tu.size < 2:
            continue
        ru = rr if tu.size == t.size else np.array([np.median(rr[a:a + n]) for a, n in zip(first, multiplicity)])
        ru, count = clean_profile(ru, cfg)
        repaired += count
        inside = (theta_grid >= tu[0]) & (theta_grid <= tu[-1])
        tmp = np.interp(theta_grid[inside], tu, ru)
        if tmp.size:
            smoothed = convolve(tmp, filt, mode='same', method='direct')
            tmp[7:-8] = smoothed[7:-8]
            radial[i, inside] = tmp
        if (i + 1) % 500 == 0:
            print(f'Unwrap {i + 1}/{rows.size}', flush=True)
    return (radial, dict(thetaGrid=theta_grid, arcGrid=radius * (theta_grid - theta_grid[0]), yGrid=y[rows], retainedRows=rows + 1), repaired)

def repair_whittaker(radial, cfg):
    valid = np.isfinite(radial)
    filled = radial.copy()
    for row in filled:
        good = np.flatnonzero(np.isfinite(row))
        if good.size:
            missing = np.flatnonzero(~np.isfinite(row))
            row[missing] = np.interp(missing, good, row[good])
    values = filled[np.isfinite(filled)]
    if not values.size:
        return (radial.copy(), dict(grossCount=0, isolatedCount=0))
    filled[~np.isfinite(filled)] = np.median(values)
    del values
    nc = radial.shape[1]
    d = diags([np.ones(nc - 2), -2 * np.ones(nc - 2), np.ones(nc - 2)], [0, 1, 2], shape=(nc - 2, nc), format='csc')
    a = (diags(np.ones(nc), 0) + cfg.whittakerLambda * (d.T @ d)).tocsc()
    band = np.zeros((3, nc))
    for k in range(3):
        band[k, :nc - k] = a.diagonal(-k)
    factor = cholesky_banded(band, lower=True)

    def solve(v):
        out = np.empty_like(v, dtype=np.float32)
        for start in range(0, len(v), cfg.whittakerBlockRows):
            stop = min(len(v), start + cfg.whittakerBlockRows)
            out[start:stop] = cho_solve_banded((factor, True), v[start:stop].T.astype(np.float64), check_finite=False).T
        return out
    rough = solve(filled)
    seed = valid & (np.abs(filled - rough) >= cfg.whittakerGrossErrorMm)
    refit = filled.copy()
    refit[seed] = rough[seed]
    del rough, seed
    fitted = solve(refit)
    del refit
    residual = np.abs(filled - fitted)
    del filled
    gross = valid & (residual >= cfg.whittakerGrossErrorMm)
    sigma = np.float32(1.4826) * np.median(residual, axis=1)
    threshold = np.maximum(cfg.whittakerModerateMinMm, cfg.whittakerMadScale * sigma)
    moderate = valid & (residual >= threshold[:, None]) & ~gross
    del residual
    support = convolve2d(moderate.astype(np.float32), np.ones((cfg.whittakerSupportRows, cfg.whittakerSupportCols), np.float32), mode='same', boundary='fill')
    isolated = moderate & (support <= cfg.whittakerMaxIsolatedSupport)
    repair = gross | isolated
    cleaned = radial.copy()
    cleaned[repair] = fitted[repair]
    cleaned[~valid] = np.nan
    return (cleaned, dict(grossCount=int(gross.sum()), isolatedCount=int(isolated.sum())))
