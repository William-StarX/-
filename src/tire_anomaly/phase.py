"""Band-wise periodic background used by the frozen geometry score."""
import numpy as np
from . import common as b
from .common import GUARD, REFERENCES
from .raw_features import masked_mean, shift_rows, repeat_residual, LAG_ADJUSTMENTS, REPEATS
BAND, CALIBRATION_PERCENTILE, CALIBRATION_FLOOR = 32, 95, 0.10
SETTINGS = {'candidate_lags_grid': list(range(8, 25))}
def estimate_period(reference):
    records = []
    for lag in SETTINGS['candidate_lags_grid']:
        a, c = (reference[:-lag].astype(float), reference[lag:].astype(float))
        good = np.isfinite(a) & np.isfinite(c)
        count = good.sum(axis=0)
        a, c = (np.where(good, a, 0), np.where(good, c, 0))
        sa, sc = (a.sum(axis=0), c.sum(axis=0))
        n = np.maximum(count, 1)
        va = (a * a).sum(axis=0) - sa * sa / n
        vc = (c * c).sum(axis=0) - sc * sc / n
        covariance = (a * c).sum(axis=0) - sa * sc / n
        eligible = (count >= 128) & (va > 1e-12) & (vc > 1e-12)
        if not eligible.any():
            records.append(dict(lag_grid=lag, eligible_columns=0, mean_correlation=None))
            continue
        correlations = np.clip(covariance[eligible] / np.sqrt(va[eligible] * vc[eligible]), -1, 1)
        records.append(dict(lag_grid=lag, eligible_columns=int(eligible.sum()), mean_correlation=float(correlations.mean())))
    usable = [item for item in records if item['mean_correlation'] is not None]
    if not usable:
        raise ValueError('No eligible normal-reference columns for period estimation')
    chosen = max(usable, key=lambda item: (item['mean_correlation'], -item['lag_grid']))
    return (chosen['lag_grid'], records)
def flank_context(axial):
    box = masked_mean(axial, 9, axis=1)
    left, right = (np.full_like(box, np.nan), np.full_like(box, np.nan))
    left[:, 12:] = box[:, :-12]
    right[:, :-12] = box[:, 12:]
    return np.stack((left, right), axis=-1)


def common_phase_residual(stack, reference, records):
    """Use the same comparison rows for the three normal/depth representations."""
    anchor = stack[..., 0]
    global_period, _ = estimate_period(anchor[reference])
    context = flank_context(anchor)
    residual = np.full_like(stack, np.nan)
    guard = GUARD // b.STEP
    intervals = (slice(0, reference.start - guard), reference, slice(reference.stop + guard, len(stack)))
    for first in range(0, stack.shape[1], BAND):
        cols = slice(first, first + BAND)
        try:
            period, values = estimate_period(anchor[reference, cols])
            local_correlation = next((row['mean_correlation'] for row in values if row['lag_grid'] == period))
        except ValueError:
            period, local_correlation = (global_period, None)
        records.append(dict(x_start_px=first * b.STEP, period_px=period * b.STEP, reference_correlation=local_correlation))
        for rows in intervals:
            residual[rows, cols] = repeat_residual(stack[rows, cols], context[rows, cols], period)
    return residual
def donor_background_at_new_points(old, context, rows, cols, period):
    """Return the median original donor vector and donor count for each query.

    Inputs share an evaluation interval and angular band. A query need not
    have its own original geometry, but its original flank context must
    support matching. Backgrounds with fewer than two donors remain NaN.
    """
    rows, cols = (np.asarray(rows), np.asarray(cols))
    background = np.full((len(rows), old.shape[-1]), np.nan, dtype=old.dtype)
    if not len(rows):
        return (background, np.zeros(0, dtype=int))
    active_cols, cropped_cols = np.unique(cols, return_inverse=True)
    context = context[:, active_cols]
    donors = []
    for repeat in REPEATS:
        best_cost = np.full(len(rows), np.inf, dtype=np.float32)
        best = np.full_like(background, np.nan)
        for adjust in LAG_ADJUSTMENTS:
            lag = repeat * period + adjust
            shifted_rows = rows + lag
            in_bounds = (shifted_rows >= 0) & (shifted_rows < len(old))
            if not in_bounds.any():
                continue
            candidate = old[np.clip(shifted_rows, 0, len(old) - 1), cols]
            diff = np.abs(context - shift_rows(context, lag))
            count = np.isfinite(diff).sum(axis=-1)
            cost = np.divide(np.nansum(diff, axis=-1), count, out=np.full(count.shape, np.nan, dtype=np.float32), where=count > 0)
            cost = masked_mean(cost, 9, axis=0)[rows, cropped_cols]
            better = in_bounds & (cost < best_cost) & np.isfinite(candidate).all(axis=-1)
            best_cost[better] = cost[better]
            best[better] = candidate[better]
        donors.append(best)
    donors = np.stack(donors)
    counts = np.isfinite(donors[..., 0]).sum(axis=0)
    available = counts >= 2
    if available.any():
        background[available] = np.nanmedian(donors[:, available], axis=0)
    return (background, counts)
