"""Automatic 25x25 windows and measured-point-only profile classification."""
import numpy as np
from scipy import ndimage
HALF, SIDE = 12, 25
def geometry_class(q95, missing):
    if q95 > 1.1 and missing > 4:
        return '疑似附着物'
    if q95 >= 0.3 and q95 <= 1.1 and (missing <= 4):
        return '疑似气泡'
    return '待复核'


def raw_line(source, row, meta):
    with source.open('rb') as handle:
        handle.seek(int(meta['source_row_offsets'][row]))
        vals = np.fromstring(handle.readline(), sep=',')
    assert len(vals) == 3 + 2 * meta['source_shape'][1]
    x, z = (vals[3::2], vals[4::2])
    ids = np.flatnonzero(np.isfinite(x) & np.isfinite(z) & ((x != 0) | (z != 0)) & (x <= 100))
    a = np.unwrap(np.arctan2(z[ids].astype('f4').astype(float) - meta['cz'], x[ids].astype('f4').astype(float) - meta['cx']))
    theta = meta['theta']
    a += 2 * np.pi * np.rint((np.median(theta) - np.median(a)) / (2 * np.pi))
    return (ids, a)


def raw_center(source, peak, meta):
    rows, theta = (meta['retained_rows'], meta['theta'])
    row = int(rows[int(np.rint(peak['y']))])
    ids, a = raw_line(source, row, meta)
    target = np.interp(peak['x'], np.arange(len(theta)), theta)
    return (int(ids[np.argmin(abs(a - target))]), row)
def raw_window(source, offsets, full_width, bounds, center, packed):
    x0, y0, x1, y1 = bounds
    radii = []
    with source.open('rb') as handle:
        handle.seek(int(offsets[y0]))
        for row in range(y0, y1):
            line = handle.readline()
            if handle.tell() - len(line) != offsets[row]:
                handle.seek(int(offsets[row]))
                line = handle.readline()
            fields = line.split(b',', 4 + 2 * x1)
            values = np.fromstring(b','.join(fields[3 + 2 * x0:3 + 2 * x1]), sep=',')
            if len(values) != 2 * (x1 - x0):
                raise ValueError(f'Unexpected fields in {source}, row {row}')
            x, z = (values[::2], values[1::2])
            valid = np.isfinite(x) & np.isfinite(z) & ((x != 0) | (z != 0))
            expected = np.unpackbits(packed[row], count=full_width)[x0:x1]
            np.testing.assert_array_equal(valid, expected)
            radii.append(np.where(valid, np.hypot(x - center[0], z - center[1]), np.nan))
    return np.array(radii)


def local_residual(radius, inside):
    yy, xx = np.indices(radius.shape, dtype=float)
    yy = (yy - yy.mean()) / max(np.ptp(yy), 1)
    xx = (xx - xx.mean()) / max(np.ptp(xx), 1)
    design = np.stack((np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy), axis=-1)
    around = ~inside & np.isfinite(radius)
    lo, hi = np.percentile(radius[around], [2, 98])
    use = around & (radius >= lo) & (radius <= hi)
    fit = np.linalg.lstsq(design[use], radius[use], rcond=None)[0]
    return design @ fit - radius
def box_sum(values):
    acc = np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)))
    return acc[SIDE:, SIDE:] - acc[:-SIDE, SIDE:] - acc[SIDE:, :-SIDE] + acc[:-SIDE, :-SIDE]


def metrics(block):
    valid = np.isfinite(block)
    miss = float(100 * (~valid).mean())
    q = float(np.percentile(block[valid], 95)) if valid.any() else None
    return dict(p95_mm=q, missing_pct=miss, area=int(block.size), valid=int(valid.sum()), prediction=geometry_class(q, miss) if q is not None else '待复核')


def classify(source, meta, peak):
    x, y = raw_center(source, peak, meta)
    height, width = meta['source_shape']
    left, top = (max(0, x - 72), max(0, y - 72))
    bounds = (left, top, min(width, x + 73), min(height, y + 73))
    radius = raw_window(source, meta['source_row_offsets'], width, bounds, (meta['cx'], meta['cz']), meta['source_acquired'])
    yy, xx = np.indices(radius.shape)
    core = (abs(xx + left - x) <= 32) & (abs(yy + top - y) <= 32)
    residual = local_residual(radius, core)
    original = metrics(residual[core])
    ax, ay = (x - left, y - top)
    fixed = residual[max(ay - HALF, 0):ay + HALF + 1, max(ax - HALF, 0):ax + HALF + 1]
    centered = metrics(fixed)
    valid = np.isfinite(residual)
    counts = box_sum(valid.astype(int))
    sums = box_sum(np.where(valid, residual, 0))
    ry, rx = np.indices(counts.shape)
    dx, dy = (rx + HALF - ax, ry + HALF - ay)
    eligible = (abs(dx) <= HALF) & (abs(dy) <= HALF) & (counts * 2 >= SIDE ** 2)
    ri, ci = np.where(eligible)
    rect = None
    if len(ri):
        means = sums[ri, ci] / counts[ri, ci]
        best = np.lexsort((ci, ri, dx[ri, ci] ** 2 + dy[ri, ci] ** 2, -means))[0]
        row, col = (int(ri[best]), int(ci[best]))
        block = residual[row:row + SIDE, col:col + SIDE]
        selected = metrics(block)
        rect = [left + col, top + row, left + col + SIDE - 1, top + row + SIDE - 1]
        assert rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3] and core[row:row + SIDE, col:col + SIDE].all()
        assert np.isclose(means[best], np.nanmean(block))
    else:
        selected = dict(p95_mm=None, missing_pct=None, area=0, valid=0, prediction='待复核')
    result = dict(tire=source.parents[1].name[:10], rank=peak['rank'], flat_xy=[peak['x'], peak['y']], raw_xy=[x, y], original=original, centered=centered, automatic=selected, selected_xyxy=rect, eligible_windows=len(ri))
    return result
def hole_component(block):
    labels, n = ndimage.label(~np.isfinite(block), structure=np.ones((3, 3)))
    if not n:
        return np.zeros_like(block, dtype=bool)
    return labels == np.bincount(labels.ravel())[1:].argmax() + 1


def get_patch(row, source, meta):
    x, y = row['raw_xy']
    h, w = meta['source_shape']
    left, top = (max(x - 72, 0), max(y - 72, 0))
    bounds = (left, top, min(x + 73, w), min(y + 73, h))
    raw = raw_window(source, meta['source_row_offsets'], w, bounds, (meta['cx'], meta['cz']), meta['source_acquired'])
    yy, xx = np.indices(raw.shape)
    core = (abs(xx + left - x) <= 32) & (abs(yy + top - y) <= 32)
    residual = local_residual(raw, core)
    x0, y0, x1, y1 = row['selected_xyxy']
    block = residual[y0 - top:y1 - top + 1, x0 - left:x1 - left + 1]
    measured = metrics(block)
    assert block.shape == (25, 25)
    for key in ('p95_mm', 'missing_pct'):
        np.testing.assert_allclose(measured[key], row['automatic'][key], atol=1e-09)
    assert measured['prediction'] == row['automatic']['prediction']
    return (residual, (left, top), block)


def assess(row, block):
    steps = [np.abs(np.diff(block, axis=axis)) for axis in (0, 1)]
    q99 = [float(np.nanpercentile(d, 99)) if np.isfinite(d).any() else None for d in steps]
    max_q = max((q for q in q99 if q is not None), default=None)
    old = row['automatic']['prediction']
    new = old
    if old == '疑似附着物' and (max_q is None or max_q <= 0.8):
        new = '疑似气泡' if row['original']['prediction'] == '疑似气泡' else '待复核'
    hole = hole_component(block)
    yy, xx = np.where(hole)
    return dict(tire=row['tire'], rank=row['rank'], raw_xy=row['raw_xy'], flat_xy=row['flat_xy'], selected_xyxy=row['selected_xyxy'], p95_mm=row['automatic']['p95_mm'], missing_pct=row['automatic']['missing_pct'], missing=int((~np.isfinite(block)).sum()), largest_hole=int(hole.sum()), hole_width=int(np.ptp(xx) + 1) if xx.size else 0, hole_height=int(np.ptp(yy) + 1) if yy.size else 0, row_step_p99_mm=q99[0], col_step_p99_mm=q99[1], max_step_p99_mm=max_q, count_over_08=sum((int((d > 0.8).sum()) for d in steps)), original=row['original']['prediction'], previous=old, prediction=new)
