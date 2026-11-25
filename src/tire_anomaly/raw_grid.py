"""Acquire organized XYZ samples without filling missing raw points."""
import hashlib
from pathlib import Path
import numpy as np
MAX_VALID_X_MM, MAX_RADIAL_SPREAD_MM = 100.0, 2.0
def sample_profile(x, z, cx, cz, target, native_step, stride):
    """Select at most one original return in each angular cell, without filling."""
    acquired = np.isfinite(x) & np.isfinite(z) & ((x != 0) | (z != 0))
    source = np.flatnonzero(acquired & (x <= MAX_VALID_X_MM))
    indices = np.full(len(target), -1, dtype=np.int32)
    ambiguous = np.zeros(len(target), dtype=bool)
    if source.size < 2:
        return (indices, ambiguous)
    dx, dz = (x[source] - cx, z[source] - cz)
    angle = np.unwrap(np.arctan2(dz, dx))
    angle += 2 * np.pi * np.rint((np.median(target) - np.median(angle)) / (2 * np.pi))
    radius = np.hypot(dx, dz)
    cell = np.floor((angle - target[0]) / (stride * native_step) + 0.5).astype(np.int32)
    in_grid = (cell >= 0) & (cell < len(target))
    source, cell, angle, radius = (source[in_grid], cell[in_grid], angle[in_grid], radius[in_grid])
    error = np.abs(angle - target[cell])
    supported = error <= min(1.0, stride / 2) * native_step
    source, cell, radius, error = (source[supported], cell[supported], radius[supported], error[supported])
    low = np.full(len(target), np.inf)
    high = np.full(len(target), -np.inf)
    np.minimum.at(low, cell, radius)
    np.maximum.at(high, cell, radius)
    ambiguous = high - low > MAX_RADIAL_SPREAD_MM
    best_error = np.full(len(target), np.inf)
    np.minimum.at(best_error, cell, error)
    closest = (error == best_error[cell]) & ~ambiguous[cell]
    selected = np.full(len(target), len(x), dtype=np.int32)
    np.minimum.at(selected, cell[closest], source[closest])
    found = selected < len(x)
    indices[found] = selected[found]
    return (indices, ambiguous)

def read_grid(source, stored, stride=2):
    source = Path(source)
    source_name, sample = str(source), source.stem
    expected_sha256 = str(stored['source_sha256']) if 'source_sha256' in stored else None
    source_shape = stored['source_shape']
    offsets, retained = stored['source_row_offsets'], stored['retained_rows']
    theta, retained_y = stored['theta'], stored['y']
    cx, cz = float(stored['cx']), float(stored['cz'])
    origin = (stride - 1) / 2
    grid_rows = np.arange(len(retained) // stride) * stride + origin
    grid_columns = np.arange(len(theta) // stride) * stride + origin
    row_indices = grid_rows.astype(int)
    source_rows = retained[row_indices]
    y_origin = float(retained_y[row_indices[0]])
    angular_step = (theta[-1] - theta[0]) / (len(theta) - 1)
    target = theta[0] + angular_step * grid_columns
    shape = (len(grid_rows), len(grid_columns))
    grid = np.full((*shape, 3), np.nan, dtype=np.float32)
    source_columns = np.full(shape, -1, dtype=np.int32)
    width = 3 + 2 * int(source_shape[1])
    digest = hashlib.sha256()
    ambiguous_cells = 0
    raw_row = 0
    grid_row = 0
    with source.open('rb', buffering=1024 * 1024) as stream:
        header = stream.readline()
        fields = header.decode('utf-8-sig').strip().split(',')
        if len(fields) != width or fields[3:5] != ['X0', 'Y0']:
            raise ValueError(f'Unexpected raw CSV schema: {sample}')
        digest.update(header)
        line_offset = len(header)
        for line in stream:
            digest.update(line)
            if not line.isspace():
                if grid_row < len(source_rows) and raw_row == source_rows[grid_row]:
                    if line_offset != offsets[raw_row]:
                        raise ValueError(f'Raw scan line offset changed: {sample} row {raw_row}')
                    values = np.fromstring(line, sep=',')
                    if len(values) != width or values[2] != retained_y[row_indices[grid_row]]:
                        raise ValueError(f'Raw scan line does not match coordinates: {sample} row {raw_row}')
                    x, z = values[3::2], values[4::2]
                    columns, ambiguous = sample_profile(x, z, cx, cz, target, angular_step, stride)
                    valid = columns >= 0
                    grid[grid_row, valid, 0] = x[columns[valid]]
                    grid[grid_row, valid, 1] = values[2] - y_origin
                    grid[grid_row, valid, 2] = z[columns[valid]]
                    source_columns[grid_row] = columns
                    ambiguous_cells += int(ambiguous.sum())
                    grid_row += 1
                raw_row += 1
            line_offset += len(line)

    source_sha256 = digest.hexdigest()
    if ((expected_sha256 is not None and source_sha256 != expected_sha256) or raw_row != source_shape[0]
            or grid_row != len(source_rows)):
        raise ValueError(f'Raw CSV or coordinate binding changed: {sample}')
    metadata = dict(source=source_name, source_sha256=source_sha256,
                    source_rows=source_rows, source_columns=source_columns,
                    grid_rows=grid_rows, grid_columns=grid_columns, y_origin=y_origin,
                    cx=cx, cz=cz, theta=target, raw_shape=tuple(source_shape),
                    stride=stride, max_radial_spread_mm=MAX_RADIAL_SPREAD_MM,
                    ambiguous_cells=ambiguous_cells,
                    acquired_cells=int(np.count_nonzero(source_columns >= 0)))
    return grid, metadata
