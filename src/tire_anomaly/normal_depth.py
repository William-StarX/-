"""The fixed 4 mm PCA normal/depth neighborhood representation."""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
def _box_counts(integral, y0, y1, x0, x1):
    return integral[y1[:, None], x1] - integral[y0[:, None], x1] - integral[y1[:, None], x0] + integral[y0[:, None], x0]


def scale_features(grid, cx, cz, half_rows, half_cols, *, same_sheet, min_valid=0.8, max_radial_spread_mm=2.0):
    """Return axial/angle normal components and negative mean radius on the 4px grid.

    An output cell is centered between the four measured stride-2 grid points
    in its 2x2 center. The even window shares that same center. NaN denotes
    an unavailable neighborhood or one mixing separated radial returns.
    """
    height, width = grid.shape[:2]
    if grid.shape != (height, width, 3) or same_sheet.shape != (height, width):
        raise ValueError('Expected H×W×3 acquired XYZ and an H×W sheet mask')
    if min(half_rows, half_cols) < 1 or not 0 < min_valid <= 1:
        raise ValueError('Expected positive window spans and a valid fraction in (0, 1]')
    result = np.full((height // 2, (width + 1) // 2, 3), np.nan, dtype=np.float32)
    size = (2 * half_rows, 2 * half_cols)
    area = size[0] * size[1]
    y = np.arange(1, height, 2)
    x = np.arange(1, width, 2)
    y = y[(y >= half_rows) & (y + half_rows <= height)]
    x = x[(x >= half_cols) & (x + half_cols <= width)]
    if not len(y) or not len(x):
        return result
    x0, x1 = (x - half_cols, x + half_cols)
    offsets_y, offsets_x = np.array([[-1, -1], [-1, 0], [0, -1], [0, 0]]).T
    for first in range(0, len(y), 96):
        centers = y[first:first + 96]
        start, stop = (centers[0] - half_rows, centers[-1] + half_rows)
        rows = centers - start
        y0, y1 = (rows - half_rows, rows + half_rows)
        tile = grid[start:stop].astype(np.float64)
        valid = np.isfinite(tile).all(axis=-1) & same_sheet[start:stop]
        integral = np.pad(valid, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        count = _box_counts(integral, y0, y1, x0, x1)
        supports = _box_counts(integral, rows - 1, rows + 1, x - 1, x + 1) >= 3
        supports &= count >= min_valid * area
        for a, b, c, d in ((y0, rows, x0, x1), (rows, y1, x0, x1), (y0, y1, x0, x), (y0, y1, x, x1)):
            supports &= _box_counts(integral, a, b, c, d) >= area / 4
        ii, jj = np.nonzero(supports)
        if not len(ii):
            continue
        ys, xs = (rows[ii], x[jj])
        cy, cx_idx = (ys[:, None] + offsets_y, xs[:, None] + offsets_x)
        center = tile[cy, cx_idx]
        center_valid = valid[cy, cx_idx]
        radius = np.hypot(tile[..., 0] - cx, tile[..., 2] - cz)
        center_radius = np.where(center_valid, radius[cy, cx_idx], 0).sum(axis=1) / center_valid.sum(axis=1)
        low = minimum_filter(np.where(valid, radius, np.inf), size, mode='constant', cval=np.inf)[ys, xs]
        high = maximum_filter(np.where(valid, radius, -np.inf), size, mode='constant', cval=-np.inf)[ys, xs]
        allowed = (center_radius - low <= max_radial_spread_mm) & (high - center_radius <= max_radial_spread_mm)
        ii, jj, ys, xs = (ii[allowed], jj[allowed], ys[allowed], xs[allowed])
        if not len(ii):
            continue
        center_xz = np.where(center_valid[allowed, :, None], center[allowed][:, :, [0, 2]], 0).sum(axis=1)
        center_xz /= center_valid[allowed].sum(axis=1, keepdims=True)
        radial = center_xz - [cx, cz]
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        fraction = count[ii, jj] / area

        def average(values):
            sums = uniform_filter(np.where(valid, values, 0.0), size, mode='constant')
            return sums[ys, xs] / fraction
        depth = -average(radius)
        tile -= tile[valid].mean(axis=0)
        mean = np.column_stack([average(tile[..., axis]) for axis in range(3)])
        covariance = np.empty((len(ii), 3, 3), dtype=np.float64)
        for axis in range(3):
            for other in range(axis + 1):
                cov = average(tile[..., axis] * tile[..., other]) - mean[:, axis] * mean[:, other]
                covariance[:, axis, other] = covariance[:, other, axis] = cov
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, :, 0]
        nx, axial, nz = normal.T
        rx, rz = radial.T
        orientation = np.where(nx * rx + nz * rz >= 0, 1.0, -1.0)
        angular = nz * rx - nx * rz
        nondegenerate = (eigenvalues[:, 1] > 1e-12) & np.isfinite(radial).all(axis=-1)
        features = np.column_stack((orientation * axial, orientation * angular, depth))
        result[centers[ii[nondegenerate]] // 2, x[jj[nondegenerate]] // 2] = features[nondegenerate]
    return result
