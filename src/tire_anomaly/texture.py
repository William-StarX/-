"""Fixed local-height and directional-texture preprocessing."""
import numpy as np
from scipy import ndimage
STEP, CLOSE_RADIUS, LINE_HALF_LENGTH = 4, 4, 10
def reduce_radial(radial, shape):
    """Average native radius into the same cells used by the DINO heatmap."""
    reduced = np.empty(shape, dtype=np.float32)
    valid = np.empty(shape, dtype=bool)
    width = radial.shape[1]
    for start in range(0, shape[0], 64):
        stop = min(start + 64, shape[0])
        padded = np.full(((stop - start) * STEP, shape[1] * STEP), np.nan, dtype=np.float32)
        padded[:, :width] = radial[start * STEP:stop * STEP]
        cells = padded.reshape(stop - start, STEP, shape[1], STEP)
        present = np.isfinite(cells)
        count = present.sum(axis=(1, 3))
        values = np.where(present, cells, 0).sum(axis=(1, 3))
        reduced[start:stop] = values / np.maximum(count, 1)
        valid[start:stop] = count >= 0.8 * STEP ** 2
    return (reduced, valid)


def local_height(radius, valid):
    """Approximate the MATLAB disk-closing envelope on the smaller score grid."""
    if not valid.any():
        raise ValueError('No valid cells for geometric reranking')
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = radius[tuple(nearest)]
    del nearest
    size = 2 * CLOSE_RADIUS + 1
    axis = np.arange(-CLOSE_RADIUS, CLOSE_RADIUS + 1)
    disk = axis[:, None] ** 2 + axis[None, :] ** 2 <= CLOSE_RADIUS ** 2
    background = ndimage.grey_closing(filled, footprint=disk)
    support = ndimage.uniform_filter(valid.astype(np.float32), size, mode='constant') >= 0.65
    sigma = (2 / STEP, 4 / STEP)
    value = ndimage.gaussian_filter(np.where(support, background, 0), sigma)
    weight = ndimage.gaussian_filter(support.astype(np.float32), sigma)
    envelope = value / np.maximum(weight, 1e-06)
    return np.where(valid & support, np.maximum(envelope - radius, 0), np.nan)


def remove_texture(height):
    """Subtract elongated components that support a near-horizontal line."""
    signal = np.nan_to_num(height)
    texture = np.zeros_like(signal)
    dx = np.arange(-LINE_HALF_LENGTH, LINE_HALF_LENGTH + 1)
    for degrees in range(-25, 26, 5):
        dy = np.rint(dx * np.tan(np.deg2rad(degrees))).astype(int)
        line = np.zeros((dy.max() - dy.min() + 1, dx.size), dtype=bool)
        line[dy - dy.min(), dx - dx.min()] = True
        response = ndimage.grey_opening(signal, footprint=line, mode='constant', cval=0)
        np.maximum(texture, response, out=texture)
    return np.where(np.isfinite(height), np.maximum(signal - texture, 0), np.nan)
