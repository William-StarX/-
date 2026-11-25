"""Prepare scanner CSVs without consuming any defect annotations."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from . import flatten as flat
from .common import digest
def read_scan(source, shape, scratch, max_x):
    """Keep the original column indices and stream the organized scan to disk."""
    x = np.lib.format.open_memmap(scratch / 'x.npy', mode='w+', dtype=np.float32, shape=shape)
    z = np.lib.format.open_memmap(scratch / 'z.npy', mode='w+', dtype=np.float32, shape=shape)
    valid = np.lib.format.open_memmap(scratch / 'valid.npy', mode='w+', dtype=bool, shape=shape)
    y = np.empty(shape[0], dtype=np.float64)
    offsets = np.empty(shape[0], dtype=np.int64)
    acquired = np.empty((shape[0], (shape[1] + 7) // 8), dtype=np.uint8)
    width = 3 + 2 * shape[1]
    count = 0
    with source.open('rb') as stream:
        header = stream.readline().decode('utf-8-sig').strip().split(',')
        if len(header) != width or header[3:5] != ['X0', 'Y0']:
            raise ValueError(f'CSV fields do not match the organized BMP: {source}')
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            if count == shape[0]:
                raise ValueError(f'CSV has more rows than the BMP: {source}')
            values = np.fromstring(line, sep=',')
            if values.size != width:
                fields = line.rstrip().split(b',')
                if len(fields) != width:
                    raise ValueError(f'CSV row {count + 1} has {len(fields)} fields, expected {width}')
                values = np.array([float(field) if field.strip() else np.nan for field in fields])
            y[count] = values[2]
            xx, zz = (values[3::2], values[4::2])
            present = np.isfinite(xx) & np.isfinite(zz) & ((xx != 0) | (zz != 0))
            x[count], z[count] = (xx, zz)
            acquired[count] = np.packbits(present)
            valid[count] = present & (xx <= max_x) & np.isfinite(y[count])
            offsets[count] = offset
            count += 1
    if count != shape[0]:
        raise ValueError(f'CSV has {count} rows but BMP has {shape[0]}: {source}')
    return (x, y, z, valid, acquired, offsets)


def repair_to_disk(radial, destination, cfg):
    fixed = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32, shape=radial.shape)
    halo = cfg.whittakerSupportRows // 2
    count = 0
    for start in range(0, len(radial), cfg.whittakerBlockRows):
        end = min(start + cfg.whittakerBlockRows, len(radial))
        left = max(0, start - halo)
        window = radial[left:min(end + halo, len(radial))]
        repaired, _ = flat.repair_whittaker(window, cfg)
        center = repaired[start - left:end - left]
        count += int(np.count_nonzero(np.isfinite(center) & (center != radial[start:end])))
        fixed[start:end] = center
    fixed.flush()
    del fixed
    return count

def prepare_scan(source, image, destination):
    source, destination = Path(source).resolve(), Path(destination)
    with Image.open(image) as preview:
        shape = preview.height, preview.width
    cfg = flat.Config()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix='prepare-', ignore_cleanup_errors=True) as temporary:
        scratch = Path(temporary)
        x, y, z, valid, acquired, offsets = read_scan(source, shape, scratch, cfg.maxValidX)
        cx, cz, radius, _ = flat.fit_circle(x, z, valid, cfg)
        radial, grids, _ = flat.unwrap_to_grid(x, y, z, valid, cx, cz, radius, cfg, output_path=scratch/'normal.npy')
        repair_to_disk(radial, destination/'radial.npy', cfg)
        del radial
        def smooth_only(profile, config):
            return (flat.normalized_conv(profile) if np.isfinite(profile).sum() >= 3 else profile), 0
        with patch.object(flat, 'clean_profile', smooth_only):
            alternate, alternate_grids, _ = flat.unwrap_to_grid(x, y, z, valid, cx, cz, radius, cfg, output_path=scratch/'tail.npy')
        np.testing.assert_array_equal(grids['thetaGrid'], alternate_grids['thetaGrid'])
        np.testing.assert_array_equal(grids['retainedRows'], alternate_grids['retainedRows'])
        repair_to_disk(alternate, destination/'tail_radial.npy', cfg)
        np.savez_compressed(destination/'coordinates.npz', source=str(source), source_sha256=digest(source), source_shape=shape,
                            source_row_offsets=offsets, source_acquired=acquired,
                            retained_rows=grids['retainedRows']-1, theta=grids['thetaGrid'], y=grids['yGrid'],
                            cx=cx, cz=cz, radius=radius)
        del alternate, x, z, valid
    return destination
