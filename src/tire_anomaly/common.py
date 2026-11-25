"""Fixed geometry and scoring settings for the one selected pipeline."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

TILE, STRIDE, INPUT, PATCH, STEP = 512, 256, 224, 14, 4
MIN_VALID_FRACTION, NMS_RADIUS, MAX_CANDIDATES = 0.8, 96, 20
REFERENCES, GUARD = (4096, 4608), 64
def tile_starts(length):
    end = (length + STEP - 1) // STEP * STEP - TILE
    if end < 0:
        raise ValueError('The scan region is smaller than one tile')
    starts = list(range(0, end + 1, STRIDE))
    if starts[-1] != end:
        starts.append(end)
    return starts


def read_tile(image, row, col):
    tile = image[row:row + TILE, col:col + TILE]
    if tile.shape == (TILE, TILE):
        return tile
    padded = np.full((TILE, TILE), np.nan, dtype=np.float32)
    padded[:tile.shape[0], :tile.shape[1]] = tile
    return padded


def candidate_peaks(score):
    remaining = np.where(np.isfinite(score), score, -np.inf)
    radius = NMS_RADIUS // STEP
    candidates = []
    for rank in range(1, MAX_CANDIDATES + 1):
        row, col = np.unravel_index(remaining.argmax(), remaining.shape)
        value = remaining[row, col]
        if not np.isfinite(value):
            break
        candidates.append(dict(rank=rank, x=float((col + 0.5) * STEP - 0.5), y=float((row + 0.5) * STEP - 0.5), score=float(value)))
        remaining[max(0, row - radius):row + radius + 1, max(0, col - radius):col + radius + 1] = -np.inf
    return candidates


def label_mask(polygon, score_shape):
    mask = Image.new('1', (score_shape[1], score_shape[0]))
    origin = (STEP - 1) / 2
    points = [((x - origin) / STEP, (y - origin) / STEP) for x, y in polygon]
    ImageDraw.Draw(mask).polygon(points, fill=1)
    return np.asarray(mask)
def digest(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
        return value.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def layout(radial):
    end = len(radial) // STEP * STEP
    intervals = ((0, REFERENCES[0] - GUARD), (REFERENCES[-1] + TILE + GUARD, end))
    rows = [left + r for left, right in intervals for r in tile_starts(right - left)]
    columns = tile_starts(radial.shape[1])
    return dict(eval_rows=end, evaluation_intervals=intervals, reference_rows=REFERENCES, tile_rows=rows, tile_columns=columns, tiles=len(rows) * len(columns), reference_shots=2, map_shape=list(radial.shape))


def evaluation_mask(shape):
    mask = np.ones(shape, dtype=bool)
    mask[(REFERENCES[0] - GUARD) // STEP:(REFERENCES[-1] + TILE + GUARD) // STEP] = False
    return mask
def image_array(radial, low, high):
    finite = np.isfinite(radial)
    scaled = np.where(finite, np.clip((radial - low) / (high - low), 0, 1), 0.5)
    return np.rint(scaled * 255).astype(np.uint8)
def bubble_ranks(score, points):
    """Fraction of the scored tire at or below the score at each candidate center."""
    score = np.asarray(score)
    domain = np.sort(score[np.isfinite(score)])
    values = [float(score[int(y) // 4, int(x) // 4]) for x, y in points]
    return [float(np.searchsorted(domain, value, side='right') / len(domain)) if np.isfinite(value) and len(domain) else 0.0 for value in values]
