"""RRF main candidates and fixed-budget band-local supplementation."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import common as b
ALTERNATE, K = ('secondary', 'primary'), 60
INNER, OUTER, MAD_SCALE, SCALE_FLOOR = 48, 96, 1.4826, 1e-6
def distance(p, q):
    return max(abs(p['x'] - q['x']), abs(p['y'] - q['y']))


def take(selected, queue, source):
    for candidate in queue:
        if all((distance(candidate, p) > b.NMS_RADIUS for p in selected)):
            selected.append(dict(candidate, source=source, source_rank=candidate['rank'], rank=len(selected) + 1))
            return True
    return False


def merge(primary, geometry):
    selected = [dict(p, source='primary', source_rank=p['rank']) for p in primary[:5]]
    queues = {'primary': iter(primary[5:]), 'secondary': iter(geometry)}
    while len(selected) < b.MAX_CANDIDATES:
        before = len(selected)
        for source in ALTERNATE:
            take(selected, queues[source], source)
            if len(selected) == b.MAX_CANDIDATES:
                return selected
        if len(selected) == before:
            break
    return selected
def r3(primary, geometry):
    tail = sorted(primary[5:], key=lambda p: not any((distance(p, q) <= b.NMS_RADIUS for q in geometry)))
    return merge(primary[:5] + tail, geometry)
def supplement(original, geometry):
    prefix = original[:19]
    for candidate in geometry:
        if all((distance(candidate, p) > b.NMS_RADIUS for p in prefix)):
            return prefix + [dict(candidate, source='tail_geometry', source_rank=candidate['rank'], rank=20)]
    return original
def associate(primary, support):
    """Maximize the number of nearby pairs, then minimize total distance."""
    distances = np.array([[distance(a, c) for c in support] for a in primary])
    allowed = distances <= b.NMS_RADIUS
    penalty = (len(primary) + 1) * (b.NMS_RADIUS + 1)
    rows, cols = linear_sum_assignment(np.where(allowed, distances, penalty))
    return {r: c for r, c in zip(rows, cols) if allowed[r, c]}


def rerank(primary, support):
    matches = associate(primary, support)
    scores = []
    for index, peak in enumerate(primary):
        partner = support[matches[index]] if index in matches else None
        score = 1 / (K + peak['rank'])
        if partner is not None:
            score += 1 / (K + partner['rank'])
        scores.append(dict(peak, main_rank=peak['rank'], rrf_score=score, support_rank=partner['rank'] if partner is not None else None, support_distance_px=distance(peak, partner) if partner is not None else None))
    ordered = sorted(scores, key=lambda peak: (-peak['rrf_score'], peak['main_rank']))
    return [dict(peak, rank=index) for index, peak in enumerate(ordered, 1)]
def band_peaks(scores, band, start, radius, limit):
    remaining = np.where(np.isfinite(scores), scores, -np.inf)
    peaks = []
    for rank in range(1, limit + 1):
        row, col = np.unravel_index(remaining.argmax(), remaining.shape)
        value = remaining[row, col]
        if value <= 0:
            break
        peaks.append(dict(band=band, local_rank=rank, x=float((start + col + 0.5) * b.STEP - 0.5), y=float((row + 0.5) * b.STEP - 0.5), score=float(value)))
        remaining[max(0, row - radius):row + radius + 1, max(0, col - radius):col + radius + 1] = -np.inf
    return peaks

def grid_indices(peak):
    return int(peak['y']) // b.STEP, int(peak['x']) // b.STEP
def add_significance(score, peak):
    row, col = grid_indices(peak)
    width = 128 // b.STEP
    outer = OUTER // b.STEP
    inner = INNER // b.STEP
    y0, y1 = (max(0, row - outer), min(score.shape[0], row + outer + 1))
    x0 = max(peak['band'] * width, col - outer)
    x1 = min((peak['band'] + 1) * width, score.shape[1], col + outer + 1)
    local = score[y0:y1, x0:x1]
    ring = np.maximum(np.abs(np.arange(y0, y1) - row)[:, None], np.abs(np.arange(x0, x1) - col)) > inner
    background = local[ring & np.isfinite(local)].astype(np.float64)
    if not background.size:
        return dict(peak, background_cells=0, background_median=None, background_mad=None, local_significance=0.0)
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    significance = (peak['score'] - median) / max(MAD_SCALE * mad, SCALE_FLOOR)
    return dict(peak, background_cells=int(background.size), background_median=median, background_mad=mad, local_significance=significance)


def priority(peak, occupancy):
    band = peak['band']
    return (occupancy[band], -peak['local_significance'], -peak['score'], band, peak['local_rank'])

def select_supplement(score, original, budget=24):
    width = 128 // b.STEP
    occupancy = Counter(int(p['x'] // 128) for p in original)
    proposals = [add_significance(score, p)
                 for band, start in enumerate(range(0, score.shape[1], width))
                 for p in band_peaks(score[:, start:start+width], band, start, 48//b.STEP, 40)
                 if all(distance(p, q) > b.NMS_RADIUS for q in original)]
    selected = []
    for peak in sorted(proposals, key=lambda p: priority(p, occupancy)):
        if all(distance(peak, q) > b.NMS_RADIUS for q in selected):
            selected.append(dict(peak, rank=len(selected)+1))
        if len(selected) == budget:
            break
    return selected
