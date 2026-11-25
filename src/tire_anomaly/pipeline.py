"""Inference for the single selected main-screening and supplement pipeline."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from . import common as b
from .candidates import r3, supplement, rerank, select_supplement
from .frozen_geometry import make_score
from .fusion import classify, REVIEW
from .geometry import classify as geometry_classify, get_patch, assess
from .texture import reduce_radial, local_height, remove_texture


def profile(source, meta, peak, sample):
    region = geometry_classify(source, meta, peak)
    region["tire"] = sample
    if region["selected_xyxy"]:
        result = assess(region, get_patch(region, source, meta)[2])
    else:
        result = dict(tire=sample, rank=peak["rank"], flat_xy=region["flat_xy"], raw_xy=region["raw_xy"],
                      p95_mm=None, missing_pct=None, max_step_p99_mm=None, prediction=REVIEW)
    return result


def infer(prepared, model_dir, fold, output, device="cpu", source=None, backbone_source=None,
          backbone_weights=None, fusion_head="final", batch_size=16):
    from .model import Dino, load_head, nearest_map, bubble_map
    import torch
    prepared, model_dir, output = Path(prepared), Path(model_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = json.loads((model_dir/"final_model.json").read_text(encoding="utf-8"))
    checkpoint = next(item for item in model["checkpoints"] if item["fold"] == fold)
    weights = model_dir.parent/checkpoint["path"]
    if b.digest(weights) != checkpoint["sha256"]:
        raise ValueError("Selected MLP checkpoint SHA-256 mismatch")
    with np.load(prepared/"coordinates.npz", allow_pickle=False) as stored:
        meta = {key: stored[key] for key in stored.files}
    source = Path(source or str(meta["source"]))
    if not source.is_file():
        raise FileNotFoundError("Supply --raw-csv after moving prepared data to another machine")
    sample = prepared.name
    radial = np.load(prepared/"radial.npy", mmap_mode="r")
    tail_radial = np.load(prepared/"tail_radial.npy", mmap_mode="r")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = Dino(backbone_weights or model_dir/"backbone/dinov2_vits14_pretrain.pth", device, backbone_source)
    head = load_head(weights, backbone.device)
    tick = time.perf_counter()
    print("Generating main candidate evidence", flush=True)
    main_map = nearest_map(radial, backbone, 224, True, batch_size)
    support_map = nearest_map(radial, backbone, 448, False, batch_size)
    domain = np.isfinite(main_map)
    radius, valid = reduce_radial(radial, main_map.shape)
    texture = remove_texture(local_height(radius, valid))
    radius, valid = reduce_radial(tail_radial, main_map.shape)
    tail_texture = remove_texture(local_height(radius, valid))
    cap = max(float(np.nanpercentile(texture[4096//4:5120//4], 99.9)), 1e-6)
    support_map *= 1+.5*np.clip(np.nan_to_num(texture)/cap, 0, 1)
    support_map[~domain] = np.nan
    primary = r3(b.candidate_peaks(main_map), b.candidate_peaks(np.where(domain, texture, np.nan)))
    primary = supplement(primary, b.candidate_peaks(np.where(domain, tail_texture, np.nan)))
    primary = rerank(primary, b.candidate_peaks(support_map))
    if len(primary) != 20:
        raise ValueError("Scan does not provide the expected 20 main candidates")
    print("Scoring bubbles and raw-cloud profiles", flush=True)
    probabilities = bubble_map(radial, backbone, head, batch_size)
    main_rows = [profile(source, meta, peak, sample) for peak in primary]
    percentiles = b.bubble_ranks(probabilities, [(p["x"], p["y"]) for p in primary])
    selected_head = model["final_head"] if fusion_head == "final" else next(h["head"] for h in model["fold_heads"] if h["fold"] == fold)
    main = classify(main_rows, percentiles, selected_head)
    print("Generating frozen geometry and 24 supplement candidates", flush=True)
    geometry_score, geometry_info = make_score(source, meta, main_map.shape)
    additional = select_supplement(geometry_score, primary)
    extra = [profile(source, meta, p, sample) for p in additional]
    extra_percentiles = b.bubble_ranks(probabilities, [(p["x"], p["y"]) for p in additional])
    order = sorted(range(len(extra)), key=lambda i: (-extra_percentiles[i], i))
    branch = {i: rank for rank, i in enumerate(order, 1)}
    for i, row in enumerate(extra):
        row.update(bubble_percentile=extra_percentiles[i],
                   bubble_priority_score=1/(10+row["rank"])+1/(10+branch[i]),
                   local_significance=additional[i]["local_significance"])
    result = dict(sample=sample, method="dino_mlp-identity", fold=fold, fusion_head=fusion_head,
                  main=main, supplement=extra, runtime_seconds=time.perf_counter()-tick,
                  geometry=geometry_info, checkpoint_sha256=checkpoint["sha256"])
    b.save_json(output/"predictions.json", result)
    with (output/"candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["stage", "candidate_id", "raw_col", "raw_row", "p95_mm", "step_p99_mm", "missing_pct", "prediction", "bubble_percentile"]
        writer = csv.writer(stream)
        writer.writerow(columns)
        for stage, rows in (("main", main), ("supplement", extra)):
            for row in rows:
                writer.writerow([stage, row["rank"], *row["raw_xy"], row["p95_mm"], row["max_step_p99_mm"], row["missing_pct"], row["prediction"], row["bubble_percentile"]])
    print("Saved " + str(output/"predictions.json"), flush=True)
    return result
