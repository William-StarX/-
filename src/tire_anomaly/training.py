"""Train only the retained DINO MLP on existing frozen feature arrays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import make_head


def loss_value(logits, targets):
    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.adaptive_avg_pool2d(targets, logits.shape[-2:])
    weight = .1+.9*targets
    bce = -(targets*F.logsigmoid(logits.float())+.1*(1-targets)*F.logsigmoid(-logits.float())).sum()/weight.sum()
    probability = logits.float().sigmoid()
    positive = targets.sum((1, 2, 3)) > 0
    dice = (2*(probability*targets).sum((1, 2, 3))+1)/(probability.sum((1, 2, 3))+targets.sum((1, 2, 3))+1)
    return bce+((1-dice)*positive).sum()/positive.sum().clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Float16/32 N x 384 x 16 x 16 normalized DINO features")
    parser.add_argument("--masks", type=Path, required=True, help="N x 1 x H x W targets in [0,1]")
    parser.add_argument("--records", type=Path, required=True, help="JSON array containing tire_id for every row")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--protocol", type=Path, default=Path("models/training_protocol.json"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    fold = next(row for row in protocol["folds"] if row["fold"] == args.fold)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    features, masks = np.load(args.features, mmap_mode="r"), np.load(args.masks, mmap_mode="r")
    if features.shape[0] != len(records) or masks.shape[0] != len(records):
        raise ValueError("Features, masks and records must have identical row counts")
    if any(row.get("transform", "identity") != "identity" for row in records):
        raise ValueError("This release trains only original identity crops")
    train = np.asarray([i for i, row in enumerate(records) if row["tire_id"] in fold["train"]])
    validation = np.asarray([i for i, row in enumerate(records) if row["tire_id"] in fold["validation"]])
    if not len(train) or not len(validation):
        raise ValueError("Training and validation tire groups must both be present")
    torch.manual_seed(protocol["seed"]+args.fold)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"]+args.fold)
    rng = np.random.default_rng(protocol["seed"]+args.fold)
    device = torch.device(args.device)
    model = make_head().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fold["learning_rate"], weight_decay=.0001)
    best, stale, history = float("inf"), 0, []
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 21):
        model.train()
        order = np.tile(train, 4)
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            indices = order[start:start+32]
            x = torch.from_numpy(np.array(features[indices], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.array(masks[indices], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_value(model(x), y)
            else:
                loss = loss_value(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        totals = np.zeros(4)
        with torch.inference_mode():
            for start in range(0, len(validation), 32):
                indices = validation[start:start+32]
                x = torch.from_numpy(np.array(features[indices], dtype=np.float32)).to(device)
                y = torch.from_numpy(np.array(masks[indices], dtype=np.float32)).to(device)
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x)
                else:
                    logits = model(x)
                if y.shape[-2:] != logits.shape[-2:]:
                    y = F.adaptive_avg_pool2d(y, logits.shape[-2:])
                p = logits.float().sigmoid()
                positive = y.sum((1, 2, 3)) > 0
                dice = (2*(p*y).sum((1, 2, 3))+1)/(p.sum((1, 2, 3))+y.sum((1, 2, 3))+1)
                totals += [float(-(y*F.logsigmoid(logits.float())+.1*(1-y)*F.logsigmoid(-logits.float())).sum()),
                           float((.1+.9*y).sum()), float(((1-dice)*positive).sum()), int(positive.sum())]
        value = totals[0]/totals[1]+totals[2]/max(totals[3], 1)
        history.append(dict(epoch=epoch, validation_loss=float(value)))
        if value < best-1e-6:
            best, stale = value, 0
            torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, args.output/"best.pt")
        else:
            stale += 1
        print(epoch, value, flush=True)
        if stale >= 5:
            break
    (args.output/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
