"""Frozen DINOv2 ViT-S/14 with the only retained 384→128→1 task head."""
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from . import common as b

BACKBONE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
BACKBONE_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def make_head():
    return nn.Sequential(nn.Conv2d(384, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))


def load_head(checkpoint, device):
    head = make_head().to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return head.eval()


class Dino:
    def __init__(self, weights, device="cpu", source=None):
        if b.digest(weights) != BACKBONE_SHA256:
            raise ValueError("Backbone SHA-256 does not match the selected model")
        if source:
            sys.path.insert(0, str(Path(source).resolve()))
            from dinov2.hub.backbones import dinov2_vits14
            self.model = dinov2_vits14(pretrained=False)
        else:
            self.model = torch.hub.load("facebookresearch/dinov2:" + BACKBONE_COMMIT,
                                        "dinov2_vits14", pretrained=False, trust_repo=True)
        self.device = torch.device(device)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def describe(self, tiles, low, high, size=224):
        images, masks = [], []
        grid = size // 14
        for tile in tiles:
            valid = np.isfinite(tile)
            scaled = np.where(valid, np.clip((tile-low)/(high-low), 0, 1), .5)
            images.append(np.asarray(Image.fromarray(scaled).resize((size, size), Image.Resampling.BILINEAR)))
            fraction = Image.fromarray(valid.astype(np.float32)).resize((grid, grid), Image.Resampling.BOX)
            masks.append(np.asarray(fraction).ravel() >= .8)
        tensor = torch.from_numpy(np.stack(images)).to(self.device)[:, None]
        tokens = self.model.forward_features((tensor-self.mean)/self.std)["x_norm_patchtokens"]
        return F.normalize(tokens, dim=-1), torch.from_numpy(np.stack(masks)).to(self.device)

    @torch.inference_mode()
    def task_features(self, tiles):
        images = []
        for tile in tiles:
            low, high = np.nanpercentile(tile, [1, 99])
            if not np.isfinite(low) or not high > low:
                raise ValueError("A 512px tile has no usable P1/P99 range")
            image = Image.fromarray(b.image_array(tile, low, high))
            image = np.asarray(image.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)[None]/255
            images.append(image.astype(np.float16).astype(np.float32))
        tensor = torch.from_numpy(np.stack(images)).to(self.device)
        tokens = self.model.forward_features((tensor-self.mean)/self.std)["x_norm_patchtokens"]
        features = F.normalize(tokens, dim=-1).transpose(1, 2).reshape(-1, 384, 16, 16)
        return features.to(torch.float16).to(torch.float32)


@torch.inference_mode()
def nearest_map(radial, backbone, size, two_references, batch_size=8):
    info = b.layout(radial)
    low, high = np.nanpercentile(radial[4096:5120:4, ::4], [1, 99]).astype(np.float32)
    shape = (info["eval_rows"]//4, (radial.shape[1]+3)//4)
    score, weight = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    taper = np.maximum(np.hanning(128), .15).astype(np.float32)
    taper = np.outer(taper, taper)
    for col in info["tile_columns"]:
        refs, valid = backbone.describe([b.read_tile(radial, row, col) for row in b.REFERENCES], low, high, size)
        banks = [v[m] for v, m in zip(refs, valid)]
        if any(not len(bank) for bank in banks):
            raise ValueError("Reference strip has insufficient measured support")
        bank = torch.cat(banks)
        for start in range(0, len(info["tile_rows"]), batch_size):
            rows = info["tile_rows"][start:start+batch_size]
            query, support = backbone.describe([b.read_tile(radial, row, col) for row in rows], low, high, size)
            sim = query @ bank.T
            distance = (1-sim[..., :len(banks[0])].amax(-1)).clamp_min(0)
            if two_references:
                distance = torch.minimum(distance, (1-sim[..., len(banks[0]):].amax(-1)).clamp_min(0))
            for row, value, mask in zip(rows, distance.cpu().numpy(), support.cpu().numpy()):
                grid = size//14
                heat = np.asarray(Image.fromarray((value*mask).reshape(grid, grid)).resize((128, 128), Image.Resampling.BILINEAR))
                coverage = np.asarray(Image.fromarray(mask.astype(np.float32).reshape(grid, grid)).resize((128, 128), Image.Resampling.BILINEAR))
                roi = np.s_[row//4:row//4+128, col//4:col//4+128]
                score[roi] += heat*taper
                weight[roi] += coverage*taper
    np.divide(score, weight, out=score, where=weight > 0)
    score[weight == 0] = np.nan
    return score


@torch.inference_mode()
def bubble_map(radial, backbone, head, batch_size=32):
    layout = b.layout(radial)
    positions = [(row, col) for col in layout["tile_columns"] for row in layout["tile_rows"]]
    shape = (layout["eval_rows"]//4, (radial.shape[1]+3)//4)
    score, weight = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    taper = np.maximum(np.hanning(128), .15).astype(np.float32)
    taper = np.outer(taper, taper)
    for start in range(0, len(positions), batch_size):
        positions_here = positions[start:start+batch_size]
        features = backbone.task_features([b.read_tile(radial, row, col) for row, col in positions_here])
        if backbone.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(features)
        else:
            logits = head(features)
        probabilities = F.interpolate(logits.float().sigmoid(), size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        for (row, col), heat in zip(positions_here, probabilities):
            y, x = row//4, col//4
            hh, ww = min(128, shape[0]-y), min(128, shape[1]-x)
            score[y:y+hh, x:x+ww] += heat[:hh, :ww]*taper[:hh, :ww]
            weight[y:y+hh, x:x+ww] += taper[:hh, :ww]
    domain = b.evaluation_mask(shape)
    if not (weight[domain] > 0).all():
        raise ValueError("Uncovered cells in the fixed scan layout")
    np.divide(score, weight, out=score, where=weight > 0)
    score[~domain] = np.nan
    return score
