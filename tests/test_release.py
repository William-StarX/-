import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tire_anomaly.candidates import add_significance, distance, select_supplement
from tire_anomaly.common import digest
from tire_anomaly.fusion import classify
from tire_anomaly.geometry import assess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/replay/baseline/results"


class FinalModelRegression(unittest.TestCase):
    def test_all_200_fusion_outputs_match_original(self):
        rows = json.loads((DATA/"38-cloud-profiles/全量验证.json").read_text(encoding="utf-8"))["rows"]
        records = json.loads((DATA/"40-final-model/bubble-scores/dino_mlp-identity.json").read_text(encoding="utf-8"))
        model = json.loads((ROOT/"models/final_model.json").read_text(encoding="utf-8"))
        values = {(r["tire"], rank): value for r in records if r["split"] == "test" for rank, value in zip(r["ranks"], r["scores"])}
        with (DATA/"40-final-model/最优模型_折外预测.csv").open(encoding="utf-8-sig") as stream:
            saved = {(r["胎号"], int(r["原候选名次"])): r for r in csv.DictReader(stream)}
        checked = 0
        for fold in model["fold_heads"]:
            self.assertFalse(set(fold["train"]) & set(fold["test"]))
            for tire in fold["test"]:
                local = [r for r in rows if r["tire"] == tire]
                predicted = classify(local, [values[tire, r["rank"]] for r in local], fold["head"])
                for row in predicted:
                    expected = saved[tire, row["rank"]]
                    self.assertEqual(row["prediction"], expected["融合复判"])
                    self.assertAlmostEqual(row["attachment_score"], float(expected["附着物模型分值"]), places=10)
                    checked += 1
        self.assertEqual(checked, 200)

    def test_portable_replay_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"results"
            subprocess.run([sys.executable, str(ROOT/"scripts/replay_results.py"), "--output", str(output)], cwd=temporary, check=True, capture_output=True)
            result = json.loads((output/"核对结果.json").read_text(encoding="utf-8"))
            self.assertEqual((result["main_hits"], result["combined_hits"], result["confirmed_attachment_hits"]), (9, 10, 6))

    def test_selected_checkpoint_hashes(self):
        model = json.loads((ROOT/"models/final_model.json").read_text(encoding="utf-8"))
        self.assertEqual(model["tag"], "dino_mlp-identity")
        self.assertEqual(len(model["checkpoints"]), 5)
        for checkpoint in model["checkpoints"]:
            self.assertEqual(hashlib.sha256((ROOT/checkpoint["path"]).read_bytes()).hexdigest(), checkpoint["sha256"])
            self.assertEqual(digest(ROOT/checkpoint["path"]), checkpoint["sha256"])

    def test_no_edges_across_missing_raw_points(self):
        block = np.full((25, 25), np.nan)
        block[:, :12], block[:, 13:] = .5, 3.
        row = dict(tire="test", rank=1, raw_xy=[0, 0], flat_xy=[0, 0], selected_xyxy=[0, 0, 24, 24],
                   automatic=dict(prediction="疑似附着物", p95_mm=3., missing_pct=4.), original=dict(prediction="待复核"))
        result = assess(row, block)
        self.assertEqual(result["max_step_p99_mm"], 0.)
        self.assertEqual(result["prediction"], "待复核")

    def test_supplement_respects_budget_and_main_exclusion(self):
        score = np.zeros((250, 250), dtype=np.float32)
        peaks = []
        for row in range(15, 240, 30):
            for col in range(15, 240, 30):
                score[row, col] = 1+len(peaks)/100
                peaks.append(dict(x=col*4+1.5, y=row*4+1.5))
        main = peaks[:20]
        extra = select_supplement(score, main)
        self.assertEqual(len(extra), 24)
        for i, p in enumerate(extra):
            self.assertTrue(all(distance(p, q) > 96 for q in main+extra[:i]))

    def test_local_significance_is_background_relative(self):
        score = np.random.default_rng(3).normal(4., .6, (90, 90))
        peak = dict(x=47*4+1.5, y=42*4+1.5, band=1, local_rank=1, score=float(score[42, 47]))
        first = add_significance(score, peak)["local_significance"]
        shifted = add_significance(2*score+10, dict(peak, score=2*peak["score"]+10))["local_significance"]
        self.assertAlmostEqual(first, shifted, places=10)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Install the model extra to check the MLP")
    def test_only_selected_mlp_shape_and_weak_loss(self):
        import torch
        from tire_anomaly.model import make_head
        from tire_anomaly.training import loss_value
        head = make_head()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 49409)
        model = json.loads((ROOT/"models/final_model.json").read_text(encoding="utf-8"))
        for checkpoint in model["checkpoints"]:
            head.load_state_dict(torch.load(ROOT/checkpoint["path"], map_location="cpu", weights_only=True))
            self.assertTrue(torch.isfinite(head(torch.zeros(1, 384, 16, 16))).all())
        logits = head(torch.zeros(2, 384, 16, 16))
        self.assertEqual(tuple(logits.shape), (2, 1, 16, 16))
        target = torch.zeros(2, 1, 16, 16)
        target[0, :, 5:9, 5:9] = 1
        loss = loss_value(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))


if __name__ == "__main__":
    unittest.main()
