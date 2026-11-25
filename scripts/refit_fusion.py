"""Refit the single final fusion head from the included confirmed measurements."""
import csv
import json
from pathlib import Path

from tire_anomaly.fusion import fit_head

root = Path(__file__).resolve().parents[1]
data = root/"data/replay/baseline/results"
rows = json.loads((data/"38-cloud-profiles/全量验证.json").read_text(encoding="utf-8"))["rows"]
scores = json.loads((data/"40-final-model/bubble-scores/dino_mlp-identity.json").read_text(encoding="utf-8"))
percentiles = {(r["tire"], rank): value for r in scores if r["split"] == "test" for rank, value in zip(r["ranks"], r["scores"])}
with (data/"40-final-model/最优模型_折外预测.csv").open(encoding="utf-8-sig") as stream:
    truth = {(r["胎号"], int(r["原候选名次"])): r["已核实标注"] for r in csv.DictReader(stream)}
selected, units, values = [], [], []
for row in rows:
    key = row["tire"], row["rank"]
    if truth[key] not in ("气泡", "附着物"):
        continue
    selected.append(row)
    values.append(percentiles[key])
    units.append(("bubble", row["tire"]) if truth[key] == "气泡" else ("attachment", str(key)))
head = fit_head(selected, values, units)
output = root/"outputs/refit_final_head.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(head, ensure_ascii=False, indent=2), encoding="utf-8")
print(output)
