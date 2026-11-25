"""Replay the saved out-of-fold fusion heads and audit the ten-tire evidence.

Uses only the Python standard library. DINO score maps and training are not
regenerated: the saved fold-specific scores are the model inputs for this demo.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(root, output):
    base = root / "baseline/results"
    primary = base / "40-final-model"
    extra = base / "41-final-model-supplement"
    model = read_json(primary / "最终模型.json")
    comparison = read_json(primary / "训练与对比.json")
    supplement = read_json(extra / "补扫验证.json")
    records = read_json(primary / "bubble-scores/dino_mlp-identity.json")
    original = read_json(base / "38-cloud-profiles/全量验证.json")["rows"]
    expected = read_csv(primary / "最优模型_折外预测.csv")
    assert model["tag"] == comparison["chosen"] == supplement["model"] == "dino_mlp-identity"
    main_hash = hashlib.sha256((primary / "最优模型_折外预测.csv").read_bytes()).hexdigest()
    assert main_hash == supplement["main_csv_sha256"]
    scores = {(record["tire"], rank): value
              for record in records if record["split"] == "test"
              for rank, value in zip(record["ranks"], record["scores"])}
    fold_by_tire = {tire: fold for fold in model["fold_heads"] for tire in fold["test"]}
    reference = {(row["胎号"], int(row["原候选名次"])): row for row in expected}
    assert len(scores) == len(original) == len(reference) == 200
    predictions = []
    for tire in sorted(fold_by_tire):
        fold = fold_by_tire[tire]
        assert tire not in fold["train"]
        head = fold["head"]
        rows = sorted((r for r in original if r["tire"] == tire), key=lambda r: r["rank"])
        assert len(rows) == 20
        order = sorted(rows, key=lambda r: (-scores[tire, r["rank"]], r["rank"]))
        bubble_order = {r["rank"]: i for i, r in enumerate(order, 1)}
        local = []
        for row in rows:
            key = tire, row["rank"]
            u = scores[key]
            features = [row["p95_mm"], row["max_step_p99_mm"], math.log1p(row["missing_pct"]),
                        -math.log10(max(1 - u, 1e-6))]
            logit = head["intercept"] + sum(
                weight * (value - mean) / scale for weight, value, mean, scale
                in zip(head["coefficients"], features, head["mean"], head["scale"]))
            probability = 1 / (1 + math.exp(-logit))
            prediction = row["prediction"]
            if prediction == "待复核" and probability <= 0.2 and bubble_order[row["rank"]] <= 5:
                prediction = "疑似气泡"
            fusion = 1 / (10 + row["rank"]) + 1 / (10 + bubble_order[row["rank"]])
            saved = reference[key]
            assert prediction == saved["融合复判"]
            assert abs(probability - float(saved["附着物模型分值"])) < 1e-9
            local.append(dict(tire=tire, original_rank=row["rank"], prediction=prediction,
                              p_attachment=probability, fusion=fusion, bubble_rank=None,
                              truth=saved["已核实标注"]))
        for rank, row in enumerate(sorted((r for r in local if r["prediction"] == "疑似气泡"),
                                          key=lambda r: (-r["fusion"], r["original_rank"])), 1):
            row["bubble_rank"] = rank
        for row in local:
            saved = reference[tire, row["original_rank"]]
            assert (str(row["bubble_rank"]) if row["bubble_rank"] else "") == saved["气泡筛查名次"]
        predictions.extend(local)

    first = {tire: min((r["bubble_rank"] for r in predictions if r["tire"] == tire
                      and r["truth"] == "气泡" and r["bubble_rank"] is not None), default=None)
             for tire in fold_by_tire}
    attachments = sum(r["truth"] == "附着物" and r["prediction"] == "疑似附着物" for r in predictions)
    assert attachments == 6 and sum(v is not None for v in first.values()) == 9
    assert first == comparison["ranking"][0]["bubble_first_ranks"]
    detail_csv = read_csv(extra / "补扫240个候选.csv")
    assert len(detail_csv) == len(supplement["details"]) == 240
    extra_counts = Counter(r["点云判断"] for r in detail_csv)
    assert extra_counts == Counter(r["prediction"] for r in supplement["details"])
    by_tire = []
    times = []
    for sample in supplement["samples"]:
        tire = sample["tire"]
        rows = [r for r in detail_csv if r["胎号"] == tire]
        assert len(rows) == 24
        assert sample["counts"] == dict(Counter(r["点云判断"] for r in rows))
        assert sample["main_bubble_rank"] == first[tire]
        seconds = sum(sample["seconds"].values())
        times.append(seconds)
        by_tire.append(dict(tire=tire, main_rank=first[tire],
                            supplement_hit=sample["supplement_candidate_hit"],
                            supplement_bubble_rank=sample["supplement_bubble_hit"],
                            supplement_fused_rank=sample["supplement_reranked_hit"],
                            new_bubbles=sample["counts"].get("疑似气泡", 0), seconds=seconds,
                            confirmed_attachment_hits=sample["main_known_attachments"]))
    combined_hits = sum(r["main_rank"] is not None or r["supplement_fused_rank"] is not None for r in by_tire)
    hit_rows = [r for r in supplement["details"] if r["bubble_hit"]]
    assert combined_hits == 10 and len(hit_rows) == 1
    found = hit_rows[0]
    assert found["tire"] == "G6B1F56073" and found["order"] == 21 and found["fused_bubble_rank"] == 11
    confirmed_review = read_csv(base / "39-human-review/人工确认清单.csv")
    assert sum(r["人工确认"] == "附着物" for r in confirmed_review) == 5
    summary = dict(model=model["tag"], main_candidates=200, supplement_candidates=240,
                   main_hits=9, combined_hits=combined_hits, known_bubbles=10,
                   confirmed_attachment_hits=attachments, main_counts=dict(Counter(r["prediction"] for r in predictions)),
                   supplement_counts=dict(extra_counts), mean_seconds=sum(times) / len(times),
                   min_seconds=min(times), max_seconds=max(times), recovered=found, tires=by_tire,
                   final_metrics=comparison["ranking"][0],
                   verified_predictions=200, source_sha256=main_hash,
                   scope="Saved fold-head replay and cross-check; no retraining or full DINO inference.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "核对结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "折外融合回放.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print("已实际重算 200 个主筛候选的折外融合分值与分类，与原记录完全一致。")
    print(f"主筛气泡: 9/10; 主筛+补扫气泡: {combined_hits}/10; 已确认附着物: {attachments}/6")
    print(f"补扫新增候选 240 个，疑似气泡 {extra_counts['疑似气泡']} 个，待复核 {extra_counts['待复核']} 个。")
    print(f"原记录的补扫增量耗时: {sum(times) / len(times):.2f} 秒/胎。")
    print("56073: 主筛未覆盖的已标气泡，经补扫成功找回。")
    print("展示范围: 保存分数上的融合推理回放；未重新训练或重新运行整胎 DINO。")
    print(f"输出: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data/replay")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/replay")
    args = parser.parse_args()
    main(args.root, args.output)
