"""The selected standardized logistic head and conservative candidate relabeling."""
import numpy as np
from scipy.special import expit

BUBBLE, ATTACHED, REVIEW = "疑似气泡", "疑似附着物", "待复核"


def feature_matrix(rows, percentiles):
    return np.asarray([[r["p95_mm"], r["max_step_p99_mm"], np.log1p(r["missing_pct"]) if r["missing_pct"] is not None else np.nan,
                        -np.log10(max(1-u, 1e-6))] for r, u in zip(rows, percentiles)], dtype=float)


def classify(rows, percentiles, head):
    if len(rows) != len(percentiles):
        raise ValueError("One percentile is required for each candidate")
    values = feature_matrix(rows, percentiles)
    valid = np.isfinite(values).all(axis=1)
    probabilities = np.full(len(rows), np.nan)
    scaled = (values[valid]-head["mean"])/head["scale"]
    probabilities[valid] = expit(scaled @ np.asarray(head["coefficients"]) + head["intercept"])
    order = sorted(range(len(rows)), key=lambda i: (-percentiles[i], rows[i]["rank"]))
    branch = {index: rank for rank, index in enumerate(order, 1)}
    output = []
    for i, row in enumerate(rows):
        prediction = row["prediction"]
        if prediction == REVIEW and probabilities[i] <= .2 and branch[i] <= 5:
            prediction = BUBBLE
        output.append(dict(row, geometry_prediction=row["prediction"], prediction=prediction,
                           bubble_percentile=float(percentiles[i]),
                           attachment_score=float(probabilities[i]) if valid[i] else None,
                           bubble_priority_score=1/(10+row["rank"])+1/(10+branch[i])))
    return output


def fit_head(rows, percentiles, units):
    """Fit the same class/defect-balanced final fusion head on confirmed units."""
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    repeats = Counter(map(tuple, units))
    classes = Counter(unit[0] for unit in repeats)
    weights = np.asarray([len(rows)/(2*repeats[tuple(unit)]*classes[unit[0]]) for unit in units])
    values = feature_matrix(rows, percentiles)
    targets = np.asarray([unit[0] == "attachment" for unit in units], dtype=int)
    scaler = StandardScaler().fit(values, sample_weight=weights)
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
    model.fit(scaler.transform(values), targets, sample_weight=weights)
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                C=1., samples=len(rows), targets=dict(classes))
