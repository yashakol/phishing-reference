"""Integrate Astra's text model with frozen OOF-selected ensemble settings."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from phishing_url.__main__ import predict_bundle
from phishing_url.data import load_train, load_test, write_submission
from phishing_url.evaluation import measure
from phishing_url.rich_text import URLModel
from phishing_url.models import fit_boost, boost_probability
from phishing_url.features import make_features


def main():
    start = time.perf_counter()
    for name in ["artifacts/v2", "reports/v2", "submissions"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    urls, y, audit = load_train(ROOT / "data/train.csv")
    test_urls = load_test(ROOT / "data/test.csv")
    split = np.load(ROOT / "reports/split_indices.npz")
    dev, holdout = split["development"], split["holdout"]
    configurations = json.loads((ROOT / "reports/v2/rich_blend_search.json").read_text())
    selected = configurations[0]
    assert selected["old_text_weight"] == 0 and selected["rich_weight"] == .5 and selected["boost_weight"] == .5
    # Freeze the OOF-derived configuration before touching holdout predictions.
    (ROOT / "artifacts/v2/config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("Frozen ensemble settings:", selected, flush=True)
    sys.path.insert(0, str(ROOT / "experiments/astra"))
    development_rich = joblib.load(ROOT / "experiments/astra/development_model.joblib")
    p_rich = development_rich.predict_proba([urls[int(i)] for i in holdout])
    frame = make_features(urls)
    development_boost = fit_boost(frame.iloc[dev], y[dev], threads=2)
    p_boost = boost_probability(development_boost, frame.iloc[holdout])
    p = .5 * p_rich + .5 * p_boost
    scores = measure(y[holdout], p, selected["threshold"])
    results = {"selected_on": "development OOF only", "development_oof": selected,
               "previously_used_holdout": scores,
               "holdout_note": "Same holdout as v1, evaluated with frozen v2 parameters; not a new independent test.",
               "model": "Astra rich component TF-IDF LogisticRegression C16 + original CatBoost, equal blend"}
    print("Measured v2 holdout:", scores, flush=True)
    (ROOT / "reports/v2/url_model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    del development_rich, development_boost
    import gc
    gc.collect()
    print("Fitting rich text on all clean training URLs", flush=True)
    rich = URLModel(representation="rich", c=16, nb=False).fit(urls, y)
    original = joblib.load(ROOT / "artifacts/primary.joblib")
    bundle = {**original, "text": rich, "weight": .5, "threshold": selected["threshold"],
              "config": {"rich_text": {"representation": "rich", "c": 16, "nb": False},
                         "boost_iterations": 1000, "boost_depth": 7, "selection": selected}}
    joblib.dump(bundle, ROOT / "artifacts/v2/url_only.joblib", compress=3)
    p, hits = predict_bundle(bundle, test_urls)
    submission = write_submission(ROOT / "data/sample.csv", ROOT / "submissions/submission_url_v2.csv",
                                  (p >= bundle["threshold"]).astype(np.int8))
    results["submission"] = submission
    results["total_seconds"] = time.perf_counter() - start
    (ROOT / "reports/v2/url_model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("V2 URL-only saved:", results, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
