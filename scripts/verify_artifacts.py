"""Verify saved results, split boundaries and byte-identical offline inference."""

import gc
import json
import socket
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phishing_url.__main__ import predict_bundle
from phishing_url.data import load_test, load_train, sha256, write_submission
from phishing_url.evaluation import measure
from phishing_url.features import group_for_url


def main():
    started = time.perf_counter()
    reports = ROOT / "reports"
    urls, labels, audit = load_train(ROOT / "data/train.csv")
    test_urls = load_test(ROOT / "data/test.csv")
    splits = np.load(reports / "split_indices.npz")
    dev, holdout = splits["development"], splits["holdout"]
    assert not set(dev).intersection(holdout)
    assert sorted(np.concatenate([dev, holdout]).tolist()) == list(range(len(urls)))
    assert len(set(urls)) == len(urls)
    grouped = np.load(reports / "domain_split_indices.npz")
    training, validation = grouped["training"], grouped["validation"]
    assert not set(training).intersection(validation)
    assert set(training).union(validation) == set(dev)
    train_groups = {group_for_url(urls[int(i)]) for i in training}
    validation_groups = {group_for_url(urls[int(i)]) for i in validation}
    assert not train_groups.intersection(validation_groups)
    metrics = json.loads((reports / "metrics.json").read_text())
    for filename, expected, indices in [
        ("holdout_predictions.csv", metrics["untouched_holdout"]["primary_ensemble"], holdout),
        ("domain_predictions.csv", metrics["unseen_domain_check"]["primary_ensemble"], validation),
    ]:
        rows = pd.read_csv(reports / filename)
        assert np.array_equal(rows.clean_row.to_numpy(), indices)
        assert rows.url.tolist() == [urls[int(i)] for i in indices]
        assert np.array_equal(rows.truth.to_numpy(), labels[indices])
        score = measure(rows.truth.to_numpy(), rows.probability.to_numpy(), expected["threshold"])
        for name in ["f1", "precision", "recall", "tn", "fp", "fn", "tp"]:
            assert np.isclose(score[name], expected[name], atol=1e-12, rtol=0), (filename, name)
    checks = {"split_indices_cover_clean_train": True, "holdout_disjoint_from_development": True,
              "unique_training_urls": True, "domain_validation_disjoint": True,
              "domain_check_uses_development_only": True, "saved_metrics_recomputed": True,
              "input_sha256": audit["input_sha256"], "test_rows": len(test_urls), "models": {}}
    with tempfile.TemporaryDirectory() as temporary:
        for name in ["primary", "text"]:
            with patch.object(socket, "create_connection", side_effect=AssertionError("Network request")), patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS request")):
                bundle = joblib.load(ROOT / "artifacts" / f"{name}.joblib")
                assert bundle["input_sha256"] == audit["input_sha256"]
                probabilities, overridden = predict_bundle(bundle, test_urls)
                predicted = (probabilities >= bundle["threshold"]).astype(np.int8)
                target = Path(temporary) / f"{name}.csv"
                info = write_submission(ROOT / "data/sample.csv", target, predicted)
            original = ROOT / "submissions" / f"submission_{name}.csv"
            assert target.read_bytes() == original.read_bytes(), f"Prediction changed for {name}"
            by_url = {}
            for url, label in zip(test_urls, predicted, strict=True):
                assert url not in by_url or by_url[url] == label, "Duplicate test URL has different labels"
                by_url[url] = label
            checks["models"][name] = {**info, "byte_identical_reprediction": True,
                                      "inference_network_disabled": True, "consistent_duplicate_urls": True,
                                      "known_url_overrides": overridden,
                                      "artifact_sha256": sha256(ROOT / "artifacts" / f"{name}.joblib")}
            del bundle
            gc.collect()
    checks["verification_seconds"] = time.perf_counter() - started
    (reports / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
