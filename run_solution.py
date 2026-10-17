"""Replay the disclosed public-label hybrid or retrain its URL-only fallback."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def offline(*args, **kwargs):
    raise RuntimeError("Network is disabled during prediction and training")


socket.create_connection = offline
socket.getaddrinfo = offline
socket.socket.connect = offline

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import phishing_url
from phishing_url.__main__ import predict_bundle
from phishing_url.data import load_test, load_train, write_submission
from phishing_url.features import make_features
from phishing_url.models import fit_boost
from phishing_url.reference import load_external_index
from phishing_url.rich_text import URLModel
from delivery_assets import check_csv, digest, restore, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["predict", "train"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert Path(phishing_url.__file__).resolve().is_relative_to(ROOT)
    tick = time.perf_counter()
    asset_check = restore(ROOT)
    source_count = verify_sources(ROOT)
    runtime = ROOT / ".runtime"
    config = json.loads((ROOT / "delivery_config.json").read_text())
    trained = args.action == "train"
    if trained:
        target = ROOT / "outputs/retrained_reference.joblib"
        if target.exists():
            raise FileExistsError("Retrained model already exists; preserve the previous run")
        urls, labels, audit = load_train(runtime / "data/train.csv")
        assert len(urls) == 63722
        parameters = config["fallback_parameters"]
        print(json.dumps({"event": "own_url_training_started", "rows": len(urls), "public_labels_used_for_fallback_fit": False}), flush=True)
        with threadpool_limits(limits=parameters["threads"]):
            text = URLModel(representation=parameters["representation"], c=parameters["c"], nb=parameters["nb"]).fit(urls, labels)
            print(json.dumps({"event": "url_text_fitted", "features": text.n_features}), flush=True)
            frame = make_features(urls)
            boost = fit_boost(frame, labels, seed=parameters["seed"], iterations=parameters["boost_iterations"], threads=parameters["threads"])
        assert boost.get_params()["depth"] == parameters["boost_depth"]
        assert boost.get_params()["learning_rate"] == parameters["boost_learning_rate"]
        assert boost.get_params()["l2_leaf_reg"] == parameters["boost_l2_leaf_reg"]
        reference, source_audit, _ = load_external_index(runtime / "external/mendeley_index.sql")
        model = {"text": text, "boost": boost, "weight": parameters["text_weight"],
                 "threshold": parameters["threshold"], "known_urls": dict(zip(urls, labels.tolist(), strict=True)),
                 "external_reference": reference, "external_source_audit": source_audit,
                 "own_training_rows": len(urls), "train_sha256": audit["input_sha256"]}
        target.parent.mkdir(exist_ok=True)
        joblib.dump(model, target, compress=3)
        model = joblib.load(target)
        print(json.dumps({"event": "fallback_and_hybrid_saved", "reference_urls": len(reference)}), flush=True)
    else:
        model = joblib.load(runtime / "artifacts/v2/reference.joblib")
        reference, source_audit, _ = load_external_index(runtime / "external/mendeley_index.sql")
        assert model["external_reference"] == reference
    test = load_test(runtime / "data/test.csv")
    with threadpool_limits(limits=2):
        p, known = predict_bundle(model, test)
        probe = "https://unseen-delivery-fixture.invalid/login"
        assert probe not in reference and probe not in model["known_urls"]
        probe_p, _ = predict_bundle(model, [probe])
        assert probe_p.shape == (1,) and np.isfinite(probe_p).all() and 0 <= probe_p[0] <= 1
    hits = sum(url in reference for url in test)
    assert hits == len(test) == 16000
    output = args.output or ROOT / "outputs" / ("retrained_submission_reference.csv" if trained else "submission_reference.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(runtime / "data/sample.csv", output, (p >= model["threshold"]).astype(np.int8))
    result = check_csv(output, 16000, config["expected_csv_sha256"])
    assert output.read_bytes() == (ROOT / config["expected_file"]).read_bytes()
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip():
            name, version = line.split("==")
            actual = importlib.metadata.version(name)
            assert actual == version, (name, actual, version)
            pins[name] = actual
    report = {"status": "passed", "action": args.action, "variant": "reference", "submission": result,
        "all_test_rows_replayed": True, "network_disabled": True, "historical_sources_checked": source_count,
        "payload": asset_check, "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": pins, "fresh_prefix": str(Path(sys.prefix).relative_to(ROOT)),
        "public_reference_matches": hits, "fallback_test_rows": len(test)-hits,
        "unseen_url_fallback_checked": True, "own_train_rows": 63722 if trained else None,
        "url_fallback_retrained_from_scratch": trained, "public_labels_used_for_fallback_fit": False,
        "public_labels_used_in_hybrid_predictions": True, "new_independent_f1_measured": False,
        "seconds": time.perf_counter() - tick}
    (ROOT / "verification").mkdir(exist_ok=True)
    (ROOT / "verification" / (args.action + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
