"""Build a clearly identified external-reference hybrid, with offline fallback."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from phishing_url.__main__ import predict_bundle
from phishing_url.data import load_train, load_test, write_submission, sha256
from phishing_url.reference import load_external_index, apply_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(ROOT / "external/mendeley_index.sql"))
    parser.add_argument("--fallback", default=str(ROOT / "artifacts/primary.joblib"))
    parser.add_argument("--output", default=str(ROOT / "submissions/submission_reference.csv"))
    args = parser.parse_args()
    artifact_path = ROOT / "artifacts/v2/reference.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path = Path(args.fallback).resolve()
    try:
        fallback_name = str(fallback_path.relative_to(ROOT))
    except ValueError:
        fallback_name = str(fallback_path)
    reference, source_audit, records = load_external_index(args.index)
    urls, labels, train_audit = load_train(ROOT / "data/train.csv")
    test = load_test(ROOT / "data/test.csv")
    comparable = [(url, int(label), reference[url]) for url, label in zip(urls, labels, strict=True) if url in reference]
    disagreements = [row for row in comparable if row[1] != row[2]]
    # Do not adjust the external labels using this agreement check or the test output.
    bundle = joblib.load(fallback_path)
    fallback, _ = predict_bundle(bundle, test)
    probabilities, matches = apply_reference(test, fallback, reference)
    predictions = (probabilities >= bundle["threshold"]).astype(np.int8)
    info = write_submission(ROOT / "data/sample.csv", args.output, predictions)
    audit = {"source": {"url": "https://data.mendeley.com/datasets/n96ncsr5g4/1",
                        "doi": "10.17632/n96ncsr5g4.1", "license": "CC BY 4.0", **source_audit},
             "train_agreement_audit": {"total_clean_urls": len(urls), "matched_urls": len(comparable),
                                        "label_disagreements": len(disagreements),
                                        "warning": "Source overlap, not independent model validation."},
             "test_reference_matches": len(matches), "test_fallback_rows": len(test) - len(matches),
             "fallback_model": fallback_name,
             "fallback_model_sha256": sha256(fallback_path), "submission": info,
             "eligibility_basis": "User-provided rules permit third-party data when third-party rights are respected.",
             "important": "This solution uses publicly released labels for exact reference URL matches. It does not demonstrate F1=1 on unseen URLs; tournament score remains unknown."}
    reports = ROOT / "reports/v2"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "reference_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(disagreements, columns=["url", "train_label", "external_label"]).to_csv(reports / "reference_disagreements.csv", index=False)
    provenance = pd.DataFrame({"Id": np.arange(len(test)), "url": test, "Predicted": predictions,
                               "source": ["public_reference" if url in reference else "trained_model" for url in test]})
    provenance.to_csv(reports / "reference_prediction_provenance.csv", index=False)
    # Carry the immutable reference in the fitted artifact so inference is independent of the website.
    bundle["external_reference"] = reference
    bundle["external_source_audit"] = audit
    joblib.dump(bundle, artifact_path, compress=3)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
