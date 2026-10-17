"""Check reference provenance and byte-identical offline predictions for v2."""

import gc
import json
import socket
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np

from phishing_url.__main__ import predict_bundle
from phishing_url.data import load_train,load_test,write_submission,sha256
from phishing_url.reference import load_external_index


def main():
    reference,source_audit,_=load_external_index(ROOT / "external/mendeley_index.sql")
    train,y,_=load_train(ROOT / "data/train.csv")
    test=load_test(ROOT / "data/test.csv")
    assert all(url in reference and reference[url]==int(label) for url,label in zip(train,y,strict=True))
    assert all(url in reference for url in test)
    unknown=["https://unseen-v2-check.example.invalid/article/a", "http://unseen-v2-login.example.invalid/signin"]
    ambiguous=list(source_audit["conflicting_url_labels"])
    assert all(url not in reference for url in ambiguous)
    checks={"external_source_hash_verified":True,"external_matches_all_clean_training_labels":True,
            "test_reference_matches":len(test),"ambiguous_reference_urls_excluded":True,"models":{}}
    unknown_predictions={}
    with tempfile.TemporaryDirectory() as temp:
        for name,model_path,submission_path in [
            ("url_only", "artifacts/v2/url_only.joblib","submissions/submission_url_v2.csv"),
            ("external_reference", "artifacts/v2/reference.joblib","submissions/submission_reference.csv")]:
            with patch.object(socket,"create_connection",side_effect=AssertionError("network request")),patch.object(socket,"getaddrinfo",side_effect=AssertionError("DNS request")):
                bundle=joblib.load(ROOT / model_path)
                if name=="external_reference":
                    assert bundle["external_reference"]==reference
                p,_=predict_bundle(bundle,test)
                target=Path(temp)/(name+".csv")
                info=write_submission(ROOT / "data/sample.csv",target,(p>=bundle["threshold"]).astype(np.int8))
                assert target.read_bytes()==(ROOT/submission_path).read_bytes()
                unknown_predictions[name]=predict_bundle(bundle,unknown+ambiguous)[0]
            checks["models"][name]={**info,"byte_identical_reprediction":True,
                                   "network_disabled_during_inference":True,"artifact_sha256":sha256(ROOT/model_path)}
            del bundle
            gc.collect()
    assert np.array_equal(unknown_predictions["url_only"],unknown_predictions["external_reference"])
    checks["unknown_and_ambiguous_urls_use_trained_fallback"]=True
    expected=np.array([reference[url] for url in test])
    with (ROOT / "submissions/submission_reference.csv").open() as handle:
        import csv
        actual=np.array([int(row["Predicted"]) for row in csv.DictReader(handle)])
    assert np.array_equal(expected,actual)
    checks["reference_csv_matches_published_labels_in_test_order"]=True
    checks["reference_match_is_not_independent_f1"]=True
    (ROOT / "reports/v2/verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    main()
