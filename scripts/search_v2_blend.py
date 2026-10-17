"""Repeat the OOF ensemble search while checking exact row alignment."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from phishing_url.evaluation import best_threshold, measure


def main():
    (ROOT / "reports/v2").mkdir(parents=True, exist_ok=True)
    dev = np.load(ROOT / "reports/split_indices.npz")["development"]
    base = np.load(ROOT / "reports/development_oof.npz")
    rich = np.load(ROOT / "experiments/astra/development_oof.npz")
    assert np.array_equal(rich["indices"],dev)
    assert np.array_equal(rich["y"],base["labels"])
    y=base["labels"]
    results=[]
    for rich_weight in [.25,.5,.75,1.0]:
        for boost_weight in [0,.25,.5,.75,1.0]:
            if rich_weight+boost_weight>1:
                continue
            old_text_weight=1-rich_weight-boost_weight
            p=rich_weight*rich["probabilities"]+boost_weight*base["boost"]+old_text_weight*base["tfidf"]
            threshold,_=best_threshold(y,p)
            results.append({"rich_weight":rich_weight,"boost_weight":boost_weight,
                            "old_text_weight":old_text_weight,**measure(y,p,threshold)})
    results.sort(key=lambda row:row["f1"],reverse=True)
    (ROOT / "reports/v2/rich_blend_search.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results[0],indent=2))


if __name__ == "__main__":
    main()
