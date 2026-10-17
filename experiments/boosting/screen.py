"""Screen new structural models using only the development pool."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from enhanced import EnhancedBoost, enriched_features
from phishing_url.data import load_train
from phishing_url.evaluation import measure, best_threshold
from phishing_url.models import TextModel, fit_boost, boost_probability


def log(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def main():
    folder = Path(__file__).resolve().parent
    urls, y, _ = load_train(ROOT / "data/train.csv")
    dev = np.load(ROOT / "reports/split_indices.npz")["development"]
    train, validation = train_test_split(dev, test_size=.2, stratify=y[dev], random_state=20261006)
    np.savez_compressed(folder / "screen_split.npz", training=train, validation=validation, labels=y[validation])
    cache = folder / "features.joblib"
    if cache.exists():
        frame = joblib.load(cache)
    else:
        log("Extracting richer structural features")
        frame = enriched_features(urls)
        joblib.dump(frame, cache, compress=3)
    tr, va = frame.iloc[train], frame.iloc[validation]
    log("Baseline text on the same internal split")
    text = TextModel().fit([urls[int(i)] for i in train], y[train])
    ptext = text.predict_proba([urls[int(i)] for i in validation])
    baseline_columns = [c for c in frame.columns if c in __import__('phishing_url.features',fromlist=['feature_row']).feature_row(urls[0])]
    log("Baseline CatBoost on the same internal split")
    base = fit_boost(tr[baseline_columns], y[train], threads=2)
    pbase = boost_probability(base, va[baseline_columns])
    probabilities = {"text":ptext, "baseline_boost":pbase, "baseline_blend":.5*ptext+.5*pbase}
    records = []
    for name, p in probabilities.items():
        threshold, tuned = best_threshold(y[validation], p)
        records.append({"model":name,**measure(y[validation],p,threshold)})
    for name, params in [("richer_plain",dict(use_stats=False,iterations=1800)),
                         ("richer_crossfit_stats",dict(use_stats=True,iterations=1800))]:
        log("Fitting " + name)
        model = EnhancedBoost(**params).fit(tr, y[train])
        p = model.predict_proba(va)
        probabilities[name] = p
        threshold, tuned = best_threshold(y[validation],p)
        records.append({"model":name,"fit_seconds":model.fit_seconds,**measure(y[validation],p,threshold)})
        for weight in [.25,.5,.75]:
            blend = weight * ptext + (1-weight)*p
            blend_name = name+f"_text{weight}"
            probabilities[blend_name] = blend
            threshold, _ = best_threshold(y[validation], blend)
            records.append({"model":blend_name,**measure(y[validation],blend,threshold)})
        pd.DataFrame(records).to_csv(folder / "screen_metrics.csv",index=False)
        np.savez_compressed(folder / "screen_probabilities.npz", clean_indices=validation, labels=y[validation], **probabilities)
        joblib.dump(model, folder / (name+".joblib"), compress=3)
        log(name + " tuned screen F1=" + str(tuned))
    log(pd.DataFrame(records).sort_values('f1',ascending=False).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
