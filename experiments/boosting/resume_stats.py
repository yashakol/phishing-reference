"""Resume the statistics candidate using the already saved screening split."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from enhanced import EnhancedBoost
from phishing_url.data import load_train
from phishing_url.evaluation import best_threshold, measure


def main():
    folder = Path(__file__).resolve().parent
    urls, labels, _ = load_train(ROOT / "data/train.csv")
    split = np.load(folder / "screen_split.npz")
    train, validation = split["training"], split["validation"]
    frame = joblib.load(folder / "features.joblib")
    existing = np.load(folder / "screen_probabilities.npz")
    probabilities = {name: existing[name] for name in existing.files if name not in {"clean_indices", "labels"}}
    records = pd.read_csv(folder / "screen_metrics.csv").to_dict("records")
    print("Fitting corrected cross-fitted statistics candidate", flush=True)
    model = EnhancedBoost(iterations=1800,use_stats=True).fit(frame.iloc[train], labels[train])
    probability = model.predict_proba(frame.iloc[validation])
    threshold, score = best_threshold(labels[validation], probability)
    name = "richer_crossfit_stats"
    probabilities[name] = probability
    records.append({"model":name,"fit_seconds":model.fit_seconds,**measure(labels[validation],probability,threshold)})
    for weight in [.25,.5,.75]:
        blend = weight * probabilities["text"] + (1-weight)*probability
        threshold, _ = best_threshold(labels[validation],blend)
        blend_name = name+f"_text{weight}"
        probabilities[blend_name] = blend
        records.append({"model":blend_name,**measure(labels[validation],blend,threshold)})
    pd.DataFrame(records).to_csv(folder / "screen_metrics.csv",index=False)
    np.savez_compressed(folder / "screen_probabilities.npz",clean_indices=validation,labels=labels[validation],**probabilities)
    joblib.dump(model,folder / (name+".joblib"),compress=3)
    print(pd.DataFrame(records).sort_values("f1",ascending=False).to_string(index=False),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
