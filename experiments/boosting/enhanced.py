"""Cross-fitted domain statistics and richer path templates for CatBoost."""

import sys
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from phishing_url.features import make_features, split_url, domain_parts, categorical_columns


def template(text):
    text = text.lower()
    text = re.sub(r"[a-f0-9]{12,}", "HASH", text)
    return re.sub(r"\d+", "NUM", text)[:200] or "EMPTY"


def enriched_features(urls):
    frame = make_features(urls)
    extra = []
    extensions = frame["cat_extension"].to_numpy()
    keyword_signals = frame[[c for c in frame if c.startswith("keyword_")]].sum(axis=1).to_numpy()
    for i, url in enumerate(urls):
        parsed = split_url(url)
        host = (parsed.hostname or "").lower()
        domain, suffix, subdomain, stem, is_ip = domain_parts(host)
        pieces = [p for p in parsed.path.lower().split("/") if p]
        keys = sorted({p.split("=", 1)[0].lower() for p in parsed.query.split("&") if p})
        alpha = re.sub(r"[^a-z]", "", stem)
        extra.append({
            "cat_path_template": template(parsed.path),
            "cat_path_prefix2": template("/".join(pieces[:2])),
            "cat_last_path": template(pieces[-1] if pieces else ""),
            "cat_host_skeleton": template(host),
            "cat_domain_skeleton": template(domain),
            "cat_stem_alpha": alpha or "EMPTY",
            "cat_query_keys": "&".join(keys)[:200] or "EMPTY",
            "cat_path_ext_scheme": extensions[i] + "|" + parsed.scheme.lower(),
            "path_is_root": int(parsed.path in {"", "/"}),
            "has_encoded_email": int("%40" in url.lower() or "@" in parsed.query or "@" in parsed.path),
            "stem_vowel_ratio": sum(c in "aeiou" for c in stem) / max(len(stem), 1),
            "stem_unique_ratio": len(set(stem)) / max(len(stem), 1),
            "path_double_slashes": parsed.path.count("//"),
            "query_key_count": len(keys),
            "total_keyword_signals": int(keyword_signals[i]),
        })
    return pd.concat([frame, pd.DataFrame(extra)], axis=1)


class GroupStatistics:
    columns = ["cat_host", "cat_domain", "cat_path_template", "cat_path_prefix2", "cat_last_path", "cat_domain_skeleton"]

    def fit(self, frame, labels):
        self.prior = float(np.mean(labels))
        self.maps = {}
        for column in self.columns:
            data = pd.DataFrame({"key": frame[column].to_numpy(), "label": labels})
            self.maps[column] = data.groupby("key", sort=False).label.agg(["sum", "count"])
        return self

    def transform(self, frame):
        result = {}
        for column, mapping in self.maps.items():
            count = frame[column].map(mapping["count"]).fillna(0).to_numpy()
            positives = frame[column].map(mapping["sum"]).fillna(0).to_numpy()
            result["stat_" + column + "_known_count"] = np.log1p(count)
            result["stat_" + column + "_rate"] = (positives + self.prior) / (count + 1)
            result["stat_" + column + "_positive_count"] = np.log1p(positives)
            result["stat_" + column + "_negative_count"] = np.log1p(count - positives)
        return pd.DataFrame(result)

    def fit_transform(self, frame, labels):
        result = np.empty((len(labels), len(self.columns) * 4), dtype=np.float64)
        names = None
        for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=311).split(frame, labels):
            encoder = GroupStatistics().fit(frame.iloc[tr], labels[tr])
            values = encoder.transform(frame.iloc[va])
            result[va] = values.to_numpy()
            names = values.columns
        self.fit(frame, labels)
        return pd.DataFrame(result, columns=names)


class EnhancedBoost:
    def __init__(self, iterations=1800, depth=8, threads=2, use_stats=True):
        self.iterations, self.depth, self.threads, self.use_stats = iterations, depth, threads, use_stats

    def fit(self, frame, labels):
        started = time.perf_counter()
        frame = frame.reset_index(drop=True)
        self.statistics = GroupStatistics()
        if self.use_stats:
            added = self.statistics.fit_transform(frame, labels)
            self.model_frame = pd.concat([frame, added], axis=1)
        else:
            self.model_frame = frame
        self.model = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=.045,
                                       l2_leaf_reg=4, random_seed=2026, loss_function="Logloss",
                                       thread_count=self.threads, verbose=False, allow_writing_files=False)
        self.model.fit(self.model_frame, labels, cat_features=categorical_columns(self.model_frame))
        self.fit_seconds = time.perf_counter() - started
        self.model_frame = None
        return self

    def predict_proba(self, frame):
        frame = frame.reset_index(drop=True)
        if self.use_stats:
            frame = pd.concat([frame, self.statistics.transform(frame)], axis=1)
        return self.model.predict_proba(frame)[:, 1]
