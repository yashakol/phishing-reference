"""Offline external URL reference. Parse SQL values without executing SQL."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path

EXPECTED_SHA256 = "3de752bb74332c8f2c01cbc1355b6eda0e2a6d5bcf9a87fddf9b26b5bc8f1ccc"
EXPECTED_BYTES = 10211200
ROW = re.compile(r"^[ \t]*\((\d+), '((?:\\.|''|[^'])*)', '((?:\\.|''|[^'])*)', ([01]), '([^']*)'\)[,;]\s*$", re.MULTILINE)
ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}


def decode_mysql_string(text):
    text = text.replace("''", "'")
    return re.sub(r"\\(.)", lambda m: ESCAPES.get(m[1], m[1]), text)


def parse_index_text(text):
    rows = []
    for match in ROW.finditer(text):
        record, url, website, label, date = match.groups()
        rows.append({"record_id": int(record), "url": decode_mysql_string(url).strip(),
                     "website": decode_mysql_string(website), "label": int(label), "created_date": date})
    if not rows:
        raise ValueError("No valid index SQL records found")
    # Every tuple line must parse, so a new source format cannot silently lose records.
    candidate_lines = [line for line in text.splitlines() if re.match(r"^[ \t]*\(\d+,", line)]
    if len(rows) != len(candidate_lines):
        raise ValueError(f"Parsed {len(rows)} of {len(candidate_lines)} SQL tuple lines")
    return rows


def load_external_index(path, verify_hash=True):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_hash and (digest != EXPECTED_SHA256 or len(raw) != EXPECTED_BYTES):
        raise ValueError("External index is incomplete or differs from the published SHA-256")
    rows = parse_index_text(raw.decode("utf-8-sig"))
    values = defaultdict(list)
    for row in rows:
        values[row["url"]].append(row["label"])
    conflicts = {url: labels for url, labels in values.items() if len(set(labels)) > 1}
    reference = {url: labels[0] for url, labels in values.items() if len(set(labels)) == 1}
    audit = {"sha256": digest, "bytes": len(raw), "source_records": len(rows),
             "unique_urls": len(values), "nonconflicting_urls": len(reference),
             "conflicting_urls": len(conflicts), "duplicate_records": len(rows) - len(values),
             "conflicting_url_labels": conflicts}
    return reference, audit, rows


def apply_reference(urls, probabilities, reference):
    probabilities = probabilities.copy()
    hits = []
    for i, url in enumerate(urls):
        if url in reference:
            probabilities[i] = reference[url]
            hits.append(i)
    return probabilities, hits
